"""Public, data-only multimodal analysis boundary for installed services.

This module exposes existing Hermes transcription, vision and auxiliary-model
routing without granting plugins the agent loop, credentials, or mutable
conversation state. Results remain proposals; callers own domain validation
and confirmation.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class AnalysisSource:
    source_id: str
    kind: str  # audio | photo
    local_path: Path
    content_type: str


@dataclass(frozen=True)
class AnalysisRequest:
    request_id: str
    text: str = ""
    sources: tuple[AnalysisSource, ...] = ()
    schema: Mapping[str, Any] | None = None
    instructions: str = ""


Transcriber = Callable[[str, Optional[str]], Mapping[str, Any]]
VisionAnalyzer = Callable[[str, str], Awaitable[str]]
Extractor = Callable[[Sequence[Mapping[str, Any]], Mapping[str, Any], str], Awaitable[Mapping[str, Any]]]


class AnalysisValidationError(ValueError):
    pass


def _json_object(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        raise AnalysisValidationError("analysis provider did not return an object")
    raw = value.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise AnalysisValidationError("analysis provider did not return a JSON object")
    return parsed


async def _default_extract(
    evidence: Sequence[Mapping[str, Any]], schema: Mapping[str, Any], instructions: str
) -> Mapping[str, Any]:
    from agent.auxiliary_client import async_call_llm

    prompt = (
        "Return JSON only. Extract proposals from the supplied evidence under the supplied schema. "
        "Never invent missing facts. Every fact must contain provenance and confidence; use unknown or "
        "needs-review for uncertainty. Distinguish visible photo observations from inference.\n"
        f"Instructions: {instructions}\nSchema: {json.dumps(schema, ensure_ascii=False)}\n"
        f"Evidence: {json.dumps(list(evidence), ensure_ascii=False)}"
    )
    response = await async_call_llm(
        task="analysis",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=4000,
        timeout=90,
    )
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        content = response
    return _json_object(content)


class AnalysisService:
    def __init__(
        self,
        *,
        transcriber: Transcriber | None = None,
        vision_analyzer: VisionAnalyzer | None = None,
        extractor: Extractor | None = None,
    ) -> None:
        self._transcriber = transcriber
        self._vision_analyzer = vision_analyzer
        self._extractor = extractor or _default_extract

    async def analyze(self, request: AnalysisRequest) -> Mapping[str, Any]:
        if not request.request_id.strip() or (not request.text.strip() and not request.sources):
            raise AnalysisValidationError("request identity and evidence are required")
        if request.schema is None:
            raise AnalysisValidationError("an explicit output schema is required")

        evidence: list[Mapping[str, Any]] = []
        if request.text.strip():
            evidence.append({"source": "text", "source_id": "original", "text": request.text})

        for source in request.sources:
            path = Path(source.local_path).resolve()
            if not path.is_file():
                raise AnalysisValidationError(f"source does not exist: {source.source_id}")
            if source.kind == "audio":
                transcriber = self._transcriber
                if transcriber is None:
                    from tools.transcription_tools import transcribe_audio
                    transcriber = transcribe_audio
                result = await asyncio.to_thread(transcriber, str(path), None)
                if not result.get("success"):
                    raise RuntimeError(str(result.get("error") or "transcription failed"))
                evidence.append({"source": "transcript", "source_id": source.source_id, "text": str(result.get("transcript") or ""), "detected_language": result.get("language")})
            elif source.kind == "photo":
                analyzer = self._vision_analyzer
                if analyzer is None:
                    from tools.vision_tools import vision_analyze_tool
                    analyzer = vision_analyze_tool
                raw = await analyzer(str(path), request.instructions)
                vision = _json_object(raw)
                if vision.get("success") is False:
                    raise RuntimeError(str(vision.get("error") or vision.get("analysis") or "vision failed"))
                evidence.append({"source": "photo", "source_id": source.source_id, "observation": vision.get("analysis", vision)})
            else:
                raise AnalysisValidationError(f"unsupported source kind: {source.kind}")

        result = await self._extractor(evidence, request.schema, request.instructions)
        if not isinstance(result, Mapping):
            raise AnalysisValidationError("extractor returned invalid result")
        return {"request_id": request.request_id, "evidence": evidence, "draft": dict(result)}
