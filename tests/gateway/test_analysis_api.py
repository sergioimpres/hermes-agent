from pathlib import Path

import pytest

from gateway.analysis_api import AnalysisRequest, AnalysisService, AnalysisSource, AnalysisValidationError


@pytest.mark.asyncio
async def test_multilingual_text_and_audio_are_forwarded_without_ui_language(tmp_path: Path):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"audio")
    captured = {}

    def transcribe(path, model):
        assert model is None  # provider autodetection, no UI-language hint
        return {"success": True, "transcript": "Martín trajo самокат", "language": "mixed"}

    async def extract(evidence, schema, instructions):
        captured["evidence"] = evidence
        return {"facts": [{"field": "customer.name", "value": "Martín", "status": "observed", "confidence": 0.99, "provenance": [{"source": "transcript", "source_id": "audio-1"}]}]}

    result = await AnalysisService(transcriber=transcribe, extractor=extract).analyze(AnalysisRequest(request_id="r1", text="Клиент Martín", sources=(AnalysisSource("audio-1", "audio", audio, "audio/wav"),), schema={"type": "object"}))
    assert result["draft"]["facts"][0]["value"] == "Martín"
    assert [item["source"] for item in captured["evidence"]] == ["text", "transcript"]


@pytest.mark.asyncio
async def test_vision_is_visible_evidence_not_a_hidden_diagnosis(tmp_path: Path):
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"photo")

    async def vision(path, prompt):
        return '{"success":true,"analysis":"red folding scooter; brand unreadable; visible tyre flat"}'

    async def extract(evidence, schema, instructions):
        assert "brand unreadable" in evidence[0]["observation"]
        return {"facts": [{"field": "asset.brand", "value": None, "status": "unknown", "provenance": [{"source": "photo", "source_id": "photo-1"}]}]}

    result = await AnalysisService(vision_analyzer=vision, extractor=extract).analyze(AnalysisRequest(request_id="r2", sources=(AnalysisSource("photo-1", "photo", photo, "image/jpeg"),), schema={"type": "object"}, instructions="Report only visible details"))
    assert result["draft"]["facts"][0]["status"] == "unknown"


@pytest.mark.asyncio
async def test_invalid_source_and_provider_failure_fail_closed(tmp_path: Path):
    service = AnalysisService(extractor=lambda *_: None)
    with pytest.raises(AnalysisValidationError):
        await service.analyze(AnalysisRequest(request_id="r3", text="hello"))
    with pytest.raises(AnalysisValidationError):
        await service.analyze(AnalysisRequest(request_id="r4", sources=(AnalysisSource("x", "video", tmp_path / "missing", "video/mp4"),), schema={}))
