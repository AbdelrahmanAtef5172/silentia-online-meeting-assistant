from typing import Optional
from lib.schemas import TTSRequest, SynthesisResult
from lib.service import TTSService


class PipelineAdapter:

    def __init__(self, service: TTSService):
        self._service = service

    def process(
        self,
        llm_result,
        vision_result,
        output_mode: str = "play",
        session_id: str = None,
        output_path: Optional[str] = None,
        speed: float = 1.0,
    ) -> SynthesisResult:
        text = getattr(llm_result, "text_for_tts", None) or getattr(llm_result, "corrected_text", "")

        raw_gender = getattr(vision_result, "label", None)
        if raw_gender is not None:
            gender_str = getattr(raw_gender, "value", str(raw_gender))
        else:
            gender_str = "unknown"

        if gender_str == "no_face":
            gender_str = "unknown"

        request = TTSRequest(
            text=text,
            gender=gender_str,
            output_mode=output_mode,
            session_id=session_id,
            output_path=output_path,
            speed=speed,
        )

        return self._service.synthesize_request(request)
