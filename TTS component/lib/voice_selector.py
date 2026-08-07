import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class VoiceSelector:

    def __init__(self, config: dict):
        self._voice_cfg = config.get("voices", {})
        self._default_gender = config.get("audio", {}).get("unknown_gender_default", "male")

    def select(
        self,
        gender: str,
        provider_name: str,
    ) -> Tuple[str, str, Optional[str]]:
        warnings = []

        if gender not in ("male", "female"):
            warnings.append(
                f"Gender '{gender}' is not recognised. "
                f"Defaulting to '{self._default_gender}'."
            )
            gender = self._default_gender

        provider_voices = self._voice_cfg.get(provider_name, {})
        gender_voices = provider_voices.get(gender, [])

        if not gender_voices:
            alt_gender = "female" if gender == "male" else "male"
            gender_voices = provider_voices.get(alt_gender, [])
            if gender_voices:
                warnings.append(
                    f"No {gender} voices configured for {provider_name}. "
                    f"Using {alt_gender} voice as fallback."
                )
                gender = alt_gender

        if not gender_voices:
            raise ValueError(
                f"No voices at all configured for provider '{provider_name}'. "
                f"Check config/config.yaml -> voices -> {provider_name}."
            )

        voice_id = gender_voices[0]
        warning = "; ".join(warnings) if warnings else None

        return voice_id, gender, warning

    def list_all(self, provider_name: str) -> dict:
        return self._voice_cfg.get(provider_name, {})
