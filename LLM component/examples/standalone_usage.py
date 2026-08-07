"""
examples/standalone_usage.py
────────────────────────────
How to use CorrectionService directly in Python.
"""

from engine.service import CorrectionService
from engine.schemas import CorrectionContext

service = CorrectionService.from_config()

result = service.correct("she go store yesterday buy apple")
print(result.text_for_tts)
# → "She went to the store yesterday to buy apples."

result = service.correct("i want eat pizza my friend")
print(result.text_for_tts)
# → "I want to eat pizza with my friend."

context = CorrectionContext(
    topic_domain="medical",
    speaker_gender="female",
)
result = service.correct("patient show symptom high fever three day", context)
print(result.text_for_tts)
# → "The patient has been showing symptoms of high fever for three days."

print(f"Status:   {result.status.value}")
print(f"Provider: {result.provider_used}")
print(f"Latency:  {result.latency_ms:.0f}ms")
print(f"From cache: {result.from_cache}")
