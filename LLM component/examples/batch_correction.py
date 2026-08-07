"""
examples/batch_correction.py
─────────────────────────────
How to process a list of texts efficiently using correct_batch.
"""

from engine.service import CorrectionService
from engine.schemas import CorrectionContext

service = CorrectionService.from_config()

texts = [
    "she go store yesterday buy apple",
    "i want eat pizza my friend",
    "he go school every day",
    "they meet yesterday morning discuss project",
    "doctor prescribe medicine patient",
]

results = service.correct_batch(texts, max_concurrency=4)

for original, result in zip(texts, results):
    print(f"IN:  {original}")
    print(f"OUT: {result.text_for_tts}")
    print(f"     [{result.status.value}] {result.latency_ms:.0f}ms via {result.provider_used or 'n/a'}")
    print()

stats = service._cache.stats()
print(f"Cache stats: {stats['hits']} hits, {stats['misses']} misses, "
      f"hit rate {stats['hit_rate'] * 100:.0f}%")

# With per-item contexts
contexts = [
    CorrectionContext(topic_domain="casual"),
    CorrectionContext(topic_domain="casual"),
    CorrectionContext(topic_domain="casual"),
    CorrectionContext(topic_domain="casual"),
    CorrectionContext(topic_domain="medical"),
]
results_with_contexts = service.correct_batch(texts, contexts=contexts)
print("\nWith domain contexts:")
for text, result in zip(texts, results_with_contexts):
    print(f"  {text[:30]:30s} → {result.text_for_tts}")
