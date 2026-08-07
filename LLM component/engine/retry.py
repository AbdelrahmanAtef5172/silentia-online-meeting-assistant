import functools
import time
import logging

logger = logging.getLogger(__name__)


def with_retry(max_retries: int = 3, base_delay: float = 1.0, backoff: float = 2.0):
    """
    Exponential backoff retry decorator.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay:  Initial delay in seconds
        backoff:     Multiplier applied to delay after each retry
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = base_delay
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            "%s attempt %d/%d failed: %s. Retrying in %.1fs...",
                            func.__name__, attempt + 1, max_retries, e, delay,
                        )
                        time.sleep(delay)
                        delay *= backoff
            raise last_exception
        return wrapper
    return decorator
