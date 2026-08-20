"""Structured JSON logging with structlog."""

import logging
import os
import sys

import structlog


def setup_logging(log_level: str | None = None):
    """
    Configure structlog to emit one JSON object per line on stdout.

    Reads LOG_LEVEL from the environment when no level is passed — the variable
    is documented in .env.example and every caller invokes this with no argument,
    so taking it only as a parameter meant it was never actually applied.
    """
    level_name = (log_level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    # Log lines carry en-dashes and arrows. A Windows console defaults to
    # cp1252, where writing those raises UnicodeEncodeError inside logging — so
    # a local `uvicorn api.main:app` prints "--- Logging error ---" instead of
    # the feed. Containers are already UTF-8; this only matters off Linux.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        # Must be the stdlib factory, not PrintLoggerFactory: the processor list
        # above includes stdlib.add_logger_name, which reads `logger.name`, and a
        # PrintLogger has no such attribute. The pairing raised AttributeError on
        # the *first* log call after setup — which is inside the API's own
        # lifespan startup, so the service could not boot at all. It is also what
        # makes the basicConfig() above meaningful: records route through stdlib
        # logging rather than bypassing it.
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
