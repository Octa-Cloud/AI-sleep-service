from __future__ import annotations

import os


def env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default


def env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


# Kafka
KAFKA_BROKERS = env_str("KAFKA_BROKERS", "kafka:9092")
KAFKA_ENABLED = env_bool("KAFKA_ENABLED", True)
KAFKA_PROTOBUF_ENABLED = env_bool("KAFKA_PROTOBUF_ENABLED", False)
TOPIC_DLQ = env_str("TOPIC_DLQ", "brainwave.dlq")

# Brainwave topics
TOPIC_BRAINWAVE_INPUT_RAW = env_str("TOPIC_BRAINWAVE_INPUT_RAW", "brainwave.input.raw")
TOPIC_BRAINWAVE_SPLIT_EPOCHS = env_str("TOPIC_BRAINWAVE_SPLIT_EPOCHS", "brainwave.split.epochs")
TOPIC_BRAINWAVE_ANALYZED_EPOCH = env_str("TOPIC_BRAINWAVE_ANALYZED_EPOCH", "brainwave.analyzed.epoch")
TOPIC_BRAINWAVE_PERSIST_REQUESTS = env_str("TOPIC_BRAINWAVE_PERSIST_REQUESTS", "brainwave.persist.requests")

# Consumer groups
GROUP_BRAINWAVE_SPLITTER = env_str("GROUP_BRAINWAVE_SPLITTER", "brainwave-splitter")
GROUP_BRAINWAVE_ANALYZER = env_str("GROUP_BRAINWAVE_ANALYZER", "brainwave-analyzer")
GROUP_BRAINWAVE_AGGREGATOR = env_str("GROUP_BRAINWAVE_AGGREGATOR", "brainwave-aggregator")
GROUP_BRAINWAVE_DB_WRITER = env_str("GROUP_BRAINWAVE_DB_WRITER", "brainwave-db-writer")

# Retry
RETRY_MAX_ATTEMPTS = env_int("RETRY_MAX_ATTEMPTS", 3)
RETRY_BACKOFF_MS = env_int("RETRY_BACKOFF_MS", 200)


