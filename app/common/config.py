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

# Consumer groups - Pod별로 고유한 Group ID 사용
def get_pod_unique_group_id(base_name: str) -> str:
    """Pod별로 고유한 Consumer Group ID 생성"""
    pod_name = os.getenv("HOSTNAME", "unknown")
    return f"{base_name}-{pod_name}"

GROUP_BRAINWAVE_SPLITTER = get_pod_unique_group_id(env_str("GROUP_BRAINWAVE_SPLITTER", "brainwave-splitter"))
GROUP_BRAINWAVE_ANALYZER = get_pod_unique_group_id(env_str("GROUP_BRAINWAVE_ANALYZER", "brainwave-analyzer"))
GROUP_BRAINWAVE_AGGREGATOR = get_pod_unique_group_id(env_str("GROUP_BRAINWAVE_AGGREGATOR", "brainwave-aggregator"))
GROUP_BRAINWAVE_DB_WRITER = get_pod_unique_group_id(env_str("GROUP_BRAINWAVE_DB_WRITER", "brainwave-db-writer"))

# Sound topics
TOPIC_SOUND_INPUT_RAW = env_str("TOPIC_SOUND_INPUT_RAW", "sound.input.raw")
TOPIC_SOUND_SPLIT_EPOCHS = env_str("TOPIC_SOUND_SPLIT_EPOCHS", "sound.split.epochs")
TOPIC_SOUND_ANALYZED_EVENT = env_str("TOPIC_SOUND_ANALYZED_EVENT", "sound.analyzed.event")
TOPIC_SOUND_PERSIST_REQUESTS = env_str("TOPIC_SOUND_PERSIST_REQUESTS", "sound.persist.requests")

# Sound consumer groups
GROUP_SOUND_SPLITTER = env_str("GROUP_SOUND_SPLITTER", "sound-splitter")
GROUP_SOUND_ANALYZER = env_str("GROUP_SOUND_ANALYZER", "sound-analyzer")
GROUP_SOUND_DB_WRITER = env_str("GROUP_SOUND_DB_WRITER", "sound-db-writer")
GROUP_SOUND_AGGREGATOR = env_str("GROUP_SOUND_AGGREGATOR", "sound-aggregator")

# Retry
RETRY_MAX_ATTEMPTS = env_int("RETRY_MAX_ATTEMPTS", 3)
RETRY_BACKOFF_MS = env_int("RETRY_BACKOFF_MS", 200)


# Sound analysis
SOUND_YAMNET_MODEL_URL = env_str("SOUND_YAMNET_MODEL_URL", "https://tfhub.dev/google/yamnet/1")
SOUND_YAMNET_CLASS_MAP_URL = env_str(
    "SOUND_YAMNET_CLASS_MAP_URL",
    "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv",
)
SOUND_YAMNET_CLASS_MAP_FILENAME = env_str(
    "SOUND_YAMNET_CLASS_MAP_FILENAME",
    "yamnet_class_map.csv",
)

