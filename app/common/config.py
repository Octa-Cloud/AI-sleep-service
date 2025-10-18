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
TOPIC_REPORT_DLQ = env_str("TOPIC_REPORT_DLQ", "report.dlq")

# Brainwave topics
TOPIC_BRAINWAVE_INPUT_RAW = env_str("TOPIC_BRAINWAVE_INPUT_RAW", "brainwave.input.raw")
TOPIC_BRAINWAVE_SPLIT_EPOCHS = env_str("TOPIC_BRAINWAVE_SPLIT_EPOCHS", "brainwave.split.epochs")
TOPIC_BRAINWAVE_ANALYZED_EPOCH = env_str("TOPIC_BRAINWAVE_ANALYZED_EPOCH", "brainwave.analyzed.epoch")
TOPIC_BRAINWAVE_PERSIST_REQUESTS = env_str("TOPIC_BRAINWAVE_PERSIST_REQUESTS", "brainwave.persist.requests")

# Report topics (daily / periodic)
TOPIC_DAILY_REPORT_INPUT = env_str("TOPIC_DAILY_REPORT_INPUT", "daily.report.input")
TOPIC_DAILY_REPORT_PERSIST_REQUESTS = env_str("TOPIC_DAILY_REPORT_PERSIST_REQUESTS", "daily.report.persist.requests")

TOPIC_PERIODIC_REPORT_INPUT = env_str("TOPIC_PERIODIC_REPORT_INPUT", "periodic.report.input")
TOPIC_PERIODIC_REPORT_PERSIST_REQUESTS = env_str("TOPIC_PERIODIC_REPORT_PERSIST_REQUESTS", "periodic.report.persist.requests")

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

# Report consumer groups
GROUP_DAILY_REPORT_WORKER = get_pod_unique_group_id(env_str("GROUP_DAILY_REPORT_WORKER", "daily-report-worker"))
GROUP_PERIODIC_REPORT_WORKER = get_pod_unique_group_id(env_str("GROUP_PERIODIC_REPORT_WORKER", "periodic-report-worker"))
GROUP_DAILY_REPORT_DB_WRITER = get_pod_unique_group_id(env_str("GROUP_DAILY_REPORT_DB_WRITER", "daily-report-db-writer"))
GROUP_PERIODIC_REPORT_DB_WRITER = get_pod_unique_group_id(env_str("GROUP_PERIODIC_REPORT_DB_WRITER", "periodic-report-db-writer"))

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

# Reporting cutoff (local hour / timezone)
REPORT_CUTOFF_HOUR_LOCAL = env_int("REPORT_CUTOFF_HOUR_LOCAL", 0)  # default 00:00
REPORT_CUTOFF_TZ = env_str("REPORT_CUTOFF_TZ", "Asia/Seoul")

# Azure Agent / Project (explicit; no fallback)
AZURE_DAILY_PROJECT_ENDPOINT = env_str("AZURE_DAILY_REPORT_PROJECT_ENDPOINT", "")
AZURE_DAILY_AGENT_ID = env_str("AZURE_DAILY_REPORT_AGENT_ID", "")
AZURE_PERIODIC_PROJECT_ENDPOINT = env_str("AZURE_PERIODIC_REPORT_PROJECT_ENDPOINT", "")
AZURE_PERIODIC_AGENT_ID = env_str("AZURE_PERIODIC_REPORT_AGENT_ID", "")

# Azure credentials (EnvironmentCredential / ClientCertificateCredential support)
AZURE_CLIENT_ID = env_str("AZURE_CLIENT_ID", "")
AZURE_TENANT_ID = env_str("AZURE_TENANT_ID", "")
AZURE_CLIENT_CERTIFICATE_PATH = env_str("AZURE_CLIENT_CERTIFICATE_PATH", "")

