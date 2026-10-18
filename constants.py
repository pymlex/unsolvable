from __future__ import annotations

from utils.config_loader import env_float, env_int, env_str


DATASET_ID = env_str("DATASET_ID", "pymlex/reshuege-ru-2026")
DATASET_SPLIT = env_str("DATASET_SPLIT", "train")

SLM_MODEL_ID = env_str("SLM_MODEL_ID", "Qwen/Qwen3.5-2B")
ZVENO_BASE_URL = env_str("ZVENO_BASE_URL", "https://api.zveno.ai/v1")
ZVENO_MODEL_ID = env_str("ZVENO_MODEL_ID", "deepseek/deepseek-v4-flash-0731")

MAX_NEW_TOKENS = env_int("MAX_NEW_TOKENS", 32)
BATCH_SIZE = env_int("BATCH_SIZE", 8)
GENERATION_TEMPERATURE = env_float("GENERATION_TEMPERATURE", 0.8)
GENERATION_BATCH_SIZE = env_int("GENERATION_BATCH_SIZE", 4)
PARAPHRASES_PER_TEXT = env_int("PARAPHRASES_PER_TEXT", 4)

LENGTH_MARGIN = env_float("LENGTH_MARGIN", 0.20)
SCORE_MARGIN_DELTA = env_float("SCORE_MARGIN_DELTA", 0.5)
PAIRWISE_GAP_TAU = env_float("PAIRWISE_GAP_TAU", 0.25)
ANALYZE_MARGIN_SAMPLES = env_int("ANALYZE_MARGIN_SAMPLES", 32)

GRPO_GROUP_SIZE = env_int("GRPO_GROUP_SIZE", 4)
GRPO_EPOCHS = env_int("GRPO_EPOCHS", 1)
GRPO_LEARNING_RATE = env_float("GRPO_LEARNING_RATE", 5e-6)
GRPO_PER_DEVICE_BATCH_SIZE = env_int("GRPO_PER_DEVICE_BATCH_SIZE", 1)
GRPO_GRADIENT_ACCUMULATION_STEPS = env_int("GRPO_GRADIENT_ACCUMULATION_STEPS", 8)
GRPO_MAX_STEPS = env_int("GRPO_MAX_STEPS", 50)

SYNTHESIS_N_ITEMS = env_int("SYNTHESIS_N_ITEMS", 500)
SYNTHESIS_FEWSHOT_K = env_int("SYNTHESIS_FEWSHOT_K", 3)
SYNTHESIS_SPLIT_NAME = env_str(
    "SYNTHESIS_SPLIT_NAME",
    "train-deepseek-v4-flash-0731",
)

HF_DATASET_REPO = env_str("HF_DATASET_REPO", "pymlex/reshuege-ru-2026")
GITHUB_RESULTS_PATHS = env_str(
    "GITHUB_RESULTS_PATHS",
    "results/benchmark/ results/preferences/ results/plots/ results/groups/ results/synthesis/",
)
