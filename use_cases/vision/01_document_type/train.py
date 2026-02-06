"""Fine-tune VLM student model for document type classification.

Follows the pattern from tinker_cookbook/recipes/vlm_classifier/train.py.
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tinker_cookbook import cli_utils
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from shared.import_utils import load_config

CONFIG = load_config(__file__)
DATA_PATH = "/tmp/tinker-datasets/document_type.jsonl"


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}\nRun create_data.py first.")

    model_name_slug = CONFIG.student_model.replace("/", "-")
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"document_type-{model_name_slug}-{CONFIG.lora_rank}rank-{CONFIG.learning_rate}lr-{CONFIG.batch_size}batch-{date_str}"
    log_path = f"/tmp/tinker-cookbook/document_type/{run_name}"

    cli_utils.check_log_dir(log_path, behavior_if_exists="ask")

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=CONFIG.student_model,
        renderer_name=CONFIG.renderer_name,
        max_length=CONFIG.max_length,
        batch_size=CONFIG.batch_size,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )

    dataset = FromConversationFileBuilder(common_config=common_config, file_path=DATA_PATH)

    config = train.Config(
        log_path=log_path,
        model_name=CONFIG.student_model,
        dataset_builder=dataset,
        learning_rate=CONFIG.learning_rate,
        lr_schedule="cosine",
        num_epochs=CONFIG.num_epochs,
        lora_rank=CONFIG.lora_rank,
        save_every=20,
        eval_every=20,
    )
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
