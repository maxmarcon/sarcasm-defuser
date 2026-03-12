import pandas as pd
import os, argparse, logging
from datasets import Dataset
from huggingface_hub import create_repo, login, whoami
from transformers import (
    Trainer,
    TrainingArguments,
)
from modelbag import ModelBag
import transformers

logger = logging.getLogger(__name__)

INPUT_FILE = "train.csv"


def make_datasets(input_file) -> Dataset:
    logger.info(f"loading {input_file}")
    df = pd.read_csv(input_file, index_col=0, parse_dates=["date"])
    df = df[["comment", "neutral_comment"]].dropna()

    return Dataset.from_pandas(df)


def train_model(
    mb: ModelBag,
    train_ds: Dataset,
    output_dir: str,
    *,
    resume: bool,
    epochs: int,
    save_steps: int,
    batch_size: int,
    max_steps: int,
    hub_id: str,
):
    train_ds = train_ds.map(
        mb.preprocessor, batched=True, remove_columns=train_ds.column_names
    )

    logger.info(
        f"training {mb.model_name}{" from checkpoint " if resume else " "}for {epochs} epochs and saving state to {output_dir}"
    )

    training_args = TrainingArguments(
        hub_model_id=hub_id,
        hub_strategy="checkpoint",
        push_to_hub=hub_id is not None,
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="steps" if save_steps > 0 else "no",
        save_total_limit=1,
        logging_steps=10,
        save_steps=save_steps,
        load_best_model_at_end=False,
        remove_unused_columns=False,
        max_steps=max_steps,
    )

    trainer = Trainer(
        model=mb.model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=mb.tokenizer,
        data_collator=mb.data_collator,
    )

    trainer.train(resume_from_checkpoint=resume)

    trainer.save_model(output_dir)


def upload_model(hub_id, model, tokenizer):
    create_repo(hub_id, exist_ok=True)

    model.push_to_hub(hub_id)
    tokenizer.push_to_hub(hub_id)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(module)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    transformers.utils.logging.set_verbosity_info()

    login()

    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--input-file", "-i", default=INPUT_FILE)
    parser.add_argument("--output-dir", "-o", default="./models")
    parser.add_argument("--resume", "-r", action="store_true")
    parser.add_argument("--epochs", "-e", default=10, type=int)
    parser.add_argument("--save-steps", "-s", default=0, type=int)
    parser.add_argument("--batch_size", "-b", default=8, type=int)
    parser.add_argument("--max-steps", "-m", default=-1, type=int)
    parser.add_argument(
        "--push-during-training",
        "-pt",
        action="store_true",
        help="whether to push saved model and checkpoints to HF hub during training",
    )
    parser.add_argument(
        "--push-to-hub",
        "-p",
        action="store_true",
        help="whether to push the final model and tokenizer to HF hub",
    )

    args = parser.parse_args()

    train_ds = make_datasets(args.input_file)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info(f"training model {args.model} on {len(train_ds)} pairs...")

    model_name = f"{args.model}-sarcasm-defuser"
    model_name = model_name[model_name.rfind("/") + 1 :]
    output_dir = f"{args.output_dir}/{model_name}"
    username = whoami()["name"]
    hub_id = f"{username}/{model_name}"

    logger.info(f"saving model to {output_dir}")

    mb = ModelBag(args.model, for_training=True)

    train_model(
        mb,
        train_ds,
        output_dir,
        resume=args.resume,
        epochs=args.epochs,
        save_steps=args.save_steps,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        hub_id=hub_id if args.push_during_training else None,
    )

    if args.push_to_hub:
        logger.info("uploading model to HF")
        upload_model(hub_id, mb.model, mb.tokenizer)
