from dataclasses import dataclass
from typing import Callable, Any
from transformers import (
    GPT2LMHeadModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    AutoModelForSeq2SeqLM,
)
import torch, logging

logger = logging.getLogger(__name__)


@dataclass
class ModelBag:
    model_name: str
    model: Any
    tokenizer: Any
    preprocessor: Callable[[dict[str, list]], dict[str, list]]
    postprocessor: Callable[[list[str]], list[str]] | None
    data_collator: Any
    device: torch.device
    for_training: bool
    skip_special_tokens: bool

    def __init__(self, model_name: str, for_training=False, device=torch.device("cpu")):
        self.model_name = model_name
        self.for_training = for_training
        self.device = device
        self.postprocessor = None
        self.skip_special_tokens = False

        if "gpt2" in model_name:
            self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

            if for_training:
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                self.tokenizer.padding_side = "right"
                self._add_special_tokens()

                self.preprocessor = self._make_prompts_and_labels
                self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer)

            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                if model_name.startswith("gpt2"):
                    self._add_special_tokens()

                self.tokenizer.padding_side = "left"

                self.preprocessor = lambda x: self.tokenizer(
                    [f"{c}{self.tokenizer.bos_token}" for c in x["comment"]]
                )
                self.postprocessor = lambda x: [
                    s[s.index(self.tokenizer.bos_token) :]
                    .replace("~", "")
                    .replace(self.tokenizer.bos_token, "")
                    .replace(self.tokenizer.eos_token, "")
                    .replace(self.tokenizer.pad_token, "")
                    for s in x
                ]

                self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        elif "bart" in model_name:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            if for_training:
                self.preprocessor = self._make_inputs_and_labels
                self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer)
            else:
                self.tokenizer.padding_side = "right"
                self.preprocessor = lambda x: self.tokenizer(x["comment"])
                self.postprocessor = lambda x: [s.replace("~", "") for s in x]
                self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
                self.skip_special_tokens = True

        else:
            raise ValueError(f"Don't know how to load model: {model_name}")

    def _make_inputs_and_labels(
        self, item: dict[str, list[str]]
    ) -> dict[str, list[int]]:
        tokenized = self.tokenizer(item["comment"], padding=False)

        tokenized_labels = self.tokenizer(item["neutral_comment"], padding=False)

        tokenized["labels"] = tokenized_labels["input_ids"]
        return tokenized

    def _make_prompts_and_labels(
        self, item: dict[str, list[str]]
    ) -> dict[str, list[int]]:
        text = [
            f"{c}{self.tokenizer.bos_token}{n}{self.tokenizer.eos_token}"
            for c, n in zip(item["comment"], item["neutral_comment"])
        ]
        tokenized = self.tokenizer(text, padding=False)

        labels = [[*ids] for ids in tokenized["input_ids"]]
        for label in labels:
            bos_pos = label.index(self.tokenizer.bos_token_id)
            label[: bos_pos + 1] = [-100] * (bos_pos + 1)

        tokenized["labels"] = labels
        return tokenized

    def _add_special_tokens(self):
        logger.info(
            "adding special tokens to tokenizer and resizing model embeddings..."
        )
        self.tokenizer.add_special_tokens({"pad_token": "<|PAD|>"})
        self.tokenizer.add_special_tokens({"bos_token": "<|BOS|>"})
        self.model.resize_token_embeddings(len(self.tokenizer))
