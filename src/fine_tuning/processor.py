import torch
from datasets import load_dataset
from datasets.formatting.formatting import LazyRow
from peft import LoraConfig
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig


class FineTuningProcessor:
    MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
    DATASET_NAME = "BCCard/BCCard-Finance-Kor-QnA"
    OUTPUT_DIR = "/checkpoints"
    TRAIN_EPOCH = 1

    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.MODEL_NAME, **self.model_config)
        self.tokenizer = self._load_tokenizer()
        self.dataset = load_dataset(self.DATASET_NAME, split="train").train_test_split(test_size=0.2)
        self.train_dataset, self.test_dataset = self.dataset["train"], self.dataset["test"]
        column_names = list(self.train_dataset.features)
        self.processed_train_dataset = self.train_dataset.map(
            self.apply_chat_template,
            fn_kwargs={"tokenizer": self.tokenizer},
            num_proc=10,
            remove_columns=column_names,
        )
        self.processed_test_dataset = self.test_dataset.map(
            self.apply_chat_template,
            fn_kwargs={"tokenizer": self.tokenizer},
            num_proc=10,
            remove_columns=column_names,
        )
        collator = DataCollatorForCompletionOnlyLM("<|assistant|>", tokenizer=self.tokenizer)
        self.trainer = SFTTrainer(
            model=self.model,
            args=self.training_config,
            peft_config=self.peft_config,
            train_dataset=self.processed_train_dataset,
            eval_dataset=self.processed_test_dataset,
            data_collator=collator,
        )

    def train(self):
        self._train()
        self._evaluate()
        self.trainer.save_model(self.training_config.output_dir)

    def _train(self):
        train_result = self.trainer.train()
        metrics = train_result.metrics
        self.trainer.log_metrics(self.trainer, "train", metrics)
        self.trainer.save_metrics(self.trainer, "train", metrics)
        self.trainer.save_state(self.trainer)

    def _evaluate(self):
        self.tokenizer.padding_side = 'left'
        metrics = self.trainer.evaluate()
        metrics["eval_samples"] = len(self.processed_test_dataset)
        self.trainer.log_metrics(self.trainer, "eval", metrics)
        self.trainer.save_metrics(self.trainer, "eval", metrics)

    @property
    def _peft_config(self):
        return {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": "all-linear",
            "modules_to_save": None,
        }

    @property
    def training_config(self):
        return SFTConfig(
            bf16=True,
            do_eval=False,
            learning_rate=5.0e-06,
            log_level="info",
            logging_strategy="steps",
            lr_scheduler_type="cosine",
            num_train_epochs=self.TRAIN_EPOCH,
            output_dir=self.OUTPUT_DIR,
            overwrite_output_dir=True,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            remove_unused_columns=True,
            save_steps=100,
            save_total_limit=1,
            seed=0,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            gradient_accumulation_steps=1,
            warmup_ratio=0.2,
            max_seq_length=2048,
        )

    @property
    def peft_config(self):
        return LoraConfig(**self._peft_config)

    @property
    def model_config(self):
        return dict(
            use_cache=False,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
            torch_dtype=torch.bfloat16,
            device_map=None
        )

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        tokenizer.model_max_length = 2048
        tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        tokenizer.padding_side = 'right'
        return tokenizer

    @staticmethod
    def apply_chat_template(data: LazyRow, tokenizer):
        messages = [
            {"role": "system", "content": data['instruction']},
            {"role": "assistant", "content": data['output']}
        ]

        data["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        return data
