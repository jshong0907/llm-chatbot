from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.inference.prompts import custom_rag_prompt
from src.rag.processor import RagProcessor


class InferenceProcessor:
    MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
    MAX_TOKEN = 500
    TEMPERATURE = 0.0  # Less Temperature, More creative answer
    DO_SAMPLE = False

    def __init__(self, rag_processor: Optional[RagProcessor] = None):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.rag_processor = rag_processor

    @property
    def _generation_args(self) -> dict:
        return {
            "max_new_tokens": self.MAX_TOKEN,
            "max_length": self.MAX_TOKEN,
            "temperature": self.TEMPERATURE,
            "do_sample": self.DO_SAMPLE,
        }

    def chat(self, message: str) -> str:
        tokenized_input = self.tokenizer.encode(message, return_tensors="pt").to(self.model.device)
        output = self.model.generate(tokenized_input, **self._generation_args)[0]
        return self.tokenizer.decode(output, skip_special_tokens=True)

    def chat_with_rag(self, message: str) -> str:
        if self.rag_processor is None:
            raise NotImplementedError

        retrieved_docs = self.rag_processor.process(message)
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
        prompted_message = custom_rag_prompt.invoke(
            {
                "question": message,
                "context": docs_content,
            }
        )
        return self.chat(prompted_message)
