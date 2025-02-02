from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from inference.prompts import custom_rag_prompt
from rag.processor import RagProcessor


class InferenceProcessor:
    MODEL_NAME = "Bllossom/llama-3.2-Korean-Bllossom-3B"
    MAX_TOKEN = 200
    TEMPERATURE = 0.1
    DO_SAMPLE = True

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
            "do_sample": self.DO_SAMPLE,
        }

    def chat(self, message: str) -> str:
        if self.rag_processor is not None:
            return self._chat_with_rag(message)
        return self._chat(message)

    def _chat(self, message: str) -> str:
        tokenized_input = self.tokenizer.encode(message, return_tensors="pt").to(self.model.device)
        output = self.model.generate(tokenized_input, **self._generation_args)[0]
        return self.tokenizer.decode(output, skip_special_tokens=True)

    def _chat_with_rag(self, message: str) -> str:
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
        return self._chat(prompted_message.to_string())
