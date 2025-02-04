from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from inference.prompts import custom_rag_prompt
from rag.processor import RagProcessor


class InferenceProcessor:
    MODEL_NAME = "Bllossom/llama-3.2-Korean-Bllossom-3B"
    MAX_TOKEN = 200
    TEMPERATURE = 0.1
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
            "do_sample": self.DO_SAMPLE,
            "temperature": 0.1,  # 응답을 더 확정적으로 생성
            "repetition_penalty": 1.2,  # 반복을 억제
            "top_k": 50,  # 높은 확률의 토큰들만 선택
            "top_p": 0.9,  # 누적 확률 기반 토큰 선택 제어
        }

    def chat(self, message: str) -> str:
        if self.rag_processor is not None:
            return self._chat_with_rag(message)
        return self._chat(message)

    def _chat(self, message: str) -> str:
        tokenized_input = self.tokenizer.encode(message, return_tensors="pt").to(self.model.device)
        output = self.model.generate(tokenized_input, **self._generation_args)[0]
        result = self.tokenizer.decode(output, skip_special_tokens=True)
        answer_only = result.replace(message.strip(), "").strip()
        return answer_only

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
