from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class InferenceProcessor:
    MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
    MAX_TOKEN = 500
    TEMPERATURE = 0.0  # Less Temperature, More creative answer
    DO_SAMPLE = False

    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

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
