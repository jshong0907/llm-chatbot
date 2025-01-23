from src.inference.processor import InferenceProcessor
from src.web_ui.streamlit import StreamlitInterface


if __name__ == "__main__":
    processor = InferenceProcessor()
    ui = StreamlitInterface(processor)

    ui.run()
