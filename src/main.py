from inference.processor import InferenceProcessor
from rag.processor import RagProcessor
from web_ui.streamlit import StreamlitInterface


if __name__ == "__main__":
    rag_processor = RagProcessor()
    inference_processor = InferenceProcessor(rag_processor)
    ui = StreamlitInterface(inference_processor)

    ui.run()
