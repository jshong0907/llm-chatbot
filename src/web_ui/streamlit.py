import streamlit as st

from inference.processor import InferenceProcessor


class StreamlitInterface:
    def __init__(self, inference_processor: InferenceProcessor):
        self.inference_processor = inference_processor

    def run(self):
        st.title("Streamlit")
        user_input = st.text_input("Type your message here and press enter", key="input")

        if user_input:
            response = self.inference_processor.chat(user_input)

            col1, col2 = st.columns([1, 4])  # Adjust the ratio as needed
            with col1:
                st.write("You:")
            with col2:
                st.write(user_input)

            col1, col2 = st.columns([1, 4])  # Repeat the column layout for the response
            with col1:
                st.write("Bot:")
            with col2:
                st.write(response, unsafe_allow_html=True)  # Use unsafe_allow_html if needed for formatting