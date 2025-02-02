from langchain_core.prompts import PromptTemplate

template = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant specializing in answering questions about personal financial transaction data.
Use three sentences maximum and keep the answer as concise as possible.

Context
{context}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

custom_rag_prompt = PromptTemplate.from_template(template)
