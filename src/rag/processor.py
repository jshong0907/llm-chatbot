import os

from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RagProcessor:
    EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v3"
    CHUNK_SIZE = 100
    CHUNK_OVERLAP = 20
    DATASET_DIR = "/datasets"

    def __init__(self):
        self._install_nltk_dependencies()
        self.embeddings = HuggingFaceEmbeddings(model_name=self.EMBEDDING_MODEL_NAME, model_kwargs={"trust_remote_code": True})
        self.vector_store = InMemoryVectorStore(self.embeddings)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,  # chunk size (characters)
            chunk_overlap=self.CHUNK_OVERLAP,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
        self._add_documents()
        self.retriever: VectorStoreRetriever = self.vector_store.as_retriever()

    def _install_nltk_dependencies(self, quiet=True):
        import nltk
        nltk.download('punkt', quiet=quiet)
        nltk.download('punkt_tab', quiet=quiet)
        nltk.download('averaged_perceptron_tagger_eng', quiet=quiet)
        nltk.download('averaged_perceptron_tagger_eng', quiet=quiet)

    def _add_documents(self):
        for filename in os.listdir(self.DATASET_DIR):
            file_path = os.path.join(self.DATASET_DIR, filename)
            if os.path.isfile(file_path):
                self._add_file_to_documents(file_path)

    def _add_file_to_documents(self, file):
        loader = UnstructuredExcelLoader(file)
        _docs = loader.load()

        docs = self.text_splitter.split_documents(_docs)
        self.vector_store.add_documents(docs)

    def process(self, keyword: str) -> list[Document]:
        return self.retriever.invoke(keyword)
