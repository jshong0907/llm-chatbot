from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RagProcessor:
    EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v3"
    CHUNK_SIZE = 100
    CHUNK_OVERLAP = 20

    def __init__(self, file: str):
        self._install_nltk_dependencies()
        self.embeddings = HuggingFaceEmbeddings(model_name=self.EMBEDDING_MODEL_NAME, model_kwargs={"trust_remote_code": True})
        self.loader = UnstructuredExcelLoader(file)
        self._docs = self.loader.load()
        self.vector_store = InMemoryVectorStore(self.embeddings)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,  # chunk size (characters)
            chunk_overlap=self.CHUNK_OVERLAP,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
        self.docs = self.text_splitter.split_documents(self._docs)
        self.vector_store.add_documents(self.docs)
        self.retriever: VectorStoreRetriever = self.vector_store.as_retriever()

    def _install_nltk_dependencies(self, quiet=True):
        import nltk
        nltk.download('punkt', quiet=quiet)
        nltk.download('punkt_tab', quiet=quiet)
        nltk.download('averaged_perceptron_tagger_eng', quiet=quiet)
        nltk.download('averaged_perceptron_tagger_eng', quiet=quiet)

    def process(self, keyword: str) -> list[Document]:
        return self.retriever.invoke(keyword)
