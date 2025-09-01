import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


class GeminiRAGPipeline:
    def __init__(self):
        # Embeddings with Gemini
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )

        # Qdrant client (local)
        self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

        # LLM with Gemini
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    def setup_collection(self):
        """Create Qdrant collection if it doesn't exist"""
        try:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                # vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
        except Exception as e:
            print(f"Collection might already exist: {e}")

    def ingest_documents(self, file_paths, collection_name="documents"):
        """Ingest new documents into Qdrant."""
        documents = []

        for file_path in file_paths:
            if file_path.endswith(".txt"):
                loader = TextLoader(file_path)
            elif file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                continue

            documents.extend(loader.load())

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # Create vector store
        self.vector_store = Qdrant.from_documents(
            texts,
            embedding=self.embeddings,
            url=QDRANT_URL,
            collection_name=collection_name,
            force_recreate=True,
            api_key=QDRANT_API_KEY
        )

        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff", retriever=self.vector_store.as_retriever()
        )

        print(f"✅ Ingested {len(texts)} chunks into collection '{collection_name}'")

    def build_qa_chain(self, collection_name="documents"):
        """Build retrieval-augmented QA chain from Qdrant + Gemini."""
        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )

        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm, retriever=retriever, return_source_documents=True
        )
        return qa_chain

    def query(self, question, collection_name="documents"):
        """Query using RAG pipeline."""
        qa_chain = self.build_qa_chain(collection_name)
        result = qa_chain.invoke({"query": question})
        response = result["result"]

        return response
