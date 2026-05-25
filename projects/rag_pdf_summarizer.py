from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import gc
import warnings


warnings.filterwarnings("ignore", category=UserWarning)


def run_rag() -> None:
    print("Loading PDFs....")
    loader = PyPDFDirectoryLoader("data/")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Loaded {len(splits)} chunks from the PDFs.")

    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    vector_store = FAISS.from_documents(splits, embeddings)

    del embeddings
    gc.collect()

    llm = ChatOllama(model="phi3:mini", temperature=0)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    template = """You are a Research Assistant. Use the following context to answer the question.
    Ignore technical metadata like 'pdfTeX' or 'Document IDs'. Focus on the actual research findings.

    Context:
    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template=template)
    rag_chain = ({"context": itemgetter("question") | retriever, "question": itemgetter("question")} | prompt | llm | StrOutputParser())
    query = "What is the main topic of the PDFs? Summarize the key points."
    print("Asking the expert...")

    for chunk in rag_chain.stream({"question": query}):
        print(chunk, end="", flush=True)
    print("\n" + "-`" * 50 + "\n")


def main() -> None:
    load_dotenv()
    run_rag()


if __name__ == "__main__":
    main()
