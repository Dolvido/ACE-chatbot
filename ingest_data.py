from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
import pickle

class LogHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('conversation.log'):
            print("Log file modified, re-vectorizing...")
            vectorize()

def vectorize():
    print("Loading data...")
    loader = DirectoryLoader('./logs', glob="**/*.md")
    raw_documents = loader.load()

    print("Splitting text...")
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=600,
        chunk_overlap=100,
        length_function=len,
    )
    documents = text_splitter.split_documents(raw_documents)

    print("Creating vectorstore...")
    model_name = "BAAI/bge-small-en"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    vectorstore = FAISS.from_documents(documents, embeddings)
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

    print("Logs re-vectorized.")

if __name__ == "__main__":
    vectorize()  # Initial vectorization

    observer = Observer()
    handler = LogHandler()
    observer.schedule(handler, path='logs/message-history', recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
