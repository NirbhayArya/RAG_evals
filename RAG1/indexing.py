from dotenv import load_dotenv
from .file_reader import extract_paragraphs
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

embedding = MistralAIEmbeddings(model="mistral-embed")
vectore_store = Chroma(persist_directory="db", embedding_function=embedding)

chunks = extract_paragraphs("pdfs/Penguins_ACL.pdf")
vectore_store.add_texts(texts=chunks)
# print(f"indexed {len(chunks)} chunks")