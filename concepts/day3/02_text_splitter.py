from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader("sample.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=15,
    chunk_overlap=0
)

chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk.page_content}")

