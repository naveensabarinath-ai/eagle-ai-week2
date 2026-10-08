from langchain_community.document_loaders import TextLoader

loader = TextLoader("sample.txt")
documents = loader.load()

print(f"Loaded {len(documents)} documents.")
print(f"First document content: {documents[0].page_content}")


# pyPDFLoader   ( Needs pip install pyPdf)



# CSVLoader   

# WebBaseLoader


