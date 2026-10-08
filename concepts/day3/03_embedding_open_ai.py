from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=api_key)

print("FIRST EMBEDDING")

print(embeddings.embed_query("Hello world"))



print("SECOND EMBEDDING")


print(embeddings.embed_documents(["Hello world", "Bye bye"]))






