# Chain --> Sequence of steps.. 
# Prompt --> Model ---> Output

# Chain = promt | Model | Output

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()



prompt = ChatPromptTemplate.from_template("Explan {topic} in simple sentence")

model = ChatOpenAI(model="gpt-4o-mini",api_key=os.getenv("OPENAI_API_KEY"))

outParser = StrOutputParser()

print("All ready")

chain = prompt | model | outParser

# invoke chain to explain the topic on Pyramid
res = chain.invoke({"topic":"Pyramid"})

print(res)

## Slightly long chain
suggestionPrompt = ChatPromptTemplate.from_template( 
    "Give one short, friendly suggestion on which product is best {skill}.")


suggestionChain = suggestionPrompt | model | outParser

ans = suggestionChain.invoke({"skill":"Perfume"})

print(ans)

