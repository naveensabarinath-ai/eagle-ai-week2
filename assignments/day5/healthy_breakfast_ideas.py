import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a practical nutrition-minded cooking assistant. "
            "Suggest balanced breakfasts with whole-food ingredients.",
        ),
        (
            "human",
            "Give me exactly 5 healthy breakfast ideas. Make them varied, "
            "quick to prepare (20 minutes or less), and suitable for a typical "
            "adult. For each idea, provide a short name, the main ingredients, "
            "and one sentence explaining what makes it a balanced choice. "
            "Number the ideas from 1 to 5 and do not add an introduction or "
            "conclusion.",
        ),
    ]
)

model = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

response = model.invoke(prompt.invoke({}))
print(response.content)
