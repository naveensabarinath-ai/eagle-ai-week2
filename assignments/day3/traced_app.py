import os
from dotenv import load_dotenv


load_dotenv()
# Turn Langsmith tracking ON
os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"]="eagle-ai-week2"


print("LangSmith Connected")