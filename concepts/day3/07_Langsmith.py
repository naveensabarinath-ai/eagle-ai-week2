import os
from dotenv import load_dotenv


load_dotenv()
# Turn Langsmith tracking ON
os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_API_KEY"]="test"
os.environ["LANGSMITH_PROJECT"]="monitoring"


print("LangSmith Connected")