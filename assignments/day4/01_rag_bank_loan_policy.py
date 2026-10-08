import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = Path(__file__).resolve().parent / "Personal_Loan_Policy_RAG_Practice.pdf"
FAISS_PATH = "personal_loan_faiss"

if not api_key:
    raise ValueError("Please set OPENAI_API_KEY environment variable")


# ============================================================
# 1. LOAD BANKING LOAN PDF
# ============================================================

print("\n1. Loading Personal Loan Policy PDF...")

loader = PyPDFLoader(str(PDF_PATH))

documents = loader.load()

print(f"Pages loaded: {len(documents)}")


# ============================================================
# 2. TEXT SPLITTING
# ============================================================

print("\n2. Splitting PDF into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(documents)

print(f"Chunks created: {len(chunks)}")


# ============================================================
# 3. CREATE EMBEDDINGS
# ============================================================

print("\n3. Creating embeddings...")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# ============================================================
# 4. STORE CHUNKS IN FAISS
# ============================================================

print("\n4. Creating FAISS vector store...")

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

vectorstore.save_local(FAISS_PATH)

print(f"FAISS database saved at: {FAISS_PATH}")


# ============================================================
# 5. CREATE RETRIEVER
# ============================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)


# ============================================================
# 6. OPENAI LLM
# ============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# ============================================================
# 7. BANKING RAG PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template("""
You are a Banking Personal Loan Assistant.

You must answer the customer's question using ONLY
the information provided in the retrieved Personal
Loan Policy document.

Do not invent loan policies, rates, eligibility rules,
fees, or calculations.

If the information is not available in the retrieved
document, respond:

"I could not find this information in the Personal
Loan Policy document."

Retrieved Loan Policy Context:
--------------------------------
{context}
--------------------------------

Customer Question:
{question}

Provide a clear and concise answer.

If applicable, mention the relevant loan policy section
or PDF page number.
""")


# ============================================================
# 8. ASK QUESTION
# ============================================================

def ask_question(question):

    # Retrieve relevant banking policy chunks
    documents = retriever.invoke(question)

    # Build context
    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    # Create RAG prompt
    formatted_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    # Send context + question to OpenAI
    response = llm.invoke(formatted_prompt)

    return response.content, documents


# ============================================================
# 9. CHAT APPLICATION
# ============================================================

print("\n==========================================")
print("     PERSONAL LOAN RAG ASSISTANT")
print("==========================================")

print("\nAsk questions about the Personal Loan Policy.")
print("Type 'exit' to stop.\n")


while True:

    question = input("Customer Question: ")

    if question.lower() in ["exit", "quit"]:
        print("\nApplication stopped.")
        break

    if not question.strip():
        continue

    answer, retrieved_docs = ask_question(question)

    print("\n------------------------------------------")
    print("ANSWER")
    print("------------------------------------------")
    print(answer)

    print("\n------------------------------------------")
    print("RETRIEVED POLICY SECTIONS")
    print("------------------------------------------")

    for i, doc in enumerate(retrieved_docs, 1):

        page = doc.metadata.get("page", "Unknown")

        print(
            f"Chunk {i} -> PDF Page {page + 1}"
        )

    print()


    # 
    #What is the minimum salary required?

#What is the preferred credit score?

#Can someone with a CIBIL score of 690 get the loan?

#What is the maximum loan amount?

#How much is the foreclosure charge after 18 months?

#Can a self-employed person apply?

#What documents are required for a self-employed applicant?

#Can a person with ₹80,000 income and ₹12,000 existing EMI afford a new loan?

#What happens if the applicant changes jobs?

#Can I add a co-applicant?

#What is not mentioned in the policy?