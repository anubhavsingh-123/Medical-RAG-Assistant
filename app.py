from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)
load_dotenv()

# Load API KEYS
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone index
index_name = "medical-chatbot"
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)



# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{context}\n\nQuestion: {question}")
])


# ----------------------------
# FIXED RAG CHAIN (WORKING)
# ----------------------------
def rag_pipeline(question):
    # 1. Retrieve relevant documents
    docs = retriever.invoke(question)

    # 2. Combine docs into context text
    context = "\n\n".join([doc.page_content for doc in docs])

    # 3. Format prompt
    prompt_text = prompt.format(
        context=context,
        question=question
    )

    # 4. Generate final answer from Gemini
    response = llm.invoke(prompt_text)

    return response.content


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form["msg"]

    answer = rag_pipeline(user_input)

    return str(answer)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
