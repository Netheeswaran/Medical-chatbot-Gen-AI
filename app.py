from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableMap
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)

load_dotenv()

api_key = os.environ.get('PINECONE_API_KEY')


openai_api_key = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = api_key
os.environ["OPENAI_API_KEY"] = openai_api_key

# Load the PDF files, split the text, and download embeddings
embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"

#Embed each document and upload to the existing index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create a retriever from the Pinecone vector store
# This will allow us to retrieve relevant documents based on user queries
retriever = docsearch.as_retriever(search_type="similarity",search_kwargs={"k": 3})

# Create a question-answering chain using the OpenAI model and the retriever
llm = OpenAI(temperature=0.4, max_tokens=500)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
# Compose RAG chain manually using LCEL style
rag_chain = (
    RunnableMap({
        "context": lambda x: retriever.invoke(x["input"]),
        "input": lambda x: x["input"]
    })
    | question_answer_chain
)



@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response)
    return str(response)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
