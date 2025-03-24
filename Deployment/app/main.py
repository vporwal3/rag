import os
import json
import openai
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

# You might pull this from environment variables in production
# openai.api_key = os.getenv("OPENAI_API_KEY", "replace_with_key_if_needed")
openai.api_key = "Key"

# Initialize at import-time so it can be reused across requests
embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=openai.api_key
)

# Load FAISS store
vectorstore = FAISS.load_local(
    folder_path=os.path.join(os.path.dirname(__file__), "faiss_store"),
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

llm = ChatOpenAI(
    model_name="gpt-4",  # or "gpt-3.5-turbo", etc.
    openai_api_key=openai.api_key,
    temperature=0.1
)

template_text = """
You are a helpful answer generation assistant who can take in a question and generate a comprehensive and detailed answer based on the question provided.

You will be given the question along with context that is extracted from the book "FINANCIAL MODELING" from Simon Benninga. These contexts 
are basically 3 sections from any chapter within the group that may help answering the question better.

Important thing to note: The context is just given for reference. You have to generate a comprehensive answer based on your own knowledge.

Below you can find the details:

Question - {question}

Context:
{context}

Important things to note:
The context is just given for reference. You have to generate a detailed answer based on your own knowledge. 
Where-ever applicable keep a good balance between theoretical (formulas, definitions) and practical examples. You can also give examples 
on how you can do it in Excel where-ever necessary.
"""

prompt = PromptTemplate(
    template=template_text,
    input_variables=["context", "question"]
)

# Create the RetrievalQA chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

def run_rag_query(user_query: str) -> str:
    """Runs the RAG pipeline on the provided user query."""
    response = rag_chain.run(user_query)
    return response

# def handle_request(query: str) -> dict:
#     """
#     Handle the request by running the user query through the RAG pipeline.
#     Return a dict that can be serialized as JSON.
#     """
#     if not query:
#         return {"error": "No query provided"}

#     # Run RAG
#     try:
#         result = run_rag_query(query)
#         return {"answer": result}
#     except Exception as e:
#         return {"error": str(e)}

def handle_request(query: str) -> dict:
    print("[DEBUG] handle_request called with query:", query)  # 1
    if not query:
        print("[DEBUG] No query provided")
        return {"error": "No query provided"}
    try:
        print("[DEBUG] about to run run_rag_query...")
        result = run_rag_query(query)
        print("[DEBUG] run_rag_query returned:", result)
        return {"answer": result}
    except Exception as e:
        print("[ERROR] Exception in handle_request:", e)
        return {"error": str(e)}

# def handle_request(query: str) -> dict:
#     try
#     import os
#     import json
#     import openai
#     from langchain.chains import RetrievalQA
#     from langchain.chat_models import ChatOpenAI
#     from langchain.vectorstores import FAISS
#     from langchain.embeddings import OpenAIEmbeddings
#     from langchain.prompts import PromptTemplate

#     print("[DEBUG] handle_request called with query:", query)

#     if not query:
#         return {"error": "No query provided"}

#     try:
#         # 1. Set your API key (ideally from env vars)
#         openai.api_key = "YOUR_ACTUAL_KEY"  # or os.environ.get("OPENAI_API_KEY")
        
#         # 2. Instantiate embeddings
#         embedding_model = OpenAIEmbeddings(
#             model="text-embedding-ada-002",
#             openai_api_key=openai.api_key
#         )

#         # 3. Load the local FAISS store
#         folder_path = os.path.join(os.path.dirname(__file__), "faiss_store")
#         print("[DEBUG] Attempting to load FAISS from:", folder_path)
#         vectorstore = FAISS.load_local(
#             folder_path=folder_path,
#             embeddings=embedding_model,
#             allow_dangerous_deserialization=True
#         )

#         retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

#         # 4. Build your chain objects
#         template_text = """
#         Your instructions...
#         """

#         prompt = PromptTemplate(
#             template=template_text,
#             input_variables=["context", "question"]
#         )

#         llm = ChatOpenAI(
#             model_name="gpt-4",
#             openai_api_key=openai.api_key,
#             temperature=0.1
#         )

#         rag_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=retriever,
#             chain_type_kwargs={"prompt": prompt}
#         )

#         # 5. Actually run the chain
#         print("[DEBUG] About to run the chain...")
#         result = rag_chain.run(query)

#         print("[DEBUG] chain returned:", result)
#         return {"answer": result}

#     except Exception as e:
#         print("[ERROR] Exception in handle_request:", e)
#         return {"error": str(e)}


