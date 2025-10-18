import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv(dotenv_path="openrouterapi.env")

openai_api_key = os.getenv("OPENAI_API_KEY")
print("Api has been read!")
print("Done. ")

llm = ChatOpenAI(
    model="z-ai/glm-4.5-air:free", 
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=openai_api_key,
    temperature=0.5
)

persist_directory = 'chroma_db_persistent'

code_samples = """
def add(a, b):
    return a + b
for i in range(5):
    print(f"Loop iteration {i}")
squares = [x**2 for x in range(10)]
print(squares)
"""
with open("code_examples.txt", "w") as f:
    f.write(code_samples)

loader = TextLoader("code_examples.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(
    documents=docs, 
    embedding=embedding_model,
    persist_directory=persist_directory
)
retriever = vectordb.as_retriever()
print("RAG retriever is now ready.")

class AgentState(TypedDict):
    messages: List[BaseMessage]
    intent: str
    retrevContent: List[str]

def intenClassifier(state: AgentState):
    latestMessage = state['messages'][-1].content.lower()
    if "generate" in latestMessage or "write" in latestMessage or "create" in latestMessage:
        intent = "generate"
    elif "explain" in latestMessage or "what is" in latestMessage or "describe" in latestMessage:
        intent = "explain"
    else:
        intent = "clarify"
    print(f"The intent of the user has been classified as: {intent}. ")
    return {"intent": intent}

def retrieveExamples(state:AgentState):
    lastMessage = state["messages"][-1].content
    retrievedDocs = retriever.invoke(lastMessage)
    contextSnipp = [doc.page_content for doc in retrievedDocs]
    print(f"Retrieved {len(contextSnipp)} context")
    return {"retrieval_context": contextSnipp}

def LLM(state:AgentState):
    intent = state["intent"]
    context = state.get("retrevContent", [])
    user_input = state['messages']
    if intent == "generate":
        template = """You are a Python code assistant. Based on the following examples, generate the Python code requested by the user.

        Examples:
        {context}

        User Request:
        {input}"""
        prompt = PromptTemplate.from_template(template).format(context="\\n---\\n".join(context), input=user_input)
    elif intent == "explain":
        template = "Explain the following Python concept like I am 5: {input}"
        prompt = PromptTemplate.from_template(template).format(input=user_input)
    else:
        prompt = "I'm not sure how to help with that. Please specify if you want me to 'generate' or 'explain' Python code."  
    response = llm.invoke(prompt)
    return {'messages': state["messages"] + [AIMessage(content=response.content)]}

def router(state:AgentState):
    if state['intent'] == "generate":
        return "retrieve"
    else:
        return "llm_call";

graph = StateGraph(AgentState)
graph.add_node("classifier", intenClassifier)
graph.add_node("retrieve", retrieveExamples)
graph.add_node("llm_call", LLM)
graph.set_entry_point("classifier")
graph.add_conditional_edges(
    "classifier",
    router,
    {
        "retrieve": "retrieve", 
        "llm_call": "llm_call"
    }
)
graph.add_edge("retrieve", "llm_call")
graph.add_edge("llm_call", END)

app = graph.compile()
print("Done. ")

print("The LangGraph Python Assistant is running. Type 'quit' to exit.")

if __name__ == "__main__":
    print("--- Running Agent.py in Direct Test Mode ---")
    print("The LangGraph Python Assistant is running. Type 'quit' to exit.")
    while True:
        user_input = input("User: ")
        if user_input.lower() == "quit":
            print("The Assistant has been terminated.")
            break
        inputs = {"messages": [HumanMessage(content=user_input)]}
        finalState = app.invoke(inputs)
        print("AI:", finalState['messages'][-1].content)