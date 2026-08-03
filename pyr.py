import os
from dotenv import load_dotenv
from typing import TypedDict, Literal
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
import arxiv
import streamlit as st
import getpass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
import asyncio
from openai import AsyncOpenAI
from agents import set_default_openai_client, set_trace_processors,function_tool
from langsmith.integrations.openai_agents_sdk import OpenAIAgentsTracingProcessor
import json
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from qdrant_client.http.exceptions import UnexpectedResponse


load_dotenv()

try:
    for key, value in st.secrets.items():
        os.environ[key] = str(value)
except Exception:
    pass

qdrant_key = st.secrets.get("QDRANT_API_KEY") or os.environ.get("QDRANT_API_KEY")
google_api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
deepseek_api_key = st.secrets.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", api_key=google_api_key)
deepseek_client = AsyncOpenAI(base_url="https://api.deepseek.com", api_key=deepseek_api_key)
set_default_openai_client(deepseek_client)


#os.environ["LANGSMITH_TRACING"] = "false"
#os.environ["LANGSMITH_ENDPOINT"] = "none"
#os.environ["LANGSMITH_API_KEY"] = "none"
#os.environ["LANGSMITH_PROJECT"] = "none"
os.environ["GOOGLE_API_KEY"] = "GOOGLE_API_KEY"
os.environ["GEMINI_API_KEY"] = "GOOGLE_API_KEY"
os.environ["DEEPSEEK_API_KEY"] = "DEEPSEEK_API_KEY"
os.environ["OPENAI_API_KEY"] = os.environ["DEEPSEEK_API_KEY"]

llm = ChatOpenAI(model="deepseek-reasoner", openai_api_base="https://api.deepseek.com", openai_api_key=deepseek_api_key, temperature=0.1, model_kwargs={"response_format": {"type": "text"}})
evaluator_llm = ChatOpenAI(model="deepseek-chat", openai_api_base="https://api.deepseek.com", openai_api_key=deepseek_api_key, temperature=0.0, model_kwargs={"response_format": {"type": "json_object"}})
collection_name = "arxiv_papers"

qdrant_key = None
try:
    qdrant_key = st.secrets.get("QDRANT_API_KEY")
except Exception:
    pass

if not qdrant_key:
    qdrant_key = os.environ.get("QDRANT_API_KEY")

client = QdrantClient(
    url="https://bb371d84-7df8-4c17-bd6d-53d45e8d997f.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key=qdrant_key,
    check_compatibility=False
)

def search_arxiv_papers(query: str) -> str:
    query_vector = embeddings.embed_query(query)
    
    search_results = client.query_points(
        collection_name="arxiv_papers",
        query=query_vector,
        limit=10
    )
    
    context_texts = [hit.payload.get("page_content", str(hit.payload)) for hit in search_results.points]
    return "\n\n---\n\n".join(context_texts) if context_texts else "No papers found."

try:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
    )
except Exception as e:
    pass

    
class State(TypedDict):
    answer: str
    question: str
    critique: str
    accurate: str


class critique(BaseModel):
    grade: Literal["accurate", "not accurate"] = Field(
        description="Decide if the answer is accurate enough or not",
    )
    critique: str = Field(
        description="If the answer is not enough accurate, provide feedback on how to improve it or what context is missing",
    )

evaluator = evaluator_llm.with_structured_output(critique)

def llm_call_generator(state: State):
    critique = state.get("critique")
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings,)
    relevant_docs = vector_store.similarity_search(state['question'], k=3)
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    formatting_instructions = ("Act as a leading domain researcher and principal investigator. Provide a rigorous, "
    "comprehensive, doctoral-level answer to the question strictly using only the provided "
    "context from the arXiv papers. Address all technical aspects, theoretical mechanisms, "
    "and trade-offs present in the text.\n\n"
    
    "Guidelines for your response:\n"
    "1. **Depth & Rigor:** Utilize formal definitions, mathematical models, or architectural "
    "mechanisms referenced in the context. Avoid surface-level generalizations.\n"
    "2. **Missing Information:** If the context lacks sufficient information, explicitly state "
    "that the answer is incomplete, indicate what is missing, and suggest specific research "
    "directions or additional arXiv queries to explore.\n"
    "3. **Structure:** Use clear academic headings, bullet points for key technical takeaways, "
    "and a short concluding summary.\n"
    "4. **Citations:** Explicitly include the title, publication date, and source/authors of "
    "the specific research papers referenced for each major claim.\n"
    "5. **Tone:** Maintain a professional, concise, and objective register focused purely on "
    "accurate technical synthesis without unnecessary padding.")
    
    if critique:
        prompt = (f"Based on this research:\n{context_text}\n\n"f"Write an answer about {state['question']} but take into account this feedback: {critique}"f"formatting_instructions: {formatting_instructions}")
        
    else:
        prompt = (f"Based on this research:\n{context_text}\n\n"f"Write an answer about {state['question']}"f"formatting_instructions: {formatting_instructions}")
    msg = llm.invoke(prompt)
    
    if hasattr(msg, "content"):
        content = msg.content
    else:
        content = msg

    if isinstance(content, list):
        text_pieces = [item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in content]
        response_content = "".join(text_pieces)
    elif isinstance(content, dict) and "text" in content:
        response_content = content["text"]
    else:
        response_content = str(content)   
    return {"answer": response_content}


def llm_call_evaluator(state: State):
    eval_prompt = (
        f"You are a strict technical evaluator. Grade the following answer for the given question.\n"
        f"Question: {state['question']}\n"
        f"Answer: {state['answer']}\n\n"
        f"You must output a valid JSON object with EXACTLY two keys:\n"
        f"1. \"grade\": either \"accurate\" or \"not accurate\"\n"
        f"2. \"critique\": feedback on how to improve it if it's not accurate (or empty string if accurate).\n")
    
    msg = evaluator_llm.invoke(eval_prompt)
    content = msg.content if hasattr(msg, "content") else str(msg)
    content_upper = content.upper()
    
    if "NOT ACCURATE" in content_upper:
        grade = "not accurate"
        critique_text = content.replace("NOT ACCURATE", "", 1).strip()
    else:
        grade = "accurate"
        critique_text = ""
        
    return {"accurate": grade, "critique": critique_text}


def route_answer(state: State):

    if state["accurate"] == "accurate":
        return "Accepted"
    elif state["accurate"] == "not accurate":
        return "Rejected + critique"


optimizer_builder = StateGraph(State)

optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",route_answer,{"Accepted": END,"Rejected + critique": "llm_call_generator",},)

optimizer_workflow = optimizer_builder.compile()

#streamlit

st.header("Agentic Research Assistant for Arxiv Papers")
st.write("scholarly articles in the fields of physics, mathematics, computer science, \n"
"quantitative biology, quantitative finance, statistics,\n"
"electrical engineering and systems science, and economics")


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Let's start researching! 👇"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


#@traceable(name="prompt")
if prompt := st.chat_input(""):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Running workflow...", expanded=True) as status:
            try:
                st.write("Searching papers...")
                key_prompt = f"identify and extract important core technical keywords from user's question. Output only those keywords \n:{prompt}"
                key_response = llm.invoke(key_prompt)
                if hasattr(key_response, "content"):
                    key_prompt = key_response.content
                elif isinstance(key_response, dict):
                    key_prompt = key_response.get("text", str(key_response))
                else:
                    key_prompt = str(key_response)
                search_client = arxiv.Client()
                search_client.query_url_format = "https://export.arxiv.org/api/query?{}"
                search_query = arxiv.Search(query=key_prompt,max_results=3)
                search_results = list(search_client.results(search_query))
                raw_docs = [Document(page_content=r.summary,metadata={"Title": r.title, "Published": str(r.published)}) for r in search_results]

                if raw_docs:
                    st.write("Splitting and indexing documents...")
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    docs = text_splitter.split_documents(raw_docs)
                    
                    vector_store = QdrantVectorStore(
                        client=client,
                        collection_name=collection_name,
                        embedding=embeddings,
                    )
                    vector_store.add_documents(docs)
                
                st.write("Running self-correcting generation loop...")
                final_state = optimizer_workflow.invoke({"question": prompt})

                raw_answer = final_state.get("answer", "No response generated.")
                if hasattr(raw_answer, "content"):
                    assistant_response = raw_answer.content
                elif isinstance(raw_answer, dict):
                    assistant_response = raw_answer.get("text", str(raw_answer))
                else:
                    assistant_response = str(raw_answer)
                status.update(label="Workflow completed successfully!", state="complete", expanded=False)
            except Exception as e:
                assistant_response = f"An error occurred: {str(e)}"
                status.update(label="Workflow failed.", state="error", expanded=True)

        st.markdown(assistant_response)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
