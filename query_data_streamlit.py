import streamlit as st

from langchain.callbacks import StreamlitCallbackHandler
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

import pickle

from secret import openai_api_key

st.set_page_config(
    page_title="ChatLangChain",
    page_icon="🦜",
    layout="centered",
    initial_sidebar_state="collapsed",
)

"# Chat🦜🔗"

template = """
You are an AI assistant who is designed to answer questions about CPG brand advertising at grocery retailers.

You are given a question followed by extracted parts of a long document to use as context to answer the question. 
Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.

If the question is not about CPG brand advertising at grocery retailers, politely inform them that you are tuned to only answer questions about CPG brand advertising at grocery retailers.

Format your response in Markdown.

---
Question: {question}
---

---
Context: {context}
---

Answer in Markdown:
"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])


def load_retriever():
    with open("vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    retriever = VectorStoreRetriever(vectorstore=vectorstore)
    return retriever


def get_custom_prompt_qa_chain(memory):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=openai_api_key)
    retriever = load_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    )
    return model


starter_message = "Ask me anything about Kroger Advertising!"

if "msg" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    memory.chat_memory.add_message(msg)

if prompt := st.chat_input(placeholder=starter_message):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = get_custom_prompt_qa_chain(
            {"question": prompt}
            # "chat_history": st.session_state.messages},
            # callbacks=[st_callback],
            # include_run_info=True,
        )
        st.session_state.messages.append(AIMessage(content=response["output"]))
        st.write(response["output"])
        memory.save_context({"input": prompt}, response)
        st.session_state["messages"] = memory.buffer
        run_id = response["__run"].run_id
