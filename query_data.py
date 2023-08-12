from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chat_models import ChatOpenAI

from langchain.memory import ConversationBufferMemory
import pickle

_template = """
Given the following conversation chat history and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question will be about CPG brand advertising at grocery retailers.

Chat History:
{chat_history}

Follow Up Question: {question}

Standalone question:
"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

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


def get_basic_qa_chain():
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    retriever = load_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    model = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory
    )
    return model


def get_custom_prompt_qa_chain():
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    retriever = load_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # see: https://github.com/langchain-ai/langchain/issues/6635
    # see: https://github.com/langchain-ai/langchain/issues/1497
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    )
    return model


def get_condense_prompt_qa_chain():
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    retriever = load_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # see: https://github.com/langchain-ai/langchain/issues/5890
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    )
    return model


def get_qa_with_sources_chain():
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    retriever = load_retriever()
    history = []
    model = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, return_source_documents=True
    )

    def model_func(question):
        # bug: this doesn't work with the built-in memory
        # hacking around it for the tutorial
        # see: https://github.com/langchain-ai/langchain/issues/5630
        new_input = {"question": question["question"], "chat_history": history}
        result = model(new_input)
        history.append((question["question"], result["answer"]))
        return result

    return model_func


chain_options = {
    "basic": get_basic_qa_chain,
    "with_sources": get_qa_with_sources_chain,
    "custom_prompt": get_custom_prompt_qa_chain,
    "condense_prompt": get_condense_prompt_qa_chain,
}
