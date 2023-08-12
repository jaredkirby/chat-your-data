import streamlit as st

from langchain.callbacks import StreamlitCallbackHandler
from langchain.schema import SystemMessage, AIMessage, HumanMessage

from query_data_test import chain_options

st.set_page_config(
    page_title="ChatLangChain",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

"# Chat🦜🔗"

if __name__ == "__main__":
    for msg in st.session_state.messages:
        if isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        memory.chat_memory.add_message(msg)

    chain = chain_options["basic"]()
    starter_message = (
        "What is the minimum budget requirements to run a Pinterest ad with Kroger?"
    )
    if prompt := st.chat_input(placeholder=starter_message):
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            response = chain(
                {"question": prompt, "chat_history": st.session_state.messages},
                callbacks=[st_callback],
                include_run_info=True,
            )
            st.session_state.messages.append(AIMessage(content=response["output"]))
            st.write(response["output"])
            memory.save_context({"input": prompt}, response)
            st.session_state["messages"] = memory.buffer
            run_id = response["__run"].run_id

            col_blank, col_text, col1, col2 = st.columns([10, 2, 1, 1])
            with col_text:
                st.text("Feedback:")

            with col1:
                st.button("👍", on_click=send_feedback, args=(run_id, 1))

            with col2:
                st.button("👎", on_click=send_feedback, args=(run_id, 0))
