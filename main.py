import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from Agent import app

st.set_page_config(
    page_title="LangGraph Python Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 LangGraph Python Assistant")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)


if prompt := st.chat_input("Ask me to generate or explain Python code..."):

    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("human"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        inputs = {"messages": st.session_state.messages}
        
        final_state = app.invoke(inputs)
        
        response = final_state['messages'][-1]

        st.session_state.messages.append(response)
        with st.chat_message("ai"):
            st.markdown(response.content)