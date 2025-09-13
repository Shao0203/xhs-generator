import streamlit as st
from chat import get_chat_response


st.title('💬智能聊天助手')

if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'ai', 'content': '你好, 我是你的AI助手!'}]

for message in st.session_state.messages:
    st.chat_message(message['role']).write(message['content'])

prompt = st.chat_input()

if prompt:
    st.session_state.messages.append({'role': 'human', 'content': prompt})
    st.chat_message('human').write(prompt)

    with st.spinner('AI is thinking...'):
        response = get_chat_response(prompt)

    st.session_state.messages.append({'role': 'ai', 'content': response})
    st.chat_message('ai').write(response)

print('#####', st.session_state)
