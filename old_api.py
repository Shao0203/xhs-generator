import os
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


def get_chat_response(prompt, memory, api_key=os.getenv('KIMI_API_KEY')):
    llm = ChatOpenAI(
        model='kimi-k2-0711-preview',
        base_url='https://api.moonshot.cn/v1',
        api_key=api_key
    )
    chain = ConversationChain(llm=llm, memory=memory)

    response = chain.invoke({'input': prompt})
    return response['response']


memory = ConversationBufferMemory(return_messages=True)
print(get_chat_response('我是亮亮，你是什么模型?(回答20字以内)', memory))
print(get_chat_response('我上一个问题是什么？你记得我名字吗?', memory))
# 我是Kimi，由月之暗面训练的大语言模型
# 你上一个问题是“你是什么模型?”，我记得你叫亮亮。


# ---------- 以下是streamlit前端代码 ----------


# import streamlit as st
# from langchain.memory import ConversationBufferMemory
# from chat import get_chat_response


# st.title('💬智能聊天助手')

# if 'memory' not in st.session_state:
#     st.session_state.memory = ConversationBufferMemory(return_messages=True)
#     st.session_state.messages = [{'role': 'ai', 'content': '你好, 我是你的AI助手!'}]

# for message in st.session_state.messages:
#     st.chat_message(message['role']).write(message['content'])

# prompt = st.chat_input()

# if prompt:
#     st.session_state.messages.append({'role': 'human', 'content': prompt})
#     st.chat_message('human').write(prompt)

#     with st.spinner('AI is thinking...'):
#         response = get_chat_response(prompt, st.session_state.memory)

#     st.session_state.messages.append({'role': 'ai', 'content': response})
#     st.chat_message('ai').write(response)
