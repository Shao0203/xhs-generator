from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os


# 1) 简单的 in-memory session store（生产可换 DB/file）
_store = {}


def _get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """返回指定 session 的聊天历史，如果没有就新建"""
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]


def get_chat_response(prompt_text, session_id="user1"):
    """根据历史聊天记录做出回答"""
    # 初始化 LLM
    llm = ChatOpenAI(
        model='kimi-k2-0711-preview',
        base_url='https://api.moonshot.cn/v1',
        api_key=os.getenv('KIMI_API_KEY')
    )

    # PromptTemplate：把 history 插进 prompt（MessagesPlaceholder）
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="history"),   # 自动占位：历史消息会被插入到这里
        ("human", "{input}"),                           # 模板把用户输入放在这里
    ])

    # 把 prompt 和 llm 串起来
    chain = prompt | llm

    # 包装成支持多轮对话的 带历史的 runnable
    chat_chain = RunnableWithMessageHistory(
        chain,
        _get_session_history,
        input_messages_key="input",        # prompt 的 key
        history_messages_key="history",    # 历史的 key
    )

    # 使用：必须在 config 里提供 session_id
    response = chat_chain.invoke(
        {"input": prompt_text},
        config={"configurable": {"session_id": session_id}},
    )
    return response.content


# # 示例对话
# print('111:', get_chat_response('我是Tom。你是谁?(回答20字以内)'))
# print('222:', get_chat_response('上一个问题是什么？我的名字是什么？'))
# print('333:', get_chat_response('你能记住我们之前的对话吗？'))


# # ---------------------------------------------
# # 旧写法 ConversationChain + ConversationBufferMemory
# import os
# from langchain_openai import ChatOpenAI
# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory


# def get_chat_response(prompt, memory, api_key=os.getenv('KIMI_API_KEY')):
#     llm = ChatOpenAI(
#         model='kimi-k2-0711-preview',
#         base_url='https://api.moonshot.cn/v1',
#         api_key=api_key
#     )
#     chain = ConversationChain(llm=llm, memory=memory)

#     response = chain.invoke({'input': prompt})
#     return response['response']


# memory = ConversationBufferMemory(return_messages=True)
# print(get_chat_response('我是亮亮，你是什么模型?(回答20字以内)', memory))
# print(get_chat_response('我上一个问题是什么？你记得我名字吗?', memory))
# # 我是Kimi，由月之暗面训练的大语言模型
# # 你上一个问题是“你是什么模型?”，我记得你叫亮亮。
