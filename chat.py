from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os

# 用一个 dict 存储不同 session 的对话历史
store = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """返回指定 session 的聊天历史，如果没有就新建"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 初始化 LLM
llm = ChatOpenAI(
    model='kimi-k2-0711-preview',
    base_url='https://api.moonshot.cn/v1',
    api_key=os.getenv('KIMI_API_KEY')
)

# 定义 prompt（必须告诉 RunnableWithMessageHistory 怎么用 input 和 history）
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),   # 历史消息
    ("human", "{input}"),                           # 用户输入
])

# 把 prompt 和 llm 串起来
chain = prompt | llm

# 包装成支持多轮对话的 runnable
chat_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",        # prompt 的 key
    history_messages_key="history",    # 历史的 key
)


def get_chat_response(prompt, session_id="user1"):
    """获取一次对话的响应"""
    response = chat_chain.invoke(
        {"input": prompt},
        config={"configurable": {"session_id": session_id}},
    )
    return response.content


# 示例对话
print(get_chat_response("牛顿提出过哪些知名的定律？"))
print(get_chat_response("我上一个问题是什么？"))


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
# print(get_chat_response("牛顿提出过哪些知名的定律，请简答？", memory))
# print(get_chat_response("我上一个问题是什么？", memory))
