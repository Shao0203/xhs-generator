import streamlit as st
from utils import gen_xhs

st.header('小红书AI创作助手')
with st.sidebar:
    api_key = st.text_input('输入OpenAI API密钥')
    st.markdown('[获取OpenAI密钥](https://platform.openai.com/account/api-keys)')

theme = st.text_input('主题')
submit = st.button('开始写作', type='primary')

if submit and not api_key:
    st.info('请输入密钥')
    st.stop()
if submit and not theme:
    st.info('请输入主题')
    st.stop()
if submit:
    with st.spinner('AI正在创作中...请稍等'):
        result = gen_xhs(theme, api_key)
    st.success('创作完成！')
    left_column, right_column = st.columns(2)
    with left_column:
        for key, title in enumerate(result.themes):
            st.markdown(f'##### 标题{key+1}')
            st.write(title)
    with right_column:
        st.markdown("##### 正文")
        st.write(result.content)
