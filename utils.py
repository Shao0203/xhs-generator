import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from prompt_template import system_template_text, user_template_text


class Xhs(BaseModel):
    themes: List[str] = Field(
        description='five generated themes', min_items=5, max_items=5)
    content: str = Field(description='generated main content')


def gen_xhs(theme, api_key):
    prompt = ChatPromptTemplate([
        ('system', system_template_text),
        ('user', user_template_text)
    ])

    model = ChatOpenAI(
        model='kimi-k2-0711-preview',
        base_url='https://api.moonshot.cn/v1',
        api_key=api_key
    )

    parser = PydanticOutputParser(pydantic_object=Xhs)

    chain = prompt | model | parser
    result = chain.invoke({
        'parser_instructions': parser.get_format_instructions(),
        'theme': theme,
    })

    return result


# result = gen_xhs('Large Language Model', os.getenv('KIMI_API_KEY'))
# print(result)
