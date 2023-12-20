import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.memory import (
    ConversationBufferMemory,
    FileChatMessageHistory,
)

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
IS_DEBUG_MODE = os.getenv("DEBUG") == "true"

chat_llm = ChatOpenAI(api_key=API_KEY)

memory = ConversationBufferMemory(
    memory_key="content_memory",
    return_messages=True,
    chat_memory=FileChatMessageHistory("history.json"),
)

chat_prompt = ChatPromptTemplate(
    messages=[
        MessagesPlaceholder(variable_name="content_memory"),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
    input_variables=["content", "content_memory"],
)

chain = LLMChain(
    llm=chat_llm,
    prompt=chat_prompt,
    output_key="chat_response",
    memory=memory,
    verbose=IS_DEBUG_MODE,
)

while True:
    response = input(">> ")
    result = chain(response)
    print(result["chat_response"])
