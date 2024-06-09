import streamlit as st
from langchain_openai import ChatOpenAI
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain.vectorstores.chroma import Chroma

def generate_response(input_text,openai_api_key):
    """
    用户密钥对OpenAI API进行身份验证，发送提示并获取AI生成的响应
    该函数接收用户的提示作为参数，并使用st.info来在蓝色框中显示AI生成的响应
    """
    llm = ChatOpenAI(temperature=0.8, openai_api_key =openai_api_key, base_url="https://api.chatanywhere.tech/v1")
    output = llm.invoke(input_text)
    st.info(output.content)

# 添加检索问答
def get_vectordb():
    # 定义embeddings
    embedding = ZhipuAIEmbeddings()
    # vdb持久化路径
    persist_directory = './data_base/vector_db/chroma'
    # load vdb
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    return vectordb

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
## 带有历史记录的问答链
def get_chat_qa_chain(question:str, openai_api_key: str):
    vectordb = get_vectordb()
    llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.8,openai_api_key = openai_api_key,base_url="https://api.chatanywhere.tech/v1")
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )
    retriever = vectordb.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever = retriever,
        memory = memory
    )
    result = qa({"question":question})
    return result['answer']

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
# 不带历史记录的问答链
def get_qa_chain(question:str,openai_api_key:str):
    vectordb = get_vectordb()
    llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0.3,openai_api_key = openai_api_key,base_url="https://api.chatanywhere.tech/v1")
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
        案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
        {context}
        问题: {question}
        """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    result = qa_chain({"query": question})
    return result["result"]


# Streamlit 应用程序界面
def main():
    st.title('🦜🔗 GPT-5o')
    openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

    # 添加一个选择按钮来选择不同的模型
    selected_method = st.radio(
        "你想选择哪种模式进行对话?",
        ["None","qa_chain","chat_qa_chain"],
        captions= ["不使用检索问答的普通模式","不带历史记录的检索问答模式","带历史记录的检索问答模式"]
    )

    # 用于跟踪对话历史
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    messages = st.container(height=300)
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append({"role": "user", "text": prompt})

        if selected_method == "None":
            # 调用 respond 函数获取回答
            answer = generate_response(prompt, openai_api_key)
        elif selected_method == "qa_chain":
            answer = get_qa_chain(prompt, openai_api_key)
        elif selected_method == "chat_qa_chain":
            answer = get_chat_qa_chain(prompt,openai_api_key)
        
        # 检查回答是否为 None
        if answer is not None:
            # 将LLM的回答添加到对话历史中
            st.session_state.messages.append({"role": "assistant", "text": answer})

        # 显示整个对话历史
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])   

if __name__ == "__main__":
    main()