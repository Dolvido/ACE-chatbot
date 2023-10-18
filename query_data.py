from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chat_models import ChatOpenAI

from langchain.memory import ConversationBufferMemory
import pickle


from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.callbacks import StreamingStdOutCallbackHandler

from langchain.memory import ConversationBufferMemory
import pickle

from langchain.llms import GPT4All


local_path = "D:/guanaco-7B.ggmlv3.q5_1.bin"  
callbacks = [StreamingStdOutCallbackHandler()]
#llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)

_template = """You have a conversation history and a follow-up question. Your task is to transform the follow-up question into a standalone one. 
Ensure the question is clear and can be understood without the need for additional context. 

Chat History:
{chat_history}


Follow Up Input: {question}

Rephrased Standalone Question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI assistant for answering questions about the internal state of an ACE (Autonomous Cognitive Entity).
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
=========
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=[
                           "question", "context"])


def load_retriever():
    with open("vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    retriever = VectorStoreRetriever(vectorstore=vectorstore)
    return retriever

def get_basic_qa_chain():
    # Replaced ChatOpenAI with GPT4All and removed model_name parameter
    llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True) 
    retriever = load_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory)
    return model

def get_custom_prompt_qa_chain():
    # Similar changes as above
    llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True) 
    retriever = load_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT})
    return model


def get_condense_prompt_qa_chain():
    # Similar changes as above
    llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True) 
    retriever = load_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT})
    return model


def get_qa_with_sources_chain():
    llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True) 
    retriever = load_retriever()
    history = []
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True)

    def model_func(question):
        # bug: this doesn't work with the built-in memory
        # hacking around it for the tutorial
        # see: https://github.com/langchain-ai/langchain/issues/5630
        new_input = {"question": question['question'], "chat_history": history}
        result = model(new_input)
        history.append((question['question'], result['answer']))
        return result

    return model_func

# No changes in chain_options
chain_options = {
    "basic": get_basic_qa_chain,
    "with_sources": get_qa_with_sources_chain,
    "custom_prompt": get_custom_prompt_qa_chain,
    "condense_prompt": get_condense_prompt_qa_chain
}