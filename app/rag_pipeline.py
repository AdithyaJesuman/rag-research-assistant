from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from app.config import EMBEDDING_MODEL, RERANKER_MODEL, VECTOR_DB_PATH, TOP_K_RETRIEVAL, TOP_K_RERANK

set_llm_cache(InMemoryCache())

def load_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    vectordb = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )
    
    base_retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K_RETRIEVAL})
    
    reranker_model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    compressor = CrossEncoderReranker(model=reranker_model, top_n=TOP_K_RERANK)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(compression_retriever, question_answer_chain)
    
    return rag_chain

qa_chain = load_pipeline()
