from langchain.llms import GooglePalm
from dotenv import load_dotenv
load_dotenv()
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


API_KEY=os.environ["GOOGLE_API_KEY"]
llm=GooglePalm(
    google_api_key=API_KEY,
    temperature=0.5
)


embeddings = HuggingFaceEmbeddings()
vectordb_file_path='FAISS_INDEX' #look to the function

def create_vector_db():
    loader = CSVLoader(file_path='faqs.csv', source_column='prompt')  # Assuming 'prompt' is the correct column name

    data = loader.load()

    vector_db=FAISS.from_documents(documents=data,embedding=embeddings )
    vector_db.save_local('FAISS_INDEX')
def get_qa_chain():
    vector_db=FAISS.load_local(vectordb_file_path,embeddings )
    retriever=vector_db.as_retriever(score_threshold=0.7)
    prompt_template=""" Given the following context and a question, generate an answer bases on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context.
    If the answer is not found in the context, kindly state, "I dont know." Dont try to make up an asnwer. 

    CONTEXT:{context}
    QUESTION:{question}"""

    PROMT=PromptTemplate(
    template=prompt_template, input_variables=['context','question']
    )
    chain=RetrievalQA.from_chain_type(llm=llm,
            chain_type='stuff',
            retriever=retriever,
            input_key='query',
            return_source_documents=False,
            chain_type_kwargs={'prompt':PROMT}
           )
    return chain



if __name__=='__main__':
    chain=get_qa_chain()
    print(chain('Do you provide internship'))