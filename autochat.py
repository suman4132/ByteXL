from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
loader = PyPDFLoader("Cloud Computing.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=0,
    )
chunks = splitter.split_documents(docs)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriver = vectorstore.as_retriever(search_type="similarity",
 search_kwargs={"k": 3})
chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.1
)
prompt = PromptTemplate(
    template="You are a helpful assistant. Use the following pieces of context to answer the question at the end.\n\n{context}\n\nQuestion: {question}\nAnswer:",
    input_variables=["context", "question"]
)
question = 'What is cloud computing?'
retrive_docs = retriver.invoke(question)
context = "\n\n".join([doc.page_content for doc in retrive_docs])
formatted_prompt = prompt.invoke({
    "context":context,
    "question":question
})
parser = StrOutputParser()
response = chat_model.invoke(formatted_prompt)
print(parser.invoke(response.content))

