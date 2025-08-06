from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.5)

user_input = input("Ask a question: ")
res = model.invoke(user_input)
print(res)