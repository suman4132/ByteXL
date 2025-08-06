from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
llm = GoogleGenerativeAI(
    model="gemini-2.5-pro"
)

user_input = input("Ask a question: ")
res = llm.invoke(user_input)
print(res)