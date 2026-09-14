import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def send_message(user_input: str) -> str:
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    response = llm.invoke(user_input)
    content = str(response.content)
    if not content:
        raise RuntimeError("LLM returned an empty response")
    return content


def run_chat() -> None:
    while True:
        try:
            user_input = input("You: ")
        except EOFError:
            break
        if user_input.strip().lower() == "quit":
            break
        try:
            response = send_message(user_input)
        except Exception:
            print("Sorry, something went wrong processing that message. Please try again.")
            continue
        print(response)


if __name__ == "__main__":
    run_chat()
