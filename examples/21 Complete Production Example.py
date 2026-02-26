# Complete Production ChatBot Example
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from datetime import datetime
from dotenv import load_dotenv
load_dotenv('keys/.env')

class ProductionChatbot:
    def __init__(self, db_path="chat_history.db"):
        # Setup prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant.
            Current time: {current_time}
            Be concise and helpful!"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Build chain
        self.model = AzureChatOpenAI(model="gpt-5.1-chat")
        self.chain = self.prompt | self.model | StrOutputParser()
        
        # Database path
        self.db_path = db_path
        
        # Wrap with history
        self.conversational_chain = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )
    
    def get_session_history(self, session_id: str):
        """Get persistent SQL history for a session."""
        return SQLChatMessageHistory(
            session_id=session_id,
            connection_string=f"sqlite:///{self.db_path}"
        )
    
    def chat(self, user_input: str, session_id: str):
        """Send a message and get response."""
        config = {"configurable": {"session_id": session_id}}
        
        try:
            response = self.conversational_chain.invoke(
                {
                    "input": user_input,
                    "current_time": datetime.now().strftime("%Y-%m-%d %H:%M")
                },
                config=config
            )
            return response
        except Exception as e:
            print(f"Error: {e}")
            return "I encountered an error. Please try again."
    
    def clear_history(self, session_id: str):
        """Clear conversation history for a session."""
        history = self.get_session_history(session_id)
        history.clear()

# Usage
if __name__ == "__main__":
    bot = ProductionChatbot()
    session = "user_12345"
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ["quit", "exit"]:
            break
        
        if user_input.lower() == "clear":
            bot.clear_history(session)
            print("History cleared!")
            continue
        
        response = bot.chat(user_input, session)
        print(f"\nAI: {response}")
