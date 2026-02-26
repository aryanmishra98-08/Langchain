# Summary Memory Pattern
from typing import List
from pydantic import BaseModel, Field
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
load_dotenv('keys/.env')

class SummaryChatMessageHistory(BaseChatMessageHistory, BaseModel):
    """Keeps a summary + recent messages."""
    
    messages: List[BaseMessage] = Field(default_factory=list)
    llm: AzureChatOpenAI = Field(default_factory=lambda: AzureChatOpenAI(model="gpt-3.5-turbo"))
    max_messages: int = Field(default=10)
    summary: str = Field(default="")
    
    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add messages and summarize if needed."""
        self.messages.extend(messages)
        
        # If too many messages, summarize older ones
        if len(self.messages) > self.max_messages:
            # Summarize first half of messages
            to_summarize = self.messages[:self.max_messages // 2]
            
            # Create summary prompt
            conversation_text = "\n".join([
                f"{msg.__class__.__name__}: {msg.content}"
                for msg in to_summarize
            ])
            
            summary_prompt = f"""Previous summary: {self.summary}
            New conversation:
            {conversation_text}
            Create a concise summary of the key information from this conversation."""
            
            # Generate summary
            new_summary = self.llm.invoke(summary_prompt).content
            self.summary = new_summary
            
            # Keep only recent messages
            self.messages = self.messages[self.max_messages // 2:]
            
            # Add summary as system message
            if self.summary:
                self.messages.insert(0, SystemMessage(content=f"Summary of earlier conversation: {self.summary}"))
    
    def clear(self) -> None:
        """Clear everything."""
        self.messages = []
        self.summary = ""

# Setup (similar to before)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | AzureChatOpenAI(model="gpt-5.1-chat") | StrOutputParser()

summary_store = {}

def get_summary_history(session_id: str):
    if session_id not in summary_store:
        summary_store[session_id] = SummaryChatMessageHistory(max_messages=6)
    return summary_store[session_id]

chain_with_summary = RunnableWithMessageHistory(
    chain,
    get_summary_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Test with many messages
config = {"configurable": {"session_id": "summary_test"}}

# Conversation that will trigger summarization
topics = [
    "I'm planning a trip to Japan next month",
    "I want to visit Tokyo, Kyoto, and Osaka",
    "My budget is $3000",
    "I'm interested in temples and traditional culture",
    "I also love trying local food",
    "What should I pack?",
    "Any restaurant recommendations?"
]

for topic in topics:
    response = chain_with_summary.invoke({"input": topic}, config=config)
    print(f"User: {topic}")
    print(f"AI: {response[:100]}...\n")

# Still remembers key info from summary
response = chain_with_summary.invoke(
    {"input": "What's my budget again?"},
    config=config
)
print(f"AI: {response}")  # Should remember $3000 from summary