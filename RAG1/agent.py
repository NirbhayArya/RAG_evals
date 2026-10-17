import RAG1.tools as tl
from .prompts import system_agent_prompt
from langchain_groq import ChatGroq
from .tools import collecting_info
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from .objects import Answer
from langchain.agents.structured_output import ToolStrategy

llm = ChatGroq(model="openai/gpt-oss-20b",temperature=0.0)

agent = create_agent(
    model=llm,
    tools=[collecting_info],
    system_prompt=system_agent_prompt,
    response_format= ToolStrategy(Answer)
)

def ask(question) -> dict:
    """ Answer the user's question about the pdf"""

    tl.last_retrieved_context = []

    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]}
    )

    structured_response = result["structured_response"]

    return {
        "answer" : structured_response.answer,
        "context": structured_response.context
    } 

# print(ask("what is the diet of Penguins"))