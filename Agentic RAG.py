from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.googlesearch import GoogleSearch
from crewai_tools import PDFSearchTool  # Use LangChain's PDFSearchTool
from phi.tools import Tool  # Import the Tool class from phi
from pydantic import Field  # Import Field from pydantic
import requests
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the Groq model
groq_model = Groq(id="mixtral-8x7b-32768")

# Download the PDF file
pdf_url = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf'
response = requests.get(pdf_url)

with open('attention_is_all_you_need.pdf', 'wb') as file:
    file.write(response.content)

# Wrap PDFSearchTool in a phi-compatible Tool
class PDFSearchWrapper(Tool):
    pdf_search_tool: PDFSearchTool = Field(default=None, exclude=True)  # Define pdf_search_tool as a field

    def __init__(self, pdf_path: str):
        super().__init__(name="PDFSearchTool", type="function")  # Add the required 'type' field
        self.pdf_search_tool = PDFSearchTool(pdf=pdf_path)  # Initialize the PDFSearchTool

    def run(self, query: str) -> str:
        """Run the PDF search tool."""
        return self.pdf_search_tool.run(query)

# Web Search Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=groq_model,
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

# PDF Search Agent
pdf_search_agent = Agent(
    name="PDF Search Agent",
    role="Search within PDF documents for information",
    model=groq_model,
    tools=[PDFSearchWrapper(pdf_path="attention_is_all_you_need.pdf")],  # Use the wrapped tool
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

# Router Agent
router_agent = Agent(
    name="Router Agent",
    role="Route user queries to the appropriate agent",
    model=groq_model,
    instructions=[
        "Analyze the user query and decide whether to use the Web Search Agent or PDF Search Agent.",
        "If the query is about general knowledge or requires web search, route it to the Web Search Agent.",
        "If the query is about specific topics covered in the PDF, route it to the PDF Search Agent.",
    ],
    show_tool_calls=True,
    markdown=True,
)

# Multi-Agent System
multi_ai_agent = Agent(
    team=[router_agent, web_search_agent, pdf_search_agent],
    instructions=[
        "Always include sources",
        "Use tables to display data when applicable",
    ],
    show_tool_calls=True,
    markdown=True,
)

# Example Query
query = "who is narendra modi?"

# Print the response
multi_ai_agent.print_response(query, stream=True)

# The PDFSearchTool internally uses Hugging Face's BAAI/bge-small-en-v1.5 as the default embedding model.

# You can explicitly configure the embedding model by passing the config parameter to the PDFSearchTool.

# The embeddings are stored in ChromaDB for efficient retrieval during querying.