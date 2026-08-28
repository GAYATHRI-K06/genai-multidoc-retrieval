## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex
## NAME :GAYATHRI K
## REG NO: 212223230061
### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:
Extracting specific, nuanced information from a collection of dense academic papers is a slow and inefficient manual process. Standard search tools rely on exact keywords and fail to understand the conceptual context of a user's question. This program aims to build an AI agent that can intelligently query multiple documents to synthesize precise answers to complex questions.

### DESIGN STEPS:

#### STEP 1: Load PDF documents and create specialized search and summary tools for each paper.

#### STEP 2: Initialize an AI agent with an OpenAI model, giving it access to all the created tools.

#### STEP 3:  Query the agent with a specific question about one paper to get a detailed answer from its content.

### PROGRAM:
```
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()

import nest_asyncio
nest_asyncio.apply()
```
## Setup an agent over 3 papers
```
urls = [
    "https://openreview.net/pdf?id=VtmBAGCN7o",
    "https://openreview.net/pdf?id=6PmJoRfdaK",
    "https://openreview.net/pdf?id=hSyW5go0v8",
]

papers = [
    "AIeffectiveness.pdf",
    "foundationofai.pdf",
    "learningwithai.pdf",
]

from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
len(initial_tools)
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)
response = agent.query("Give me a summary of AIeffectiveness document")
print(str(response))

response = agent.query("Give me a summary of foundationofai and learningwithai document")
print(str(response))

response = agent.query("Tell me what are the major areas focused on foundationofai document")
print(str(response))
```
### OUTPUT:
<img width="940" height="352" alt="image" src="https://github.com/user-attachments/assets/dd5d732e-dc78-4c3a-aa35-c599eb946608" />
<img width="668" height="625" alt="image" src="https://github.com/user-attachments/assets/6f192f2e-9de2-4bb1-b3a8-1c6c2f60840a" />
<img width="960" height="385" alt="image" src="https://github.com/user-attachments/assets/22992b53-e2a2-4e5f-9173-d1421076a74a" />


### RESULT:
The system successfully retrieves and synthesizes relevant information from multiple documents, providing concise and relevant answers to the user's query. Performance is evaluated based on the accuracy, relevance, and coherence of the responses.
