from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index
from rag_helper import RAGBase
from openai import OpenAI
from dotenv import load_dotenv

from toyaikit.llm import OpenAIClient
from toyaikit.tools import Tools
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import OpenAIResponsesRunner, DisplayingRunnerCallback

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

files = reader.read()

documents = []

for file in files:
    doc = file.parse()
    documents.append(doc)

print(len(documents)) # 72

index = Index(
    text_fields=["content"],
    keyword_fields=["filename"]
)

index.fit(documents)

question = "How does the agentic loop keep calling the model until it stops?"

search_results = index.search(
    question,
    boost_dict={"content": 2.0},
    num_results=5
)

print([doc["filename"] for doc in search_results]) # '01-agentic-rag/lessons/14-agentic-loop.md'

# ['01-agentic-rag/lessons/14-agentic-loop.md',
# '01-agentic-rag/lessons/15-frameworks.md',
# '01-agentic-rag/lessons/13-function-calling.md',
# '01-agentic-rag/lessons/11-agents-intro.md',
# '01-agentic-rag/lessons/16-other-frameworks.md']

# Building RAG Agent
load_dotenv()
openai_client = OpenAI()
rag_agent = RAGBase(
    index=index,
    llm_client=openai_client,
)

answer, usage = rag_agent.rag(question)
print(answer)
# It keeps calling the model in a `while True` loop and checks whether the model returned any `function_call` items.

# - If there is a function call, the code runs the tool, appends the tool result to `messages`, and loops again.
# - If there are no function calls in that turn, it `break`s and stops.

# - So the stop condition is: **no function calls this turn**.

print(usage) # 7125

# Use gpt-5.4-mini. How many input (prompt) tokens did we send to the model for this request?
# 7125

chunks = chunk_documents(documents, size=2000, step=1000)
print(len(chunks)) # 295

# Q5
index_chunks = Index(
    text_fields=["content"],
    keyword_fields=["filename"]
)

index_chunks.fit(chunks)

question_chunks = "How does the agentic loop keep calling the model until it stops?"

openai_client = OpenAI()
rag_agent = RAGBase(
    index=index_chunks,
    llm_client=openai_client,
)

answer, usage = rag_agent.rag(question_chunks)
print(answer)
print(usage) # 2308 Almost 3x less than the original usage

# 06
def search(query: str) -> dict[str, str]:
    """
    Search the documents database for entries matching the given query.
    """
    return index_chunks.search(
        query,
        num_results=5,
        boost_dict={"content": 2.0}
    )

instructions = "You're a course teaching assistant. Answer the student's question using the search tool. Make multiple searches with different keywords before answering."
query = "How does the agentic loop work, and how is it different from plain RAG?"
agent_tools = Tools()
agent_tools.add_tool(search)

chat_interface = IPythonChatInterface()
callback = DisplayingRunnerCallback(chat_interface)

runner = OpenAIResponsesRunner(
    tools=agent_tools,
    developer_prompt=instructions,
    chat_interface=chat_interface,
    llm_client=OpenAIClient(model="gpt-5.4-mini")
)

result = runner.loop(
    prompt=query,
    callback=callback,
)

search_calls = [
    msg for msg in result.all_messages
    if getattr(msg, "type", None) == "function_call" and msg.name == "search"
]

print(len(search_calls)) # 4