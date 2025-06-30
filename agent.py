# agent.py

from llama_index.llms.nvidia import NVIDIA
from llama_index.core import Settings

from llama_parse import LlamaParse

import os
import nest_asyncio
nest_asyncio.apply()

from llama_index.core.query_engine import JSONalyzeQueryEngine

from llama_index.core import SimpleDirectoryReader


doc_1 = SimpleDirectoryReader(input_files=["docs/company_it_policies.pdf"]).load_data()
doc_2 = SimpleDirectoryReader(input_files=["docs/installation_guides.pdf"]).load_data()
doc_3 = SimpleDirectoryReader(input_files=["docs/it_support_categories.pdf"]).load_data()
doc_4 = SimpleDirectoryReader(input_files=["docs/knowledge_base.pdf"]).load_data()
doc_5 = SimpleDirectoryReader(input_files=["docs/sample_conversations.pdf"]).load_data()
doc_6 = SimpleDirectoryReader(input_files=["docs/test_requests.pdf"]).load_data()
doc_7 = SimpleDirectoryReader(input_files=["docs/troubleshooting_database.pdf"]).load_data()

from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=1024)
nod_1 = splitter.get_nodes_from_documents(doc_1)
nod_2 = splitter.get_nodes_from_documents(doc_2)
nod_3 = splitter.get_nodes_from_documents(doc_3)
nod_4 = splitter.get_nodes_from_documents(doc_4)
nod_5 = splitter.get_nodes_from_documents(doc_5)
nod_6 = splitter.get_nodes_from_documents(doc_6)
nod_7 = splitter.get_nodes_from_documents(doc_7)

from llama_index.llms.nvidia import NVIDIA
from llama_index.embeddings.nvidia import NVIDIAEmbedding

from dotenv import load_dotenv
import os

load_dotenv()

NVIDIA_EMB = os.getenv("NVIDIA_EMB")
NVIDIA_MOD = os.getenv("NVIDIA_MOD")

Settings.embed_model = NVIDIAEmbedding(
    model="nvidia/nv-embedqa-e5-v5",
    api_key=NVIDIA_EMB,
    truncate="END"
)

Settings.llm = NVIDIA(
    model="meta/llama-3.1-8b-instruct",
    api_key=NVIDIA_MOD
)

from llama_index.core import SummaryIndex, VectorStoreIndex

sum_1 = SummaryIndex(nod_1)
sum_2 = SummaryIndex(nod_2)
sum_3 = SummaryIndex(nod_3)
sum_4 = SummaryIndex(nod_4)
sum_5 = SummaryIndex(nod_5)
sum_6 = SummaryIndex(nod_6)
sum_7 = SummaryIndex(nod_7)

vec_1 = VectorStoreIndex(nod_1)
vec_2 = VectorStoreIndex(nod_2)
vec_3 = VectorStoreIndex(nod_3)
vec_4 = VectorStoreIndex(nod_4)
vec_5 = VectorStoreIndex(nod_5)
vec_6 = VectorStoreIndex(nod_6)
vec_7 = VectorStoreIndex(nod_7)

comp_pol_sum_eng    = sum_1.as_query_engine(response_mode="tree_summarize", use_async=True)
inst_guid_sum_eng   = sum_2.as_query_engine(response_mode="tree_summarize", use_async=True)
it_sup_cat_sum_eng  = sum_3.as_query_engine(response_mode="tree_summarize", use_async=True)
know_base_sum_eng   = sum_4.as_query_engine(response_mode="tree_summarize", use_async=True)
sample_conv_sum_eng = sum_5.as_query_engine(response_mode="tree_summarize", use_async=True)
test_req_sum_eng    = sum_6.as_query_engine(response_mode="tree_summarize", use_async=True)
troub_db_sum_eng    = sum_7.as_query_engine(response_mode="tree_summarize", use_async=True)

comp_pol_vec_eng    = vec_1.as_query_engine()
inst_guid_vec_eng   = vec_2.as_query_engine()
it_sup_cat_vec_eng  = vec_3.as_query_engine()
know_base_vec_eng   = vec_4.as_query_engine()
sample_conv_vec_eng = vec_5.as_query_engine()
test_req_vec_eng    = vec_6.as_query_engine()
troub_db_vec_eng    = vec_7.as_query_engine()

from llama_index.core.tools import QueryEngineTool

sum_comp_pol_tool = QueryEngineTool.from_defaults(
    query_engine=comp_pol_sum_eng,
    description=(
        "Generates a concise summary of IT policy sections, including password requirements, "
        "software installation rules, hardware request procedures, remote work security expectations, "
        "and security incident protocols. Use this tool for summarising a full section or getting an overview "
        "of how policies are structured and enforced."
    )
)

vec_comp_pol_tool = QueryEngineTool.from_defaults(
    query_engine=comp_pol_vec_eng,
    description=(
        "Answers specific questions by retrieving exact clauses or values from policy content—such as password length rules, "
        "lockout thresholds, software approval requirements, or the security incident reporting address. "
        "Use this tool when the query requires quoting a policy section verbatim or combining multiple exact details."
    )
)

sum_install_tool = QueryEngineTool.from_defaults(
    query_engine=inst_guid_sum_eng,
    description=(
        "Generates concise summaries of full installation guides, capturing the main steps, prerequisites, "
        "common setup issues, and available IT support. Use this tool to summarise a full guide—such as how to install Slack, "
        "set up Office 365, or connect to the company VPN."
    )
)

vec_install_tool = QueryEngineTool.from_defaults(
    query_engine=inst_guid_vec_eng,
    description=(
        "Answers questions by retrieving specific steps, error messages, configuration values, or support details "
        "from installation instructions. Use when the query requires exact steps, troubleshooting solutions, server URLs, "
        "or contact information."
    )
)

itsup_summ_tool = QueryEngineTool.from_defaults(
    query_engine=it_sup_cat_sum_eng,
    description=(
        "Summarises each IT support category by describing the types of issues covered, how long resolution typically takes, "
        "and whether escalation is required. Use this tool when a user requests a section-level overview or wants to compare "
        "how different categories (e.g. password reset vs. hardware failure) are handled."
    )
)

itsup_summ_vec_tool = QueryEngineTool.from_defaults(
    query_engine=it_sup_cat_vec_eng,
    description=(
        "Answers specific queries by retrieving exact descriptions, resolution time estimates, or escalation conditions "
        "from any IT support category. Use when a user needs exact details—like escalation triggers for network issues, "
        "resolution time for email setup, or listed causes under a category."
    )
)

knw_base_sum_tool = QueryEngineTool.from_defaults(
    query_engine=know_base_sum_eng,
    description=(
        "Summarises full sections of the IT support knowledge base, covering common problems and recommended actions "
        "for topics like password resets, software installation, hardware support, email configuration, and network issues. "
        "Use this tool when a user needs a concise overview of an entire topic."
    )
)

knw_base_vec_tool = QueryEngineTool.from_defaults(
    query_engine=know_base_vec_eng,
    description=(
        "Answers technical questions by retrieving exact troubleshooting steps, configuration details, or IT contact info—"
        "such as email server settings, reset URLs, approved installation procedures, or hardware replacement timelines. "
        "Use this tool for fact-level guidance or when step-by-step instructions are required."
    )
)

sapmle_con_sum_tool = QueryEngineTool.from_defaults(
    query_engine=sample_conv_sum_eng,
    description=(
        "Generates a summary of the full conversation dataset, highlighting the types of scenarios included, "
        "common support categories, user request patterns, and escalation handling. "
        "Use this tool to get a high-level understanding of the document as a whole."
    )
)

sample_con_vec_tool = QueryEngineTool.from_defaults(
    query_engine=sample_conv_vec_eng,
    description=(
        "Answers specific queries by retrieving exact fields from individual IT support conversations—"
        "such as the user message, assigned category, key response elements, or escalation details. "
        "Use this tool for targeted lookups when precise language or structured fields are required."
    )
)

test_req_sum_tool = QueryEngineTool.from_defaults(
    query_engine=test_req_sum_eng,
    description=(
        "Summarises example-based instructions that demonstrate how IT support requests should be interpreted and responded to—"
        "including how user intent maps to a classification, which response elements are relevant, and when escalation is needed. "
        "Use this tool to understand the overall logic and format behind the examples."
    )
)

test_req_vec_tool = QueryEngineTool.from_defaults(
    query_engine=test_req_vec_eng,
    description=(
        "Retrieves labeled examples that demonstrate how to handle IT support requests, including the original request text, "
        "expected classification, recommended response components, and escalation flags. "
        "Use this tool to fetch specific examples that guide how to structure answers to similar user queries."
    )
)

troub_db_sum_tool = QueryEngineTool.from_defaults(
    query_engine=troub_db_sum_eng,
    description=(
        "Summarises structured troubleshooting procedures across multiple IT issues—"
        "capturing the general workflow, escalation logic, and support contact structure. "
        "Use this tool to understand how troubleshooting is typically handled or to compare steps across different problems."
    )
)

troub_db_vec_tool = QueryEngineTool.from_defaults(
    query_engine=troub_db_vec_eng,
    description=(
        "Returns exact troubleshooting steps, escalation conditions, or support contact emails for a specific issue—"
        "such as password resets, slow computer performance, WiFi failures, or email sync problems. "
        "Use this tool when step-by-step resolution instructions or specific escalation details are needed."
    )
)

from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from llama_index.core.agent.react import ReActAgent
from llama_index.core.objects import ObjectIndex
from llama_index.core import VectorStoreIndex
from llama_index.core.memory import Memory

all_tools = [
    sum_comp_pol_tool, vec_comp_pol_tool,
    sum_install_tool, vec_install_tool,
    itsup_summ_tool, itsup_summ_vec_tool,
    knw_base_sum_tool, knw_base_vec_tool,
    sapmle_con_sum_tool, sample_con_vec_tool,
    test_req_sum_tool, test_req_vec_tool,
    troub_db_sum_tool, troub_db_vec_tool
]

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)
obj_retriever = obj_index.as_retriever(similarity_top_k=3)

agent = ReActAgent.from_tools(
    tools=all_tools,
    llm=Settings.llm,
    system_prompt=(
        "You are an IT support agent and helpful assistant that can answer questions about "
        "company IT policies, software installation guides, IT support categories related "
        "to the company, company knowledge base articles, sample conversations, and troubleshooting."
    ),
    obj_retriever=obj_retriever,
    max_iterations=60
)

# attach memory so multi‐turn works
agent.memory = Memory.from_defaults(token_limit=30000)

def get_agent_response_sync(user_message: str) -> str:
    """
    Blocking call that uses the chat‐style .chat() API.
    """
    resp = agent.chat(user_message)
    return resp.response if hasattr(resp, "response") else str(resp)
