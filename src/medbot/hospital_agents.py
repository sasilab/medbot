import csv
from datetime import datetime
from typing import Sequence, Annotated, Dict, TypedDict, List
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from operator import add as add_messages

# -----------------------------------
# 1. USER, ROLE, AND PERMISSION SETUP
# -----------------------------------

ROLE_PERMISSIONS = {
    "Nurse": {
        "fields": ["Name", "Sex", "DOB", "Diagnosis", "Alerts", "Encounter History"],
        "deny": ["Prescriptions", "Medication Details", "Personal Address", "NextOfKin", "Audit Log"]
    },
    "Pharmacist": {
        "fields": ["Name", "Medication Details", "Prescriptions"],
        "deny": ["Diagnosis", "Alerts", "Personal Address", "Encounter History", "NextOfKin", "Audit Log"]
    },
    "Doctor": {
        "fields": "ALL",
        "deny": ["Audit Log"]
    },
    "Supervisor": {
        "fields": "ALL",
        "deny": []
    }
}

AUDIT_LOG = []

def load_users(filepath=r"I:\Code Space\LLM Model Project\RAG\medbot\Data\user_credentials.csv"):
    users = {}
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            users[row['username']] = {"password": row["password"], "role": row["role"]}
    return users

def authenticate(users, username, password):
    user = users.get(username)
    if user and user["password"] == password:
        return user["role"]
    return None

def check_permission(role, query):
    allowed = ROLE_PERMISSIONS[role]["fields"]
    denied = ROLE_PERMISSIONS[role]["deny"]
    if allowed == "ALL":
        return True
    for d in denied:
        if d.lower() in query.lower():
            return False
    for f in allowed:
        if f.lower() in query.lower():
            return True
    return False

def log_event(username, role, event, critical=False):
    AUDIT_LOG.append({
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "username": username,
        "role": role,
        "event": event,
        "critical": critical
    })

def classify_query_criticality(query):
    critical_keywords = [
        "chest pain", "heart attack", "code blue", "seizure", "unconscious", "emergency",
        "suicide", "allergic reaction", "anaphylaxis", "stroke", "bleeding", "collapse"
    ]
    for word in critical_keywords:
        if word in query.lower():
            return "Critical"
    return "Normal"

def view_audit_log():
    if not AUDIT_LOG:
        print("No critical or denied events logged yet.")
        return
    print("\n=== AUDIT LOG ===")
    for entry in AUDIT_LOG:
        print(
            f"{entry['timestamp']} | {entry['username']} ({entry['role']}): {entry['event']} "
            f"{'[CRITICAL]' if entry['critical'] else ''}"
        )
    print("=================")

# -----------------------------------
# 2. AGENT STATE & LANGGRAPH SETUP
# -----------------------------------

class AgentState(TypedDict):
    messages: List[BaseMessage]
    

def should_continue(state: AgentState):
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

def build_system_prompt(role):
    # Each role gets a slightly different system prompt
    prompts = {
        "Nurse": "You are a hospital nurse AI assistant. Only provide diagnosis, alerts, basic patient info, and encounter history. Never show personal addresses, next-of-kin, or full prescriptions.",
        "Pharmacist": "You are a hospital pharmacist AI assistant. Only provide medication and prescription details, and patient names. Never provide diagnosis or personal/medical history.",
        "Doctor": "You are a hospital doctor AI assistant. You may access and discuss all medical information except audit logs.",
        "Supervisor": "You are a hospital supervisor. You can access all patient data and audit logs. Monitor for critical or illegal activities."
    }
    return prompts.get(role, "You are a hospital AI assistant.")

def make_rag_tool(qa_chain, allowed_fields):
    
    from langchain_core.tools import tool
    @tool
    def medical_rag_tool(query: str) -> str:
        """
        Retrieve allowed patient information for the hospital assistant.
        Only answers questions about permitted fields for the current role.
        """
        for f in allowed_fields if isinstance(allowed_fields, list) else []:
            if f.lower() in query.lower():
                res = qa_chain.invoke({"query": query})
                return res["result"]
        return "Access denied: You are not allowed to view this information."
    return medical_rag_tool

def create_langgraph_agent(qa_chain, role):
    from langchain_core.tools import tool

    allowed_fields = ROLE_PERMISSIONS[role]["fields"]
    rag_tool = make_rag_tool(qa_chain, allowed_fields)
    tools = [rag_tool]
    system_prompt = build_system_prompt(role)
    from src.medbot.helper import create_chat_openai_llm
    llm = create_chat_openai_llm().bind_tools(tools)

    def call_llm(state: AgentState) -> AgentState:
        # Always prepend system prompt, then all history
        messages = [SystemMessage(content=system_prompt)] + list(state['messages'])
        message = llm.invoke(messages)
        # Append to conversation history
        return {'messages': state['messages'] + [message]}

    def take_action(state: AgentState) -> AgentState:
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            result = tools[0].invoke(t['args'].get('query', ''))
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        # Append ToolMessages to the message history
        return {'messages': state['messages'] + results}

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever_agent", take_action)
    graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
    graph.add_edge("retriever_agent", "llm")
    graph.set_entry_point("llm")

    return graph.compile()
