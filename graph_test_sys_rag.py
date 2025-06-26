from dotenv import load_dotenv
from typing import Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import (
    BaseMessage, HumanMessage, SystemMessage, ToolMessage
)
from langchain_core.tools import tool
from typing_extensions import TypedDict

from src.medbot.data_loader import (
    load_patient_details, load_diagnosis, load_medications, load_prescriptions,
    load_alerts, load_diabetic_indices, load_encounters, load_immunizations,
    combine_patient_documents
)
from src.medbot.helper import (
    create_chroma_vectorstore, create_chat_openai_llm, create_retrieval_qa_chain,
)

load_dotenv()

# ----- System prompts per role -----
ROLE_SYSTEM_PROMPT = {
    "Nurse": (
        "You are a hospital nurse assistant. You ONLY answer about diagnosis, alerts, "
        "and encounter history. You never reveal medications or personal details, "
        "and never fabricate information."
    ),
    "Pharmacist": (
        "You are a hospital pharmacist assistant. You ONLY answer about prescriptions "
        "and medication details. Never discuss diagnoses or patient encounters. "
        "Use only the RAG tool to get data."
    ),
    "Doctor": (
        "You are a hospital doctor assistant. You may answer all questions except audit log details. "
        "Always use the RAG tool to answer from hospital records. Never make up answers."
    ),
    "Supervisor": (
        "You are a hospital supervisor assistant. You may access all patient data. "
        "Never share audit logs with non-supervisors. Always use the RAG tool."
    ),
}

ROLE_PERMISSIONS = {
    "Nurse": ["Diagnosis", "Alerts", "Encounter History"],
    "Pharmacist": ["Prescriptions", "Medication Details"],
    "Doctor": "ALL",
    "Supervisor": "ALL"
}

# ----- Agent State -----
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    role: str
    permission_granted: bool

# ----- Permission Checking -----
def permission_checker(state: AgentState) -> Dict[str, Any]:
    last_message = state["messages"][-1]
    role = state["role"]
    allowed = ROLE_PERMISSIONS.get(role, [])
    permission = False
    if allowed == "ALL":
        permission = True
    else:
        for field in allowed:
            if field.lower() in last_message.content.lower():
                permission = True
                break
    return {
        "messages": state["messages"],
        "role": role,
        "permission_granted": permission
    }

# ----- Load RAG -----
print("Loading hospital data and initializing vector store...")
patients = load_patient_details(r"I:\Code Space\LLM Model Project\RAG\medbot\Data\patient_details.csv")
diagnoses = load_diagnosis(r"I:\Code Space\LLM Model Project\RAG\medbot\Data\diagnosis.csv")
medications = load_medications(r"I:\Code Space\LLM Model Project\RAG\medbot\Data\medications.csv")
prescriptions = load_prescriptions(r"I:\Code Space\LLM Model Project\RAG\medbot\Data\prescriptions.csv")
alerts = load_alerts(r"I:\Code Space\LLM Model Project\RAG\medbot\Data\alerts.csv")
indices = load_diabetic_indices(r"I:\Code Space\LLM Model Project\RAG\medbot\Data\diabetic_indices.csv")
encounters = load_encounters(r"I:\Code Space\LLM Model Project\RAG\medbot\Data\encounter_history.csv")
immunizations = load_immunizations(r"I:\Code Space\LLM Model Project\RAG\medbot\Data\immunizations.csv")
documents = combine_patient_documents(
    patients, diagnoses, medications, prescriptions,
    alerts, indices, encounters, immunizations
)
vectorstore = create_chroma_vectorstore(documents)
retriever = vectorstore.as_retriever()
llm = create_chat_openai_llm()
qa_chain = create_retrieval_qa_chain(llm, retriever)

# ----- Define RAG as a Tool -----
@tool
def hospital_rag_tool(query: str) -> str:
    """Retrieve hospital information from patient records only for allowed fields."""
    result = qa_chain.invoke({"query": query})
    return result["result"]

TOOLS = [hospital_rag_tool]

# ----- LLM as agent, with system prompt and tools -----
def llm_agent_node(state: AgentState) -> AgentState:
    messages = list(state["messages"])
    role = state["role"]
    system_prompt = ROLE_SYSTEM_PROMPT.get(role, "You are a hospital AI assistant.")
    # Always start with system message
    messages_with_system = [SystemMessage(content=system_prompt)] + messages
    # Use tools
    agent_llm = llm.bind_tools(TOOLS)
    # Call LLM with messages (let LLM decide to call tool)
    message = agent_llm.invoke(messages_with_system)
    return {"messages": [message], "role": role, "permission_granted": state["permission_granted"]}

# ----- Tool node: executes any tool calls in LLM response -----
def tool_executor_node(state: AgentState) -> AgentState:
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        if t["name"] == hospital_rag_tool.name:
            tool_result = hospital_rag_tool.invoke(t["args"]["query"])
            results.append(
                ToolMessage(
                    tool_call_id=t["id"],
                    name=t["name"],
                    content=tool_result
                )
            )
        else:
            results.append(
                ToolMessage(
                    tool_call_id=t["id"],
                    name=t["name"],
                    content="Invalid tool call."
                )
            )
    return {"messages": results, "role": state["role"], "permission_granted": state["permission_granted"]}

# ----- Graph wiring -----
graph_builder = StateGraph(AgentState)
graph_builder.add_node("permission_checker", permission_checker)
graph_builder.add_node("llm_agent", llm_agent_node)
graph_builder.add_node("tool_executor", tool_executor_node)

graph_builder.add_edge(START, "permission_checker")
graph_builder.add_edge("permission_checker", "llm_agent")
# Loop: LLM can call tools or finish
def has_tool_calls(state: AgentState):
    return hasattr(state["messages"][-1], "tool_calls") and len(state["messages"][-1].tool_calls) > 0
graph_builder.add_conditional_edges(
    "llm_agent", has_tool_calls, {True: "tool_executor", False: END}
)
graph_builder.add_edge("tool_executor", "llm_agent")
graph = graph_builder.compile()

# ----- Main Chat Loop -----
def run_hospital_agent():
    print(" Hospital Assistant Chat (type 'exit' to quit)")
    role = input("Enter your role (Nurse/Pharmacist/Doctor/Supervisor): ").strip().title()
    state = {
        "messages": [],
        "role": role,
        "permission_granted": False
    }
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            print("Bye!")
            break
        state["messages"] = state.get("messages", []) + [HumanMessage(content=user_input)]
        state = graph.invoke(state)
        answer = state["messages"][-1]
        print(f"Assistant: {getattr(answer, 'content', answer)}")

if __name__ == "__main__":
    run_hospital_agent()
