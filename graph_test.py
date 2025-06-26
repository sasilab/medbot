from dotenv import load_dotenv
from typing import Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from typing_extensions import TypedDict

# 1. Import your hospital data/RAG helpers
from src.medbot.data_loader import (
    load_patient_details, load_diagnosis, load_medications, load_prescriptions,
    load_alerts, load_diabetic_indices, load_encounters, load_immunizations,
    combine_patient_documents
)
from src.medbot.helper import (
    create_chroma_vectorstore, create_chat_openai_llm, create_retrieval_qa_chain,
)

load_dotenv()

# 2. Agent State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    role: str
    permission_granted: bool

# 3. Role and Permission Logic
ROLE_PERMISSIONS = {
    "Nurse": ["Diagnosis", "Alerts", "Encounter History"],
    "Pharmacist": ["Prescriptions", "Medication Details"],
    "Doctor": "ALL",
    "Supervisor": "ALL"
}

# 4. Permission Checker Node
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

# 5. Initialize RAG Chain at Startup (before the chat loop!)
print("Loading patient data and initializing RAG...")
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
print(f"Loaded {len(documents)} patient documents.")
vectorstore = create_chroma_vectorstore(documents)
retriever = vectorstore.as_retriever()
llm = create_chat_openai_llm()
qa_chain = create_retrieval_qa_chain(llm, retriever)

# 6. RAG/LLM Node (use RAG, not direct LLM)
def hospital_agent(state: AgentState) -> Dict[str, Any]:
    last_message = state["messages"][-1]
    role = state["role"]
    permission = state["permission_granted"]

    if not permission:
        response = f"Access denied: As a {role}, you do not have permission to access this information."
    else:
        # RAG: Only answer from the retrieved hospital records!
        # Add system prompt for LLM to focus on role (optional)
        result = qa_chain.invoke({"query": last_message.content})
        response = result["result"]

    return {
        "messages": state["messages"] + [HumanMessage(content=response)],
        "role": role,
        "permission_granted": permission
    }

# 7. LangGraph Workflow
graph_builder = StateGraph(AgentState)
graph_builder.add_node("permission_checker", permission_checker)
graph_builder.add_node("hospital_agent", hospital_agent)
graph_builder.add_edge(START, "permission_checker")
graph_builder.add_edge("permission_checker", "hospital_agent")
graph_builder.add_edge("hospital_agent", END)
graph = graph_builder.compile()

# 8. Main Chat Loop
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
        print(f"Assistant: {answer.content}")

if __name__ == "__main__":
    run_hospital_agent()
