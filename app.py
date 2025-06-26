from src.medbot.data_loader import (
    load_patient_details,
    load_diagnosis,
    load_medications,
    load_prescriptions,
    load_alerts,
    load_diabetic_indices,
    load_encounters,
    load_immunizations,
    combine_patient_documents
)

from src.medbot.helper import (
    create_chroma_vectorstore,
    create_chat_openai_llm,
    create_retrieval_qa_chain,
    interactive_med_query
)

from src.medbot.hospital_agents import (
    load_users, authenticate, check_permission, log_event,
    classify_query_criticality, view_audit_log, create_langgraph_agent
)

from langchain_core.messages import HumanMessage
import getpass
import sys

def main():
    # Step 1: Load users and ask for login
    users = load_users(r"I:\Code Space\LLM Model Project\RAG\medbot\Data\user_credentials.csv")
    print("=== Hospital System Login ===")
    while True:
        if len(sys.argv) >= 3:
            username = sys.argv[1]
            password = sys.argv[2]
        else:
            
            username = input("Username: ").strip()
            password = getpass.getpass("Password: ").strip()

        role = authenticate(users, username, password)

        if role:
            print(f"Login successful. Role: {role}")
            break
        else:
            print("Invalid username or password. Try again.")

    # Step 2: Load data
    patients = load_patient_details(r"I:\Code Space\LLM Model Project\RAG\medbot\Data\patient_details.csv")
    diagnoses = load_diagnosis(r"I:\Code Space\LLM Model Project\RAG\medbot\Data\diagnosis.csv")
    medications = load_medications(r"I:\Code Space\LLM Model Project\RAG\medbot\Data\medications.csv")
    prescriptions = load_prescriptions(r"I:\Code Space\LLM Model Project\RAG\medbot\Data\prescriptions.csv")
    alerts = load_alerts(r"I:\Code Space\LLM Model Project\RAG\medbot\Data\alerts.csv")
    indices = load_diabetic_indices(r"I:\Code Space\LLM Model Project\RAG\medbot\Data\diabetic_indices.csv")
    encounters = load_encounters(r"I:\Code Space\LLM Model Project\RAG\medbot\Data\encounter_history.csv")
    immunizations = load_immunizations(r"I:\Code Space\LLM Model Project\RAG\medbot\Data\immunizations.csv")

    # Step 3: Combine documents
    documents = combine_patient_documents(
        patients, diagnoses, medications, prescriptions,
        alerts, indices, encounters, immunizations
    )

    # Step 4: Build vectorstore, retriever, LLM, and RAG chain
    vectorstore = create_chroma_vectorstore(documents)
    retriever = vectorstore.as_retriever()
    llm = create_chat_openai_llm()
    qa_chain = create_retrieval_qa_chain(llm, retriever)

    # Step 5: Create RAG LangGraph Agent for the role
    rag_agent = create_langgraph_agent(qa_chain, role)

    print("\n=== HOSPITAL ASSISTANT ===")
    print("Type 'exit' to quit. Supervisors can type 'auditlog' to view audit.")
    chat_history = []
    # Step 6: Chat loop
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() == "exit":
            print("Goodbye.")
            break

        if role == "Supervisor" and query.lower() == "auditlog":
            view_audit_log()
            continue

        # Permissions/Criticality Check
        allowed = check_permission(role, query)
        criticality = classify_query_criticality(query)

        if not allowed:
            print("Access denied: You do not have permission to access this information.")
            log_event(username, role, f"Denied query: {query}", critical=False)
            continue

        if criticality == "Critical":
            print("This query is marked as CRITICAL and will be logged for supervisor review.")
            log_event(username, role, f"Critical query: {query}", critical=True)

        initial_state = {
            "messages": [HumanMessage(content=query)],

        }

        chat_history.append(HumanMessage(content=query))
        initial_state = {"messages": chat_history.copy()}
        result = rag_agent.invoke(initial_state)
        # Find the last LLM (AI) message to display
        # (may need to search backwards if multiple tool calls)
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content"):
                print("\nANSWER:")
                print(msg.content)
                chat_history.append(msg)
                break

if __name__ == "__main__":
    main()
