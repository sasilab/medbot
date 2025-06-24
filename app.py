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




def main():
    # Load data
    patients = load_patient_details(r"I:\Code Space\LLM Model Project\RAG\medbot\Data\patient_details.csv")
    diagnoses = load_diagnosis(r"I:\Code Space\LLM Model Project\RAG\medbot\Data\diagnosis.csv")
    medications = load_medications(r"I:\Code Space\LLM Model Project\RAG\medbot\Data\medications.csv")
    prescriptions = load_prescriptions(r"I:\Code Space\LLM Model Project\RAG\medbot\Data\prescriptions.csv")
    alerts = load_alerts(r"I:\Code Space\LLM Model Project\RAG\medbot\Data\alerts.csv")
    indices = load_diabetic_indices(r"I:\Code Space\LLM Model Project\RAG\medbot\Data\diabetic_indices.csv")
    encounters = load_encounters(r"I:\Code Space\LLM Model Project\RAG\medbot\Data\encounter_history.csv")
    immunizations = load_immunizations(r"I:\Code Space\LLM Model Project\RAG\medbot\Data\immunizations.csv")

    # Combine
    documents = combine_patient_documents(
        patients, diagnoses, medications, prescriptions,
        alerts, indices, encounters, immunizations
    )


    vectorstore = create_chroma_vectorstore(documents)
    retriever = vectorstore.as_retriever()
    llm = create_chat_openai_llm()
    qa_chain = create_retrieval_qa_chain(llm, retriever)
    interactive_med_query(qa_chain)

if __name__ == "__main__":
    main()