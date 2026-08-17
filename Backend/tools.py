from langchain_core.tools import tool

# Since tools.py and rag.py are in the same 'backend' folder, 
# we can import get_retriever directly.
from rag import get_retriever

@tool
def search_3gpp_standards(query: str) -> str:
    """
    Search the 3GPP telecom standards document for technical answers.
    ALWAYS use this tool first when asked a technical question about telecom, networks, or 3GPP.
    """
    try:
        # Fetch the retriever we built in rag.py
        retriever = get_retriever()
        
        # Search the vector database for the top matches
        docs = retriever.invoke(query)
        
        if not docs:
            return "No relevant information found in the 3GPP document."
        
        # Combine the retrieved chunks into a clean context string for the LLM to read
        return "\n\n".join([f"Source section ({d.metadata.get('section')}):\n{d.page_content}" for d in docs])
        
    except Exception as e:
        return f"Error accessing the 3GPP database: {str(e)}"


@tool
def escalate_to_human(reason: str) -> str:
    """
    Use this tool ONLY if the answer is not found in the 3GPP document, 
    or if the user explicitly asks to speak to a human engineer.
    """
    # In a real enterprise application, this would trigger a Jira ticket or send an email.
    # For this Mavenir assignment, returning a simulated ticket ID is perfect.
    ticket_id = "TKT-10492" 
    
    return (
        f"I have escalated this issue to a human engineer. "
        f"Reference Ticket: {ticket_id}. "
        f"Reason for escalation: {reason}"
    )

if __name__ == "__main__":
    print("--- Testing Tool 1: 3GPP Document Retriever ---")
    test_query = "What is the architecture for 5G core network?"
    print(f"Executing query: '{test_query}'...\n")
    
    try:
        result = search_3gpp_standards.invoke(test_query)
        print("RESULT:")
        print(result[:500] + "...\n[Output truncated for length]" if len(result) > 500 else result)
    except Exception as e:
        print(f"Tool 1 Failed with Error: {e}")

    print("\n" + "="*50 + "\n")
    
    print("--- Testing Tool 2: Human Escalation ---")
    try:
        escalation_result = escalate_to_human.invoke("Test escalation reason due to missing data.")
        print("RESULT:")
        print(escalation_result)
    except Exception as e:
        print(f"Tool 2 Failed with Error: {e}")