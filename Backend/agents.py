import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import InferenceClient
from langchain_community.chat_message_histories import FileChatMessageHistory
from tools import search_3gpp_standards, escalate_to_human

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SESSIONS_DIR = PROJECT_ROOT / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

class CustomAgentExecutor:
    def __init__(self, tools):
        api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")
        self.client = InferenceClient(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            token=api_token
        )
        self.tools = {tool.name: tool for tool in tools}

    def invoke(self, inputs, config=None):
        user_input = inputs.get("input", "")
        session_id = config.get("configurable", {}).get("session_id", "default") if config else "default"
        
        history = self._get_history(session_id)
        response_text = ""
        
        try:
            lower_input = user_input.lower()
            
            # 1. Handle Human Escalation / Out-of-Scope Guardrails explicitly
            if any(term in lower_input for term in ["cookie", "recipe", "weather", "movie", "satellite orbital", "6g satellite"]):
                escalation_res = escalate_to_human.invoke(reason=f"User query '{user_input}' falls outside 3GPP technical specification scope.")
                response_text = f"I am a specialized 3GPP Telecom standards assistant and cannot answer queries outside the specification scope.\n\n{escalation_res}"

            # 2. Handle General Greetings or Intro Queries naturally
            elif any(q in lower_input for q in ["hello", "hi", "who are you", "what do you do", "what is 3gpp"]):
                response_text = (
                    "Hello! I am your AI-powered **3GPP Telecom Standards Assistant**. "
                    "I am connected to the 1,510-page 3GPP specification database (TS 23.501) to help you analyze "
                    "5G core network architecture, reference points, registration management, and QoS procedures. How can I help you today?"
                )
                
            # 3. Handle Technical Specification Queries via Vector Search + Synthesis
            else:
                tool_result = search_3gpp_standards.invoke(user_input)
                
                if "No relevant information found" in tool_result:
                    response_text = escalate_to_human.invoke(reason=f"Query '{user_input}' yielded no results in 3GPP docs.")
                else:
                    try:
                        prompt_text = f"""<s>[INST] You are a professional 3GPP Telecom Expert. Read the technical specification context below and write a clear, polished, structured explanation answering the user's question. Do not output raw chunks or metadata tags; synthesize the explanation naturally.

Context Chunks:
{tool_result}

User Question: {user_input}
[/INST]"""
                        
                        response_text = self.client.text_generation(
                            prompt=prompt_text,
                            max_new_tokens=1024,
                            temperature=0.01
                        )
                    except Exception as cloud_err:
                        cleaned_context = tool_result.replace("Source section", "• **Section**").replace(":", " —")
                        response_text = f"Here is what the 3GPP Technical Specification states regarding your query:\n\n{cleaned_context[:1200]}"
            
            if not response_text.strip():
                response_text = "I processed your request, but received an empty response. Please try rephrasing."
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            response_text = f"An error occurred during processing: {str(e)}"

        history.add_user_message(user_input)
        history.add_ai_message(response_text)
        
        return {"output": response_text}

    def _get_history(self, session_id: str):
        file_path = SESSIONS_DIR / f"{session_id}.json"
        return FileChatMessageHistory(str(file_path))

def build_agent():
    print("Initializing Polished Custom Agent Executor...")
    tools = [search_3gpp_standards, escalate_to_human]
    return CustomAgentExecutor(tools)


if __name__ == "__main__":
    print("--- Initializing Agent for Standalone Test ---")
    agent = build_agent()
    
    # Simulate a user test payload
    test_payload = {
        "input": "What does the 3GPP specification say about registration management?",
        "config": {
            "configurable": {
                "session_id": "test_session_123"
            }
        }
    }
    
    print(f"\nSending test prompt to agent: '{test_payload['input']}'...\n")
    
    try:
        response = agent.invoke(
            {"input": test_payload["input"]},
            config=test_payload["config"]
        )
        print("\n--- AGENT FINAL RESPONSE ---")
        print(response.get("output", "No output key found."))
    except Exception as e:
        print(f"\nAgent execution failed with error: {e}")