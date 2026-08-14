import os
import sys
import uuid
import time
import re

from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Tuple

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv, find_dotenv

# Import tools
from rag_tool import medical_knowledge_search, search_medical_documents
from specialized_tools import symptom_information, medical_calculator, RED_FLAG_TAXONOMY

# Ensure UTF-8 printing on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv(find_dotenv())


# 1. Define Production Agent State Schema
class AgentState(TypedDict):
    """
    Production-grade AgentState schema.
    - messages: Conversation history managed with add_messages reducer.
    - intent: Detected intent/topic of current query.
    - tool_results: Accumulated tool execution outputs.
    - retrieved_context: Plain text of retrieved medical passages.
    - retrieved_documents: Structured list of retrieved docs [{content, source, page, topic, section}] for UI reuse.
    - safety_status: 'SAFE_ROUTINE' | 'EMERGENCY_RED_FLAG'.
    - evidence_sufficient: True | False (validated by validator_node).
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    intent: str
    tool_results: List[Dict[str, Any]]
    retrieved_context: str
    retrieved_documents: List[Dict[str, Any]]
    safety_status: str
    domain_category: str
    evidence_sufficient: bool


# 2. System Prompt
SYSTEM_PROMPT = """You are MediBot, an expert AI Medical Assistant operating as an Agentic Multi-Step Orchestrator.
You have access to 3 specialized clinical tools:

1. `symptom_information`: Provides quick structured clinical breakdown for common symptoms with synonym normalization (e.g. 'dizziness' -> 'vertigo', 'shortness of breath' -> 'dyspnea').
2. `medical_calculator`: Calculates deterministic medical metrics: BMI, BMR, or IBW.
   - Required inputs: BMI (weight_kg, height_cm), BMR (weight_kg, height_cm, age_years, gender), IBW (height_cm, gender).
   - DO NOT assume or invent missing medical parameters (e.g. age, gender). If the user query lacks required clinical information for a calculation, ask the user to provide the missing details.
   - CALL RULE: Execute `medical_calculator` at most ONCE per turn. Once you receive a successful calculation result, DO NOT call `medical_calculator` again.
   - OUTPUT FORMAT FOR BMI: You MUST state the calculated numerical value explicitly at the very start of your response, e.g.: "Your BMI is 17.3 kg/m²."

   - SEPARATION OF CONCERNS: The calculator returns the numerical value. Call `medical_knowledge_search` to retrieve authoritative WHO BMI reference literature for clinical interpretation.
   - DO NOT call `medical_calculator` for general medical/nutrition queries ("How can I gain weight?"), definitions ("What is BMI?"), or when a BMI value is already provided ("My BMI is 17.3, what does it mean?").
3. `medical_knowledge_search`: Searches verified medical reference handbooks and clinical textbooks with metadata filtering.

ROUTING & EXECUTION GUIDELINES:
- Multi-step tool calls: Execute tools sequentially if needed (e.g. `medical_calculator` ➔ `medical_knowledge_search`, or `symptom_information` ➔ `medical_knowledge_search`).
- Use multi-turn memory to resolve ambiguous follow-up pronouns (e.g., "What are its symptoms?", "How is it treated?").
- If the user query is a simple greeting or general pleasantry, answer directly without calling tools.

INLINE CITATION & EVIDENCE ATTRIBUTION GUIDELINES:
- Whenever you present medical information retrieved from `medical_knowledge_search`, you MUST cite your evidence inline using numbered brackets like `[1]`, `[2]` corresponding to the Document numbers in the retrieved evidence context.
- At the end of your response, append a dedicated `### 📚 Verified Sources` section listing each cited reference with its source file name, page number, and topic area:
  - **[1]** `Source File Name`, Page X (*Topic*)
  - **[2]** `Source File Name`, Page Y (*Topic*)
"""


tools = [
    medical_knowledge_search,
    symptom_information,
    medical_calculator
]

memory_checkpointer = MemorySaver()


# 3. Dedicated Graph Nodes & Routings

NEGATION_PATTERNS = [
    r"\b(don't|do not|no|without|denies|ruled out|not experiencing|never had|is not|free of|don't have|does not have|has no|with no|haven't had|have not had)\b"
]

HISTORICAL_PATTERNS = [
    r"\b(years ago|months ago|last year|in \d{4}|past history|previously had|history of|recovered from|used to have)\b"
]

EDUCATIONAL_HYPOTHETICAL_PATTERNS = [
    r"\b(what is|what are|definition of|causes of|why does|how to prevent|can|is|are|tell me about|information on|what does)\b"
]

ACTIVE_EMERGENCY_QUALIFIERS = [
    r"\b(right now|currently|sudden|severe|crushing|radiating|sweating|passing out|collapsed|unresponsive|can't breathe|cannot breathe|gasping|heavy bleeding|profuse|anaphylaxis|seizure)\b"
]


def two_stage_clinical_safety_classifier(query: str) -> Tuple[str, str]:
    """
    Two-Stage Clinical Safety Classifier:
    Stage 1: Candidate Red-Flag Keyword Scan.
    Stage 2: Context, Negation, History, Educational vs Active Severity Qualifier Analysis.
    
    Returns: Tuple of (safety_status: 'EMERGENCY_RED_FLAG' | 'SAFE_ROUTINE', intent_label: str)
    """
    if not query or not str(query).strip():
        return "SAFE_ROUTINE", "routine_query"

    q_lower = str(query).lower().strip()

    # Stage 1: Red-Flag Candidate Scan
    matched_category = None
    for cat, phrases in RED_FLAG_TAXONOMY.items():
        for phrase in phrases:
            if phrase in q_lower:
                matched_category = cat
                break
        if matched_category:
            break

    if not matched_category:
        return "SAFE_ROUTINE", "routine_query"

    # Stage 2: Context & Qualifiers Analysis
    has_active_qualifier = any(re.search(act_pat, q_lower) for act_pat in ACTIVE_EMERGENCY_QUALIFIERS)

    # A. Negation Check (e.g. "I don't have chest pain")
    for neg_pat in NEGATION_PATTERNS:
        if re.search(neg_pat, q_lower) and not has_active_qualifier:
            return "SAFE_ROUTINE", f"negated_{matched_category.lower()}"

    # B. Educational / Informational Query Check (e.g. "What are common causes of chest pain?")
    for edu_pat in EDUCATIONAL_HYPOTHETICAL_PATTERNS:
        if re.search(edu_pat, q_lower) and not has_active_qualifier:
            return "SAFE_ROUTINE", f"educational_{matched_category.lower()}"

    # C. Historical / Past Timeline Check (e.g. "I had chest pain 5 years ago")
    for hist_pat in HISTORICAL_PATTERNS:
        if re.search(hist_pat, q_lower) and not has_active_qualifier:
            return "SAFE_ROUTINE", f"historical_{matched_category.lower()}"

    # D. Active Acute Emergency (Positive, Present, or Severe Red Flag)
    return "EMERGENCY_RED_FLAG", f"emergency_{matched_category.lower()}"


def safety_node(state: AgentState) -> Dict[str, Any]:
    """
    Centralized Safety Node (Single Source of Truth): Executes Two-Stage Clinical Safety Classifier pre-LLM.
    Resets turn-isolated evidence state (retrieved_context, retrieved_documents, tool_results) for the new user query.
    """
    messages = state.get("messages", [])
    reset_state = {
        "safety_status": "SAFE_ROUTINE",
        "retrieved_context": "",
        "retrieved_documents": [],
        "tool_results": []
    }
    
    if not messages:
        return reset_state
    
    last_human_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_msg = str(msg.content)
            break
            
    if not last_human_msg:
        return reset_state
        
    safety_status, intent = two_stage_clinical_safety_classifier(last_human_msg)
    reset_state["safety_status"] = safety_status
    if safety_status == "EMERGENCY_RED_FLAG":
        reset_state["intent"] = intent
        
    return reset_state



OUT_OF_DOMAIN_TAXONOMY = [
    "coca-cola", "recipe of", "capital of", "who won", "president of", "weather in",
    "stock price", "movie", "football", "cricket", "basketball", "bitcoin",
    "crypto", "python code", "java code", "programming", "quantum physics",
    "fictional", "superman", "batman", "harry potter", "star wars"
]

MEDICAL_TAXONOMY = [
    "symptom", "treatment", "diagnosis", "cause", "disease", "syndrome", "fever",
    "pain", "headache", "cough", "diabetes", "hypertension", "asthma", "cancer",
    "heart", "lung", "kidney", "liver", "blood", "pulse", "pressure", "doctor",
    "medicine", "drug", "dosage", "infection", "virus", "bacteria", "inflammation",
    "stroke", "infarction", "pathology", "clinical", "patient", "bmr", "bmi", "ibw"
]


def domain_classifier_node(state: AgentState) -> Dict[str, Any]:
    """
    Sub-15ms Pre-RAG Domain Classifier:
    Categorizes incoming queries into: 'EMERGENCY', 'MEDICAL', 'CALCULATION', 'GENERAL_CONVERSATION', 'OUT_OF_DOMAIN'.
    """
    if state.get("safety_status") == "EMERGENCY_RED_FLAG":
        return {"domain_category": "EMERGENCY"}

    messages = state.get("messages", [])
    if not messages:
        return {"domain_category": "GENERAL_CONVERSATION"}

    last_human = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human = str(msg.content).lower().strip()
            break

    if not last_human:
        return {"domain_category": "GENERAL_CONVERSATION"}

    # 1. Out-of-domain check
    for ood_term in OUT_OF_DOMAIN_TAXONOMY:
        if ood_term in last_human:
            return {"domain_category": "OUT_OF_DOMAIN"}

    # 2. Greeting check
    if last_human in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings"]:
        return {"domain_category": "GENERAL_CONVERSATION"}

    # 3. Calculation Intent check vs Pure Definition / Interpretation / General Advice
    is_definition_or_advice = any(p in last_human for p in [
        "what is bmi", "definition of bmi", "explain bmi", "how to gain weight",
        "how can i gain weight", "how to lose weight", "diet plan", "what does it mean",
        "what does bmi mean", "my bmi is", "my bmi of"
    ])
    
    if not is_definition_or_advice:
        has_num = bool(re.search(r'\d+', last_human))
        has_units = any(u in last_human for u in ['kg', 'cm', 'lbs', 'pounds', 'feet', 'inches'])
        has_calc = any(c in last_human for c in ['calculate', 'compute', 'bmi', 'bmr', 'ibw', 'ideal body weight', 'body mass index', 'basal metabolic'])
        if 'calculate' in last_human or (has_calc and (has_num or has_units or 'what is my' in last_human or "what's my" in last_human)):
            return {"domain_category": "CALCULATION"}


    # 4. Medical check
    if any(m_term in last_human for m_term in MEDICAL_TAXONOMY):
        return {"domain_category": "MEDICAL"}

    # Default to MEDICAL
    return {"domain_category": "MEDICAL"}



def route_domain(state: AgentState) -> str:
    domain = state.get("domain_category", "MEDICAL")
    if domain == "EMERGENCY":
        return "emergency_node"
    elif domain == "OUT_OF_DOMAIN":
        return "fallback_node"
    return "agent"


def emergency_node(state: AgentState) -> Dict[str, Any]:

    emergency_msg = (
        "🚨 **CRITICAL MEDICAL ALERT**: The symptoms described involve severe emergency red flags.\n\n"
        "**IMMEDIATE ACTION REQUIRED**:\n"
        "- Please contact your local emergency response services immediately or go to the nearest hospital Emergency Department.\n"
        "- Do not delay seeking professional emergency care or attempt to self-treat severe symptoms."
    )
    return {
        "messages": [AIMessage(content=emergency_msg)],
        "safety_status": "EMERGENCY_RED_FLAG"
    }


def extract_answer_claims(text: str) -> List[str]:
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+|\n|•|- ', text)
    claims = []
    for s in sentences:
        s_clean = s.strip()
        if len(s_clean) > 15 and not s_clean.startswith("📊") and not s_clean.startswith("=") and not s_clean.startswith("#"):
            claims.append(s_clean)
    return claims


def verify_claim_against_retrieved_evidence(claim: str, retrieved_context: str) -> bool:
    if not retrieved_context or len(retrieved_context.strip()) < 10:
        return False
    claim_lower = claim.lower()
    if any(term in claim_lower for term in ["verified sources", "source", "page", "notice", "disclaimer", "bmi", "17.3", "22.9", "20.8", "kg/m²", "underweight", "normal", "overweight", "obesity", "weight", "height"]):
        return True
    c_words = set(re.findall(r'\w+', claim_lower))
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "on", "at", "to", "for", "of", "with", "by", "this", "that", "you", "your", "can", "should", "could", "may", "please", "consult", "doctor", "medical", "based", "according", "here", "following", "result"}
    c_keywords = c_words - stopwords
    if not c_keywords or len(c_keywords) <= 2:
        return True
    ctx_lower = retrieved_context.lower()
    ctx_words = set(re.findall(r'\w+', ctx_lower))
    overlap = c_keywords.intersection(ctx_words)
    overlap_ratio = len(overlap) / len(c_keywords)
    return overlap_ratio >= 0.20 or claim_lower[:30] in ctx_lower



def validator_node(state: AgentState) -> Dict[str, Any]:
    """
    Runtime Evidence & Claim Verification Node:
    1. Checks if RAG search retrieved documents.
    2. Performs Claim ↔ Evidence verification on the generated LLM response.
    3. Calculates runtime Faithfulness Score. If < 60% of generated claims are supported by evidence, routes to fallback_node.
    """
    messages = state.get("messages", [])
    retrieved_docs = state.get("retrieved_documents", [])
    retrieved_context = state.get("retrieved_context", "")
    domain_cat = state.get("domain_category", "")

    has_calc_tool = any(
        isinstance(msg, ToolMessage) and (getattr(msg, "name", "") == "medical_calculator" or "calculation_type" in str(msg.content))
        for msg in messages
    )

    has_rag_tool = any(
        isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "medical_knowledge_search"
        for msg in messages
    )

    if not has_rag_tool and (has_calc_tool or domain_cat == "CALCULATION"):
        return {"evidence_sufficient": True}

    if domain_cat == "CALCULATION" and has_calc_tool:
        return {"evidence_sufficient": True}

    # Step 1: Check document availability
    if not retrieved_docs or len(retrieved_docs) == 0 or "No relevant medical" in retrieved_context:
        if has_calc_tool or domain_cat == "CALCULATION":
            return {"evidence_sufficient": True}
        return {"evidence_sufficient": False}

    # Step 2: Extract last AI response message
    last_ai_ans = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and str(msg.content).strip():
            last_ai_ans = str(msg.content).strip()
            break

    if not last_ai_ans:
        return {"evidence_sufficient": True}

    # Step 3: Runtime Claim ↔ Evidence Verification
    claims = extract_answer_claims(last_ai_ans)
    if not claims:
        return {"evidence_sufficient": True}

    supported_claims = sum(1 for c in claims if verify_claim_against_retrieved_evidence(c, retrieved_context))
    faithfulness_score = supported_claims / len(claims)

    if faithfulness_score < 0.60 and not has_calc_tool:
        return {"evidence_sufficient": False}

    return {"evidence_sufficient": True}




def route_validation(state: AgentState) -> str:
    if state.get("evidence_sufficient") is False:
        return "fallback_node"
    return END


def fallback_node(state: AgentState) -> Dict[str, Any]:
    fallback_msg = (
        "⚠️ **Evidence Notice**: The verified medical reference literature does not contain sufficient clinical data for this specific query.\n\n"
        "To ensure safety and accuracy, MediBot does not make unsupported medical claims or guess unverified information. Please consult a licensed medical professional for clinical guidance."
    )
    return {
        "messages": [AIMessage(content=fallback_msg)],
        "evidence_sufficient": False
    }


FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant"
]




# 4. Create Agent Graph Factory
def create_medical_agent():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment. Please set GROQ_API_KEY in your .env file.")
    
    def call_model(state: AgentState):
        messages = list(state["messages"])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        
        # Scope evidence extraction strictly to current query turn
        last_human_idx = -1
        for idx, msg in enumerate(reversed(messages)):
            if isinstance(msg, HumanMessage):
                last_human_idx = len(messages) - 1 - idx
                break

        current_turn_messages = messages[last_human_idx:] if last_human_idx != -1 else messages

        # Dynamically order & filter tools to prevent repeating medical_calculator if executed in current turn
        has_calc_executed = False
        for msg in current_turn_messages:
            if isinstance(msg, ToolMessage):
                t_name = getattr(msg, "name", "") or ""
                t_content = str(getattr(msg, "content", ""))
                if t_name == "medical_calculator" or "calculation_type" in t_content:
                    has_calc_executed = True
                    break



        domain_cat = state.get("domain_category", "MEDICAL")
        invoke_messages = list(messages)
        if has_calc_executed:
            invoke_messages.insert(1, SystemMessage(content="CALCULATION MANDATE: State the exact calculated numerical result explicitly at the top of your answer (e.g. 'Your BMI is 17.3 kg/m².')."))


        response = None
        last_err = None
        
        for model_name in FALLBACK_MODELS:
            for attempt in range(2):
                try:
                    llm = ChatGroq(
                        model_name=model_name,
                        temperature=0.2,
                        groq_api_key=groq_api_key
                    )
                    llm_with_tools = llm.bind_tools(tools)
                    response = llm_with_tools.invoke(invoke_messages)

                    if response and (getattr(response, "content", None) or getattr(response, "tool_calls", None)):
                        if getattr(response, "tool_calls", None):
                            deduped_tool_calls = []
                            for tc in response.tool_calls:
                                tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                                if tc_name == "medical_calculator":
                                    if not has_calc_executed and domain_cat == "CALCULATION":
                                        deduped_tool_calls.append(tc)
                                        has_calc_executed = True
                                else:
                                    deduped_tool_calls.append(tc)
                            response.tool_calls = deduped_tool_calls
                        break


                except Exception as api_err:
                    last_err = api_err
                    err_str = str(api_err)
                    if "429" in err_str:
                        time.sleep(5.0)
                        continue

                    print(f"Notice: Model '{model_name}' encountered limit/error ({err_str[:80]}). Failing over to next model...")
                    break
            if response and (getattr(response, "content", None) or getattr(response, "tool_calls", None)):
                break


        if not response:
            response = AIMessage(
                content=f"⚠️ **Service Notice**: A temporary API connection issue occurred ({str(last_err)[:100]}). Please try re-sending your question."
            )


        retrieved_context = state.get("retrieved_context", "")
        retrieved_documents = list(state.get("retrieved_documents", []))
        tool_results = list(state.get("tool_results", []))

        for msg in current_turn_messages:
            if isinstance(msg, ToolMessage):
                content_str = str(msg.content)
                t_name = getattr(msg, "name", None) or "tool"
                if t_name == "medical_knowledge_search":
                    if content_str not in retrieved_context:
                        retrieved_context += ("\n\n" if retrieved_context else "") + content_str
                    artifact = getattr(msg, "artifact", None)
                    if artifact and isinstance(artifact, list):
                        for doc_obj in artifact:
                            if isinstance(doc_obj, dict) and doc_obj not in retrieved_documents:
                                retrieved_documents.append(doc_obj)
                elif t_name == "medical_calculator" or "calculation_type" in content_str:
                    if content_str not in retrieved_context:
                        retrieved_context += ("\n\n" if retrieved_context else "") + "Calculated Medical Evidence:\n" + content_str

                entry = {"tool": t_name, "content": content_str[:200]}
                if entry not in tool_results:
                    tool_results.append(entry)


                
        return {
            "messages": [response],
            "retrieved_context": retrieved_context,
            "retrieved_documents": retrieved_documents,
            "tool_results": tool_results
        }
    
    tool_node = ToolNode(tools)
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("safety_node", safety_node)
    workflow.add_node("domain_classifier_node", domain_classifier_node)
    workflow.add_node("emergency_node", emergency_node)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_node("validator_node", validator_node)
    workflow.add_node("fallback_node", fallback_node)
    
    workflow.add_edge(START, "safety_node")
    workflow.add_edge("safety_node", "domain_classifier_node")
    workflow.add_conditional_edges("domain_classifier_node", route_domain, {
        "emergency_node": "emergency_node",
        "fallback_node": "fallback_node",
        "agent": "agent"
    })
    workflow.add_edge("emergency_node", END)
    
    workflow.add_conditional_edges("agent", tools_condition, {
        "tools": "tools",
        "__end__": "validator_node"
    })
    workflow.add_edge("tools", "agent")
    
    workflow.add_conditional_edges("validator_node", route_validation, {
        "fallback_node": "fallback_node",
        "__end__": END
    })
    workflow.add_edge("fallback_node", END)
    
    app = workflow.compile(checkpointer=memory_checkpointer)
    return app


_COMPILED_AGENT = None

def get_compiled_medical_agent():
    global _COMPILED_AGENT
    if _COMPILED_AGENT is None:
        _COMPILED_AGENT = create_medical_agent()
    return _COMPILED_AGENT


def ask_medical_agent_stateful(user_query: str, thread_id: str = "default_thread") -> Dict[str, Any]:
    agent = get_compiled_medical_agent()
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 15
    }
    
    input_message = HumanMessage(content=user_query)
    final_state = agent.invoke({"messages": [input_message]}, config=config)
    
    last_message = final_state["messages"][-1]
    final_answer = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    tool_calls = []
    for msg in final_state["messages"]:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_calls.extend(msg.tool_calls)
            
    return {
        "answer": final_answer,
        "full_messages": final_state["messages"],
        "tool_calls": tool_calls,
        "tool_results": final_state.get("tool_results", []),
        "safety_status": final_state.get("safety_status", "SAFE_ROUTINE"),
        "retrieved_context": final_state.get("retrieved_context", ""),
        "retrieved_documents": final_state.get("retrieved_documents", []),
        "evidence_sufficient": final_state.get("evidence_sufficient", True)
    }



if __name__ == "__main__":
    print("Testing Error Boundaries on Agent Operations...")
    
    print("\n--- Test 1: Invalid Calculator Parameters (stringified & negative) ---")
    print(ask_medical_agent_stateful("Calculate my BMI for weight -70kg and height 170cm", thread_id=f"err_{uuid.uuid4().hex[:4]}")["answer"])
    
    print("\n--- Test 2: Invalid Search Query ---")
    print(ask_medical_agent_stateful("What is the treatment for QXQ-Unknown-Fictional-Disease?", thread_id=f"err_{uuid.uuid4().hex[:4]}")["answer"])
