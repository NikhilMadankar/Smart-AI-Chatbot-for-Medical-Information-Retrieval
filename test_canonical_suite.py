import os
import sys
import uuid
from dotenv import load_dotenv, find_dotenv
from agent import ask_medical_agent_stateful

# Ensure UTF-8 printing on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv(find_dotenv())


def run_canonical_test_suite():
    print("=" * 80)
    print("🧪 AGENT CANONICAL TEST SUITE (4 CORE PATHWAYS)")
    print("=" * 80)
    
    # 1. Medical Question ➔ RAG Tool
    print("\n[Flow 1] Medical Question ➔ RAG Tool")
    print("User: What are the diagnostic criteria and treatment of Diabetes Mellitus?")
    res1 = ask_medical_agent_stateful(
        "What are the diagnostic criteria and treatment of Diabetes Mellitus?",
        thread_id=f"flow1_{uuid.uuid4().hex[:4]}"
    )
    print("Tools Activated:", [t.get("name") for t in res1.get("tool_calls", [])])
    print("MediBot Answer:\n", res1.get("answer")[:250] + "...")
    
    # 2. BMI Question ➔ Calculator Tool
    print("\n" + "-" * 80)
    print("[Flow 2] BMI Question ➔ Calculator Tool")
    print("User: I weigh 80kg and my height is 175cm. Calculate my BMI.")
    res2 = ask_medical_agent_stateful(
        "I weigh 80kg and my height is 175cm. Calculate my BMI.",
        thread_id=f"flow2_{uuid.uuid4().hex[:4]}"
    )
    print("Tools Activated:", [t.get("name") for t in res2.get("tool_calls", [])])
    print("MediBot Answer:\n", res2.get("answer"))
    
    # 3. Symptom Question ➔ Pre-LLM Safety Check ➔ Symptom / RAG Tool
    print("\n" + "-" * 80)
    print("[Flow 3] Symptom Question ➔ Pre-LLM Safety Check ➔ Symptom / RAG Multi-Step")
    print("User: I have a mild fever and headache for 2 days. What could cause this?")
    res3 = ask_medical_agent_stateful(
        "I have a mild fever and headache for 2 days. What could cause this?",
        thread_id=f"flow3_{uuid.uuid4().hex[:4]}"
    )
    print("Tools Activated:", [t.get("name") for t in res3.get("tool_calls", [])])
    print("MediBot Answer:\n", res3.get("answer")[:300] + "...")

    # 4. Unknown Question ➔ No Hallucination / Fallback Disclaimer
    print("\n" + "-" * 80)
    print("[Flow 4] Unknown Question ➔ No Hallucination / Fallback Disclaimer")
    print("User: What is the exact gene mutation code for XYZ-999-Fictional Syndrome?")
    res4 = ask_medical_agent_stateful(
        "What is the exact gene mutation code for XYZ-999-Fictional Syndrome?",
        thread_id=f"flow4_{uuid.uuid4().hex[:4]}"
    )
    print("Tools Activated:", [t.get("name") for t in res4.get("tool_calls", [])])
    print("Evidence Sufficient:", res4.get("evidence_sufficient"))
    print("MediBot Answer:\n", res4.get("answer"))
    print("=" * 80)


if __name__ == "__main__":
    run_canonical_test_suite()
