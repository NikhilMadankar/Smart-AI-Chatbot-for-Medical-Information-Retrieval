import os
import sys
import uuid
import time
from dotenv import load_dotenv, find_dotenv


from agent import ask_medical_agent_stateful

# Ensure UTF-8 printing on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv(find_dotenv())

def run_bmi_tests():
    print("=" * 80)
    print("🧪 EXECUTING COMPREHENSIVE BMI WORKFLOW TEST SUITE")
    print("=" * 80)

    test_cases = [
        {
            "id": 1,
            "query": "I'm 50 kg and 170 cm. What is my BMI?",
            "expected_calc_calls": 1,
            "must_contain": ["17.3"]
        },
        {
            "id": 2,
            "query": "Calculate my BMI. I weigh 70 kg and I'm 175 cm.",
            "expected_calc_calls": 1,
            "must_contain": ["22.9"]
        },
        {
            "id": 3,
            "query": "Calculate my BMI.",
            "expected_calc_calls": 1,
            "must_contain": ["weight", "height"]
        },
        {
            "id": 4,
            "query": "How can I gain weight?",
            "expected_calc_calls": 0,
            "must_contain": []
        },
        {
            "id": 5,
            "query": "What is BMI?",
            "expected_calc_calls": 0,
            "must_contain": []
        },
        {
            "id": 6,
            "query": "My BMI is 17.3, what does it mean?",
            "expected_calc_calls": 0,
            "must_contain": ["17.3"]
        }
    ]

    all_passed = True

    for test in test_cases:
        time.sleep(10)


        print(f"\n--- TEST CASE {test['id']}: \"{test['query']}\" ---")
        session_id = str(uuid.uuid4())
        res = ask_medical_agent_stateful(test['query'], thread_id=session_id)

        
        answer = res.get("answer", "")
        tool_results = res.get("tool_results", [])
        
        calc_calls = sum(1 for t in tool_results if t.get("tool") == "medical_calculator")
        print(f"• Medical Calculator Executions: {calc_calls} (Expected: {test['expected_calc_calls']})")
        print(f"• Answer Snippet:\n{answer[:250]}...\n")
        
        calls_ok = (calc_calls == test['expected_calc_calls'])
        content_ok = all(term.lower() in answer.lower() for term in test["must_contain"])
        
        passed = calls_ok and content_ok
        if not passed:
            all_passed = False
        print(f"Result: {'✅ PASSED' if passed else '❌ FAILED'}")

    print("\n" + "=" * 80)
    print(f"🏆 OVERALL TEST SUITE RESULT: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print("=" * 80)

if __name__ == "__main__":
    run_bmi_tests()
