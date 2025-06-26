# tests/test_agent_responses.py

import pytest
from tests.test_cases import test_cases
from tests.agent_runner import run_agent_with_prompt

# Define your agent scripts and roles/usernames
AGENT_SCRIPTS = [
    ("app.py", "nurse1", "1"),
    ("graph_test_sys_rag.py", None, None),
    ("graph_test_sysprompt.py", None, None),
    ("graph_test.py", None, None),
]
ROLES = [
    "Nurse",         # for graph_test scripts
    "Pharmacist",
    "Doctor",
    "Supervisor"
]
# For app.py, we use username/password/role combos (must exist in your users CSV)
USER_CREDENTIALS = {
    "Nurse": ("nurse1", "1"),
    "Pharmacist": ("pharm1", "1"),
    "Doctor": ("doc1", "1"),
    "Supervisor": ("super1", "1"),
}

def is_similar(response, expected):
    """
    Checks if expected substring is in response (case insensitive).
    You can replace this with difflib similarity if you want fuzzy match.
    """
    return expected.lower() in response.lower()

@pytest.mark.parametrize("agent_script,username,password", AGENT_SCRIPTS)
@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("case", test_cases)
def test_agent_response(agent_script, username, password, role, case):
    prompt = case["prompt"]
    expected = case["expected"]

    # Pick right login for app.py, otherwise just pass role
    if agent_script.endswith("app.py"):
        user, pwd = USER_CREDENTIALS[role]
        response = run_agent_with_prompt(agent_script, user, pwd, role, prompt)
    else:
        response = run_agent_with_prompt(agent_script, username, password, role, prompt)

    # (optional) Print for debugging
    print("\n==== Test ====")
    print(f"Agent: {agent_script}, Role: {role}")
    print(f"Prompt: {prompt}")
    print(f"Expected to contain: {expected}")
    print(f"Response: {response}\n")
    assert is_similar(response, expected), (
        f"FAILED!\nPrompt: {prompt}\nExpected similar to: {expected}\nGot: {response}"
    )
