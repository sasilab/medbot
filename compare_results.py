import subprocess
import time

# Use poetry run for poetry-managed venv
AGENTS = [
    # For app.py, provide user & pass (adjust for your test users)
    ("app.py", ["poetry", "run", "python", "app.py", "nurse1", "1"]),
    ("graph_test.py", ["poetry", "run", "python", "graph_test.py"]),
    ("graph_test_sysprompt.py", ["poetry", "run", "python", "graph_test_sysprompt.py"]),
    ("graph_test_sys_rag.py", ["poetry", "run", "python", "graph_test_sys_rag.py"]),
]

#ROLES = ["Nurse", "Pharmacist", "Doctor", "Supervisor"]
ROLES = ["Nurse"]
TEST_QUERIES_FILE = "test_queries.txt"

def run_agent(agent_cmd, role, queries):
    print(f"\n\n==== Running {' '.join(agent_cmd)} as {role} ====")
    proc = subprocess.Popen(
        agent_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    # For graph_test*.py, first thing is role. For app.py, not needed (handled by argv)
    if "app.py" not in agent_cmd[3]:  # poetry run python app.py ...
        proc.stdin.write(role + "\n")
        proc.stdin.flush()
    for query in queries:
        proc.stdin.write(query + "\n")
        proc.stdin.flush()
        time.sleep(0.2)  # Optional: tiny delay for UI/LLM stability
    proc.stdin.write("exit\n")
    proc.stdin.flush()
    # Gather output
    output = []
    try:
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            output.append(line.rstrip())
    except Exception as e:
        print("Error:", e)
    proc.wait(timeout=30)
    return "\n".join(output)

def main():
    # Load queries from file
    with open(TEST_QUERIES_FILE, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]
    for role in ROLES:
        print(f"\n========== RESULTS FOR ROLE: {role} ==========\n")
        for agent_file, agent_cmd in AGENTS:
            print(f"\n=== {agent_file} ({role}) ===\n")
            output = run_agent(agent_cmd, role, queries)
            print(output)
            print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
