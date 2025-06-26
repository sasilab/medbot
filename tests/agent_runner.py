# agent_runner.py

import subprocess
import time

def run_agent_with_prompt(agent_script, username, password, role, prompt):
    """
    Run an agent script and get its output.
    If username/password are provided, pass as argv (for app.py).
    Otherwise, role is sent as stdin (for graph_test*.py).
    """
    if agent_script.endswith("app.py"):
        cmd = ["poetry", "run", "python", agent_script, username, password]
        first_input = None  # No need to pass role as stdin
    else:
        cmd = ["poetry", "run", "python", agent_script]
        first_input = role  # Send role as first line in stdin

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # Feed login/role if needed
    if first_input is not None:
        proc.stdin.write(first_input + "\n")
        proc.stdin.flush()
        time.sleep(0.1)

    # Now send prompt and exit
    proc.stdin.write(prompt + "\n")
    proc.stdin.flush()
    time.sleep(0.1)
    proc.stdin.write("exit\n")
    proc.stdin.flush()

    output_lines = []
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        output_lines.append(line.strip())

    proc.wait(timeout=30)
    # Filter agent's main response (customize this for your output structure)
    return "\n".join(output_lines)
