import subprocess
import time
import re

def check_credits():
    try:
        output = subprocess.check_output(["modal", "billing", "summary"], text=True, stderr=subprocess.STDOUT)
        match = re.search(r'Credits:\s*-([\d\.]+)', output)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"Error checking billing: {e}")
    return None

initial = check_credits()
if initial is None: initial = 24.61
limit = 29.61
print(f"Monitoring billing. Initial: ${initial}, Limit: ${limit}")

while True:
    time.sleep(60)
    used = check_credits()
    if used is not None:
        print(f"Used: ${used}")
        if used > limit:
            print("Stopping app!")
            subprocess.run(["modal", "app", "stop", "aic-frame-extracting", "-y"])
            break
