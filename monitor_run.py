import subprocess
import time
import re
import sys
import threading
import os

def check_credits():
    try:
        output = subprocess.check_output(["modal", "billing", "summary"], text=True, stderr=subprocess.STDOUT)
        match = re.search(r'Credits:\s*-([\d\.]+)', output)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"Error checking billing: {e}")
    return None

def monitor(proc):
    initial_credits = check_credits()
    if initial_credits is None:
        initial_credits = 24.0
    
    threshold = initial_credits + 5.0
    print(f"Initial used credits: ${initial_credits:.2f}. Will stop if it exceeds ${threshold:.2f}")

    while proc.poll() is None:
        time.sleep(30)
        used = check_credits()
        if used is not None:
            remaining = 30.0 - used
            print(f"[Monitor] Used: ${used:.2f}, Remaining: ${remaining:.2f}")
            if remaining < 1.0 or used > threshold:
                print(f"\n[CRITICAL] STOPPING RUN! Credits dropped below $1 (Used: ${used:.2f})")
                proc.terminate()
                time.sleep(3)
                if proc.poll() is None:
                    proc.kill()
                subprocess.run(["modal", "app", "stop", "aic-frame-extracting"], check=False)
                break

if __name__ == "__main__":
    cmd = [sys.executable, "scripts/frame_extracting/resume_production.py", "--run-id", "l21-l25-prod-v1"]
    proc = subprocess.Popen(cmd)
    t = threading.Thread(target=monitor, args=(proc,))
    t.start()
    proc.wait()
