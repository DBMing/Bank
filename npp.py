import time
from datetime import datetime

def keep_alive(file_path="/tmp/keep_alive.txt", interval_minutes=30):
    while True:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(file_path, "a") as f:
            f.write(f"Keep-alive ping at {now}\n")
        print(f"[{now}] Wrote keep-alive ping to {file_path}")
        time.sleep(interval_minutes * 60)

# 启动
keep_alive()
