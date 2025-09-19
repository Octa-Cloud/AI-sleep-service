import subprocess
import sys
import time
import os


def start():
    procs = []
    try:
        if os.getenv("QUEUE_BACKEND", "local") == "local":
            procs.append(subprocess.Popen([sys.executable, "-m", "services.fastapi.infra.queue.worker_queue"]))
    except Exception:
        pass

    time.sleep(0.5)

    procs.append(subprocess.Popen([sys.executable, "-m", "services.fastapi.worker.brainwave_analysis_worker"]))
    procs.append(subprocess.Popen([sys.executable, "-m", "services.fastapi.worker.audio_analysis_worker"]))
    if os.getenv("QUEUE_BACKEND", "local") == "local":
        procs.append(subprocess.Popen([sys.executable, "-m", "services.fastapi.infra.worker.db_writer"]))

    procs.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "services.fastapi.main:app", "--host", "0.0.0.0", "--port", "8000"]))

    for p in procs:
        p.wait()


if __name__ == "__main__":
    start()


