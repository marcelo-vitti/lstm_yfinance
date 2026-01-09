import psutil
import time
import logging

logging.basicConfig(level=logging.INFO)


def log_resource_usage():
    process = psutil.Process()
    cpu = process.cpu_percent(interval=0.1)
    mem = process.memory_info().rss / 1024**2

    logging.info(f"CPU: {cpu:.2f}% | Memory: {mem:.2f} MB")
