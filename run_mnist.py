import os
import subprocess
import sys

# 先设置环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 然后运行main.py
subprocess.run([sys.executable, "main.py"])