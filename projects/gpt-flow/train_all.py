import os
import subprocess

folder_path = "./dataset/quanzhou/od_flow"

for i in range(100):
    print('第', i + 1, '轮')
    for file_name in os.listdir(folder_path):
        if "weekend" in file_name or "test" in file_name:
            continue 
        else:
            command = ["python", "flowgpt.py", "--trainer.max_iters=101", "--trainer.file_name=" + file_name[:-4]]
            print(" ".join(command))
            subprocess.run(command, shell=True)