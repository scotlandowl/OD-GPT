import os
import subprocess

for i in range(93, 28, -4):
    # command = ["python", "generate_flow_qz.py", "--his_length=" + str(i)]
    
    # test finetuning
    command = ["python", "generate_flow_qz_1.py", "--his_length=" + str(i)]
    print(" ".join(command))
    subprocess.run(command, shell=True)