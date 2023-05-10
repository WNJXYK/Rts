import sys
sys.path.append("./")

import subprocess
import src.Utils as Utils
import src.Config as Config
import logging
import time

noise_rate_table = {
    "sym":  [0.0, 0.2, 0.4, 0.6, 0.8],
    "asym": [0.1, 0.2, 0.3, 0.4]
}
logging.basicConfig(filename='./logs.txt', level=logging.INFO)

dataset_time = time.time()
for noise_type in ["sym", "asym"]:
    for noise_rate in noise_rate_table[noise_type]:
        for seed in [0, 1, 2, 3, 4]:
            start_time = time.time()
            command = "python run.py"
            command += " --noise-type " + noise_type
            command += " --noise-rate " + str(noise_rate)
            command += " --method Proposal"
            command += " --dataset ECG5000"
            command += " --seed " + str(seed)
            command += " --gpu 0 "
            print("> ", command)

            subprocess.call(command, shell=True)
            logging.info(command + " - Time: " + str(time.time() - start_time) + "Secs")
        logging.info(noise_type + str(noise_rate) + " - Total Time: " + str((time.time() - dataset_time) / 60.0) + "Mins")
    
    