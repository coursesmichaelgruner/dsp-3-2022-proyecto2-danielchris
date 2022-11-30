import subprocess


subprocess.call(["python","background_noise_slice.py"])
subprocess.call(["python","validation_sort_script.py"])
subprocess.call(["python","training_sort_script.py"])
subprocess.call(["python","fill_sounds.py"])
subprocess.call(["python","build_spectrograms.py"])
