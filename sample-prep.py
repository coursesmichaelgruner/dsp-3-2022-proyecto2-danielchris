import subprocess


subprocess.call(["python3","validation_sort_script.py"])
subprocess.call(["python3","training_sort_script.py"])
subprocess.call(["python3","test_sort_script.py"])
subprocess.call(["python3","background_noise_slice.py"])
subprocess.call(["python3","fill_sounds.py"])
subprocess.call(["python3","build_spectrograms.py"])
