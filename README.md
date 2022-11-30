## to preprare the area:

1. place audio class directories  in a directory named "audios-original" in your project area (same path with scripts)

2. place "\_background\_noise\_" directory in the project area.

3. create two directories:
    * audios-training
    * audios-validation

the proyect area should look like this:
```
--- project-directory/
    |--- _background_noise_/
    |--- audios-original/
    |--- audios-training/
    |--- audios-training/
    |--- build_noise_slice.py 
    |--- build_spectrograms.py
    |--- cnn-model-train.py
    |--- fill_sounds.py
    |--- sample_prep.py
    |--- training_sort_script.py
    |--- validation_sorting_script.py
```
then execute `sample_prep.py`

## training

execute `cnn-model-train.py`
