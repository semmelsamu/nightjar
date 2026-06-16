# 🐦 Nightjar – Audio Generation

This part of the Nightjar project is focused on generating audio inspired by bird sounds and nature-influenced music using a RAVE model. 

The following explains how to train the RAVE model on a pre-defined dataset and how to apply
the trained model to generate sounds.

---

## 1. Data Preparation

All training data is stored in the `data/` directory.

Because the dataset is too large to store on GitHub, please ask @RosieG for access. Once downloaded, place the dataset inside the `rave/` folder as shown below:

```text
rave/
└── data/
    └── unprepared_data/
    └── prepared_data/
```

The ``data/`` folder organized into two main subfolders:

### 🔹 `unprepared_data/`

Contains raw, unprocessed datasets:

* `kaggle_birds_data.zip`
  Bird sound recordings from Kaggle
  https://www.kaggle.com/datasets/soumendraprasad/sound-of-114-species-of-birds-till-2022

* `song_data.zip`
  Songs by Cosmo Sheldrake, which incorporate and are inspired by natural and bird sounds.
  The objective of this project is to generate similar music.

---

### 🔹 `training_data/`

Is generated from the unprepared data and will be used for training the model. To generate, run the following scripts:


```bash
python scripts/step1_prepare_birds.py
python scripts/step2_prepare_songs.py
python scripts/step3_unify_data.py
```

These three preprocessing steps do the following:

#### a) Bird Dataset Processing

* Randomly selected **3 recordings per species**
* Trimmed to a maximum of **20 seconds**
* Converted to **mono `.wav` format**

#### b) Song Dataset Processing

* Sliced into **20-second non-overlapping segments**
* To increase the dataset, the songs were augmented using **offset slicing**:

  * 0s offset
  * 5s offset
  * 10s offset

  This yields ~3x as many song snippets.
* Discarded segments shorter than **5 seconds**
* Converted to **mono `.wav` format**

Intermediate outputs from step a) and b) are stored in:

```
data/unprepared_data/preprocessing_steps12/
```


#### c) Dataset Unification

* Determined the most common **sampling rate**
* Resampled all files to match this rate
* Ensured consistency:

  * Mono channel
  * `.wav` format
  * Normalized audio levels



#### Expected Output

```
=== VERIFICATION ===
Total files        : 504
Sample rates found : {44100: 504}
Channel counts     : {1: 504}
Total duration     : 144.0 min
  └ birds          : 89.2 min
  └ songs          : 54.8 min

✔ All files are 44100 Hz mono WAV.

Done. Output → data/training_data
```

> ⚠️ **Note:** RAVE requires at least **1 hour of audio** for training. More data is recommended.

---

## 2. Training the RAVE Model

It is recommended to perform training on a computer/server with a GPU of at least 8GB for the full time of training, with SSH access. 

I recommend following the tutorial https://forum.ircam.fr/article/detail/training-rave-models-on-custom-data/ for further information on setting up the environment and training a RAVE model. The main steps for training include:

### Upload Training Data

```bash
scp -r /path/to/data/training_data <user_name>@<host>:~/
```

---

### RAVE Preprocessing

```bash
rave preprocess \
  --input_path ~/training_data \
  --output_path ~/training_data_preprocessed \
  --channels 1
```
> ⚠️ **Notes:**
>
> The number of channels should fit the data. The provided scripts convert the audio to a mono channel, therefore `channels 1` is used.

---

### Start Training

I recommend setting up a screen so training can be performed when the window is closed (see tutorial).


```bash
rave train \
  --name nightjar \
  --db_path ~/training_data_preprocessed/ \
  --out_path ~/model_output/ \
  --config v2 \
  --config noise \
  --channels 1 \
  --gpu 0 \
  --save_every 100000
```

> ⚠️ **Notes:**
>
> * **RAVE v2** requires a **≥16GB GPU**. Other versions are available too.

---

## 3. Exporting the Model

Model checkpoints are saved after `save_every` many steps during training:

```bash
cd ~/model_output/nightjar_model_*/version_*/checkpoints
```

### Export a Checkpoint

```bash
rave export --run . --name epoch_1000000.ckpt
```

### Download Locally

```bash
scp <user_name>@<host>:~/model_output/.../epoch_100000.ckpt.ts ~/Downloads/
```

---

## 4. Applying the Model

After exporting, load the model into the folder ``audio_generation`` and run ``audio_generation/nightjar_music_generation_rave.ipynb`` as needed.

---

## Additional Resources

* RAVE Training Tutorial:
  https://forum.ircam.fr/article/detail/training-rave-models-on-custom-data/

---