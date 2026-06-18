# Nightjar Audio

This repository contains the Audio creation for the Nightjar project.

The Jupyter notebook ``nightjar_music_generation_rave.ipynb`` contains the steps to generate audio with the pretrained RAVE-model.

Information on how we trained the RAVE-model can be found in the ``rave/`` folder.

## Requirements

To use this repository, you need two external files/folders (ask @RosieG for access):

* `best.ckpt.ts` — the trained model checkpoint
* `data/` — the dataset folder used for audio generation. It contains three subfolders: 
    - ``output`` (empty folder to save the output to)
    - ``training_data`` (processed training data; used for visualizations and generation of new data)

Copy both into the folder ``audio_generation``.

These files are not included in GitHub because they are too large.

## Setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Jupyter notebooks

Please do **not** upload Jupyter notebooks with plots or other outputs, as they make the files too large.

Before committing notebooks, enable `nbstripout`:

```bash
nbstripout --install
```

After this, notebook outputs such as plots, images, and large embedded data will be removed automatically when committing.

