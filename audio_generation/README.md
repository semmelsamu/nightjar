# Audio Generation

This directory contains the Audio creation part for the Nightjar project.

The Jupyter notebook ``nightjar_music_generation_rave.ipynb`` contains the steps to generate audio with the pretrained RAVE-model.

Information on how we trained the RAVE-model can be found in the ``rave/`` folder.

## Requirements

To use the notebook, you need two external files/folders (ask @RosieG for access):

* a trained RAVE model of type ``*.ts``. Copy the model into ``audio_generation/models``.
* `training_data` — the (training) data used visualization and for audio generation. This is the generated dataset from the ``rave/`` directory

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
Alternatively, apply "Clear All Outputs" in the Notebook.

