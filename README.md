# TKGQA
This is the code for the manuscript my thesis "Exploring Vietnamese Question Answering Methods using time constraints".
## Installation

Create a conda environment
``` 
conda create --prefix ./myenv python=3.7
conda activate ./myenv
```

If this is not the case, try replacing ``python`` with ``python3``. If that works, replace ``python`` with ``python3`` in all commands below.

Install TKGQA requirements
```
conda install --file requirements.txt -c conda-forge
```

## Dataset and pretrained models download

Download and unzip ``data_v2.zip`` and ``models_ctrn.zip`` in the root directory.

Data: https://drive.usercontent.google.com/download?id=1xHjpkDYlp1JGqJWnefrzdaVGV2_SWksI&authuser=0
Models: https://drive.google.com/uc?id=14u6snck2DwS5dIh9iG6fekUzS4Wzf8Bc
## Running the code

TempoQR:
```
python ./train_qa_model.py --model tempoqr --supervision soft
python ./train_qa_model.py --model tempoqr --supervision hard
 ```

 CTRN: 
 ```
python ./train_qa_model.py --model ctrn --supervision soft --mode train --max_epochs 20
python ./train_qa_model.py --model ctrn --supervision hard --mode train --max_epochs 20
```
with MultilingualBERT: 
``` 
python ./train_qa_model.py --model tempoqr --supervision soft --lm_model bert-base-multilingual-cased
python ./train_qa_model.py --model ctrn --supervision soft --mode train --max_epochs 20 --lm_model bert-base-multilingual-cased
```
Other models: "entityqr" and "cronkgqa" with hard and soft supervisions.
 
To use a corrupted TKG change to "--tkg_file train_corXX.txt" and "--tkbc_model_file tcomplex_corXX.ckpt", where XX=20,33,50.

To evaluate on unseen complex questions change to "--test test_bef_and_aft" or "--test test_fir_las_bef_aft".


