# slowfast-adversarial-ml

This repo contains all work relating to current research on Facebook's SlowFast models, specifically the X3D models and evaluating them on adversarial strength. We use the [Kinetics-400](https://deepmind.com/research/open-source/kinetics) dataset to evaluate an X3D model against the benchmarks published in the [original paper](https://arxiv.org/pdf/2004.04730.pdf), and then we aim to attack this model using tools from the [ART toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/examples/get_started_pytorch.py) to assess its adversarial strength

# Table of Contents
- [Setup](#Setup)
- [Execution](#Execution)

## Setup
Note that all work has been done in a Zeblok AI-Workstation outfitted with 2 GPUs, 4 vCPUS, and 100 GB of memory, results may vary depending on your setup


Before doing anything (especially if you've recently spun up a clean instance of a JupyterLab workstation or equivalent), run the following
```bash
sudo apt-get update -y 
sudo apt-get install vim xclip ffmpeg libsm6 libxext6 software-properties-common -y
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt-get upgrade libstdc++6 -y
alias pbcopy='xclip -selection clipboard'
alias pbpaste='xclip -selection clipboard -o'
python3 -m pip install virtualenv
```
Then, clone this repo:
```bash
git clone git@github.com:asarj/slowfast-adversarial-ml.git
cd slowfast-adversarial-ml
```


### Obtaining the Kinetics-400 dataset
1. Open a terminal and make a directory to store the kinetics dataset
```bash
mkdir kinetics-400-dataset-files
cd kinetics-400-dataset-files
```
2. Download the dataset from. [here](https://deepmind.com/research/open-source/kinetics) or use the following command to download it
```bash
wget "https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz"
```
3. Extract the files
```bash
tar -xvf "kinetics400.tar.gz"
```
4. Since we are only interested in evaluating models, we will download from the `validate.csv` file. We use the validation set instead of the test set because the original [SlowFast X3D paper](https://arxiv.org/pdf/2004.04730.pdf) evaluates on the validation set as well. Note that we will not download all files, as it is 450 GB total (and Zeblok only gives us 100 GB to work with). To do this, follow the corresponding commands
```bash
cd ../ActivityNet/Crawler/Kinetics
virtualenv venv
source venv/bin/activate
python3 -m pip install joblib mkl menpo numpy pandas pytz readline setuptools six tk wheel decorator olefile youtube-dl
```

5. We then wish to use the validation set for our tests, so we will download them accordingly
```bash
mkdir ../../../kinetics-400-dataset-files/val/
python3 download.py ../../../kinetics-400-dataset-files/kinetics400/validate.csv ../../../kinetics-400-dataset-files/val/
```

We can use the following command in a new terminal to monitor how many videos are downloaded to the `val` directory. We only wish to download 1500-2500 videos (maybe more depending on your disk space), so we would stop the `download.py` script (ctrl + c) when the number of files in the `val` directory reaches this point
```bash
cd slowfast-adversarial-ml 
watch "find ./kinetics-400-dataset-files/val/ -type f | wc -l"
```


In our evaluation, we downloaded 2098 videos from the validation set.

6. Once the desired video count is reached, we need to preprocess the videos downloaded into a CSV that SlowFast will use to read from during evaluation. We use another module from Facebook Research, called [`video-nonlocal-net`](https://github.com/facebookresearch/video-nonlocal-net/blob/master/DATASET.md), to achieve this

7. First, we need to clean up the filepaths to the videos to elinimate whitespace, which can be done by running the following:
```bash
cd ../../../
python3 video-nonlocal-net/process_data/kinetics/gen_py_list.py 
mkdir kinetics-400-dataset-files/val_256/
python3 video-nonlocal-net/process_data/kinetics/downscale_video_joblib.py 
```


8. As a result of step 7, we generate a file in `./kinetics-400-dataset-files` called `vallist.txt`, which contains the filepaths to all the videos and the numeric id of each action class it belongs to in a form that is compliant. However, we need to convert this to CSV. I have already made a script that does this, and also shuffles the rows in the dataframe to help in the evaluation process, all you will need is the path to `vallist.txt` and run the following script
```bash
cd ../../
python3 scripts/build_slowfast_csv.py \
    -path "./kinetics-400-dataset-files/vallist.txt"
```

9. You will then see two files in the `./kinetics-400-dataset-files/` directory, `test_raw.csv`, which contains the paths and action classes for the raw `val` directory, and `test_256.csv`, which contains the paths and action classes for the preprocessed `val_256` directory. You can rename either or as `test.csv` to use with SlowFast, but it's preferred to rename `test_256.csv` because the videos are resized to the short edge size of 256, which helps in evaluating faster. We rename the files because SlowFast only looks for files that are either `train.csv` or `test.csv`

### Setting Up Facebook SlowFast
To install SlowFast, follow the instructions (most of which are taken from [here](https://github.com/facebookresearch/SlowFast/blob/master/INSTALL.md))
1. Install dependencies
```bash
git clone https://github.com/facebookresearch/slowfast
cd slowfast/
conda create -n slowfast_venv python=3.7 -y
source activate slowfast_venv
pip install light-the-torch
ltt install torch torchvision
pip3 install pytorchvideo
conda install -c conda-forge -c fvcore -c iopath fvcore=0.1.4 iopath -y
conda install -c conda-forge av psutil -y
pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install cython psutil sklearn simplejson opencv-python pillow
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo
``` 

If `detectron2_repo` was already installed and has a file called `detectron2.egg-info` inside it, delete that file and follow the last step again

2. Build SlowFast
```bash
python3 setup.py build develop
```

## Execution
Our research involves working primarily with SlowFast's X3D models. The X3D config files can be found [here](https://github.com/facebookresearch/SlowFast/tree/master/configs/Kinetics) and the X3D model defintion can be found [here](https://github.com/facebookresearch/SlowFast/blob/e2894034797b9d77625e36f39150380d4d26c878/slowfast/models/video_model_builder.py#L617)

To evaluate this code on the val set, pick an X3D model to use and run the following script in the terminal from the root of the `slowfast-adversarial-ml/slowfast` directory
```bash
python3 tools/run_net.py \ 
        --cfg configs/Kinetics/X3D_{model_size}.yaml \ 
        DATA.PATH_TO_DATA_DIR "../kinetics-400-dataset-files/" \ 
        DATA.PATH_LABEL_SEPARATOR " " \ 
        DATA.DECODING_BACKEND "pyav" \ 
        TRAIN.ENABLE False \ 
        DATA_LOADER.NUM_WORKERS 0 \ 
        NUM_GPUS 1 \ 
        TEST.BATCH_SIZE 16 \ 
        |& tee "../output_logs/[filename].txt"
```
We specify `DATA_LOADER.NUM_WORKERS` to be `0` and `NUM_GPUS` to be `1` despite having more resources because multithreading errors were thrown during model evaluation (which are not currently known how to solve).

In our case, we used the `X3D_M` model definition, so our script was
```bash
python3 tools/run_net.py \
        --cfg configs/Kinetics/X3D_M.yaml \
        DATA.PATH_TO_DATA_DIR "../kinetics-400-dataset-files/" \
        DATA.PATH_LABEL_SEPARATOR " " \
        DATA.DECODING_BACKEND "pyav" \
        TRAIN.ENABLE False \
        DATA_LOADER.NUM_WORKERS 0 \
        NUM_GPUS 1 \
        TEST.BATCH_SIZE 32 \
        TENSORBOARD.ENABLE True \
        |& tee "../output_logs/test_results_X3DM_trial2.txt"
```

Depending on your hardware configuration, SlowFast might crash during model evaluation, for one of several reasons
1. If you get an error that looks something like this 

```bash
RuntimeError: Failed to fetch video after 10 retries.
```
This means that the video was corrupted at a specific frame somehow. The SlowFast docs and the open issues on GitHub don't provide any remedies for this, so it is recommended to delete the file in question from the csv(s) and re-run the script

2. In the model evaluation, the `top1_acc` field is always 0 for each epoch

This means that there are no pretrained models for SlowFast to load from. You will need to download each of them from [here](https://github.com/asarj/ActionRecognitionAdversarialML/blob/master/slowfast/MODEL_ZOO.md) via `wget` and modify the config files for each X3D model in the `TEST.CHECKPOINT_PATH` attribute to point the url to this.