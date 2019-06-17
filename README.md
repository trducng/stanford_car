# Stanford training.

This is submission for the Computer Vision challenge.

## Evaluation step-by-step guide

The evaluation script can take in image or folder of images (recursively). Please look at the folder `data` in this repo to see an example input image folder. After evaluation, the script will generate a csv file, with the leftmost column is the image filename, and all other columns contain the confidence score for corresponding class.

The pretrained model is located here: https://drive.google.com/open?id=1i33mwp-xk9nfbip9Ca8ao7o69MJ1Zplv

```bash
# clone this repo
git clone https://github.com/johntd54/stanford_car
cd stanford_car

# create a new python 3.7 environment with conda
conda create -n stanford_car_eval python=3.7
conda activate stanford_car_eval

mkdir ckpt
# download the pre-trained model into folder `ckpt`
# url: https://drive.google.com/open?id=1i33mwp-xk9nfbip9Ca8ao7o69MJ1Zplv

# install supporting libraries
pip install -r requirements.txt
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
# or `pip install torch torchvision` if gpu is not available

# take note of the evaluation folder and run the evaluation script
# the result will be stored at result.csv
# remove the `--gpu` parameter on non-gpu machine
python evaluate.py [image_folder] --ckpt [checkpoint_path] --gpu
```

The above step-by-step operation will output *`./result.csv`*. This file contains 197 columns, the first column is filename, the other 196 columns are confidence score for corresponding class. The class label can be ontained at *`./data/meta.json`*.

## Model detail

The model employs some of the common deep learning techniques in computer vision:

- DenseNet: https://arxiv.org/abs/1608.06993
- Weakly-Supervised Data Augmentation Network: https://arxiv.org/abs/1901.09891
- Super-Convergence: https://arxiv.org/abs/1708.07120
- Temperature scaling:

These techniques are chosen for increasing accuracy while reducing training time and providing a good confidence score for the model.

