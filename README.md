<div align="center">
  <img src="https://ai.stanford.edu/~jkrause/cars/class_montage.jpg">
</div>

# Stanford Training

This is submission for the Computer Vision challenge.

## Evaluation step-by-step guide

The evaluation script can take in image or folder of images (recursively). Please look at the folder *`./data/samples`* in this repo to see an example input image folder. After evaluation, the script will generate a csv file, with the leftmost column is the image filename, and all other columns contain the confidence score for corresponding class.

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

The above step-by-step operation will output 2 files:
- *`./data/confidence.csv`*: this file contains 197 columns, the first column is filename, the other 196 columns are confidence score for corresponding class. The class label can be obtained at *`./data/meta.json`*.
- *`./data/result.csv`*: this file contains 3 columns, the first column is filename, the second column is the class index, and the third column is class label (looked up from `./data/meta.json`.

## Model detail

The model employs some of the common deep learning techniques in computer vision:

- DenseNet: https://arxiv.org/abs/1608.06993
- Weakly-Supervised Data Augmentation Network: https://arxiv.org/abs/1901.09891

Additionally, some techniques are also employed to improve training and prediction:

- Super-Convergence: https://arxiv.org/abs/1708.07120. Inspired by cyclical learning rate schedule, this technique cuts training time by nearly a half, while also increasing final accuracy by making model converge to a better minima.
- Temperature scaling: https://arxiv.org/abs/1706.04599. Deep learning models are usually over-confident. Temperature scaling calibrates the confidence score of model prediction, so that it is less likely for the incorrect prediction to have high confidence score.
- Add more output logits than the number of classes in dataset. This model has 500 units in the output layer, even though the Stanford dataset only has 196 classes. Through experiments on this dataset, I consistently observed higher accuracy for the larger model. Moreover, by adding more units into the output layer, I hypothesize that it would be easier to fine-tune model whenever we want to incorporate new car classes (which should happen very regularly)

Through combination of the above techniques, we attain a classification model that has higher accuracy performance, is faster to train, and easier to finetune.

These techniques are chosen for increasing accuracy while reducing training time and providing a good confidence score for the model.

