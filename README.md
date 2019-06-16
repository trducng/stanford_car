# Stanford training.

This is submission for the Computer Vision challenge.

## Evaluation step-by-step guide

The evaluation script can take in image or folder of images (recursively). Please look at the folder `data` in this repo to see an example input image folder. After evaluation, the script will generate a csv file, with the leftmost column is the image filename, and all other columns contain the confidence score for corresponding class.

```bash
# clone this repo
git clone ...

# download the pre-trained model into folder `ckpt`
mkdir ckpt
wget -0 .. ckpt

# install supporting libraries
pip install -r requirements.txt

# take note of the evaluation folder and run the evaluation script
# the result will be stored at result.csv
python evaluate.py --input_folder [input_folder] --ckpt [checkpoint_path] --gpu
```

## Model detail

The basic model employs some of the common deep learning techniques in computer vision:

- Residual Net 152: https://arxiv.org/abs/1512.03385
- Weakly-Supervised Data Augmentation Network: https://arxiv.org/abs/1901.09891

The model and training procedure makes use of several new techniques and findings, compared to the previously reported model. These improvements help attain higher accuracy result with lower amount of compute and training time.

- Super-Convergence: https://arxiv.org/abs/1708.07120
- Lottery Ticket: 
- Pointwise optimization:
- Use more logits for output classes than the number of classes in the Stanford Car dataset (1000 vs 196)

