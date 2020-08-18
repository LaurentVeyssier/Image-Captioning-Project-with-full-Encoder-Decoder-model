# Image-Captioning-Project
Generate caption on images using CNN Encoder- LSTM Decoder structure

This project is the second project of Udacity's Computer Vision Nanodegree.

The notebook uses Pytorch and the MS COCO dataset for training. The dataset contains 8000 images with 5 pre-made captions each. Additional txt files are provided to seperate the dataset into training (6000 images), validation (1000) and test (1000) sets.

I developped the notebook on Colab using GPU for training.

The notebook uses the pre-trained VGG16 network to extract features from the images. The extracted features are all saved on disk for later use during training. This speeds up the training and reduces computation needs.
