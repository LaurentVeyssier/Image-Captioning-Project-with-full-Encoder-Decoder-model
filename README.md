# Image-Captioning-Project
Generate caption on images using CNN Encoder- LSTM Decoder structure

This project is the second project of Udacity's Computer Vision Nanodegree.

The notebook uses Pytorch and the MS COCO dataset for training. COCO is a large image dataset designed for object detection, segmentation, person keypoints detection, stuff segmentation, and caption generation. The dataset contains images with 5 pre-made captions each. The large majority of these captions are between 8 to 15 words long.

Please note that GPU is required to enable COCO API.

I developped the notebook on Colab using GPU for training.

The notebook uses the pre-trained VGG16 network to extract features from the images. The extracted features are all saved on disk for later use during training. This speeds up the training and reduces computation needs.
