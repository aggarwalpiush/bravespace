# Bravespace

It is interpretable hate-meme detection systems based curation and annotation tool for designing better hate-meme detection systems. It solve following crucial problems:

- Filteration of potential hatememe using interpretable hate-meme detection system.
- Speedify the curation and further categories annotation generation to fine-tune existing hate-meme detection.
- Helpful in producing hate-meme detection systems for geographical areas with low resources.
- Interpretability features build trust and faithfulness among the moderators for confident curation.

The main motivation to build this tool to overcome lack understanding of what types hate speech being perpetuated on a platform and also content moderators having choice on what type of content they moderate based on their lived experince.

# Current state:

Bravespace frontend will let you upload the meme which will pass to backend server. On backend server, meme is segregated into image and text. We train a hate meme classifier on the hate meme dataset provided by facebook (Kiela, Douwe et al. “The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes.” ArXiv abs/2005.04790 (2020): n. pag.) with accuracy of 0.70 on the provided test set. We first extract feature of image using vgg16 network and fasttext model and apply our trained classifier to get the class of the meme. If meme is classified as hate-speeech, we find the region of image and tokens of the text responsible for the classification. we apply yolo-v3 object detection model to segment the objects available in the image as well as nltk tokenizer to tokenize the text. We calculate the hate score of the image by iteratively masking extracted objects and tokens to the image. The difference of hate score with that of original image classification score provide the influence of the objects and text contributed to hate speech. 

This tool is very easy to reproduce by adopting following steps:

### copy the github repository into your local host environment having GPU support with cuda 10.1 nvidia driver.

`git clone https://github.com/aggarwalpiush/bravespace.git`

- Go to backend folder and follow the backend/readme instructions.
- Go to frontend folder and follow the frontend/readme instructions.
