# AI-Art-Generator

[![made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18nLCUAQZJ-vuOIn04IrBMubqsV6VO_9j?usp=sharing)[![Made with Tensorflow](https://aleen42.github.io/badges/src/tensorflow.svg)](https://www.tensorflow.org/)

## Overview

### Simple Art style transferer

#### You give a style image and the model learns the features and transfers it to the content image

#### Best recommended to run with a GPU for fastest result

### How it works?

#### Neural style transfer is an optimization technique used to take two images, style reference image (such as an artwork by a famous painter), and the input image you want to style

#### The model then blends the images to give your input image a artistic style

### Further reading

#### [Gatys’ paper](https://arxiv.org/abs/1508.06576)

#### [Gradient descent](https://developers.google.com/machine-learning/crash-course/reducing-loss/gradient-descent)

### Credits
#### [Tensorflow article](https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398)
#### (It was for version Tensorflow V1)


### Some examples of the project

#### Content and style images

![image](https://cdn.discordapp.com/attachments/748848099891347498/794168270831353856/tRe7lwtniHiKzxOK0pl2g5HA6HwFwXCRc6dDhcDgcDofjIuESLYfD4XA4HI6LhEu0HA6HwFwOC4SLtFyOBwOh8PhuEi4RMvhcDgc.png)

### Output

![image](https://cdn.discordapp.com/attachments/748848099891347498/794168176110731264/uNsabtFDjw5F7SPtB5ZrBdeNPfbuXaH96JOWTIkCF3KLtdQhkyZMiQIffDA18yJAhQ5QhgYZMiQIXcoQwMfMmTIkDuUoYEPGTJky.png)

## How to run locally?

### Clone the repo

#### ``` git clone https://github.com/Rohith04MVK/AI-Art-Generator```

### Setup conda
#### ```conda create -n AI-Art-Generator python=3.8.5```
#### ```conda activate AI-Art-Generator```


### Install dependancies

#### ```pip install -r requirements.txt```
#### OR
#### ```conda install --file requirements.txt```

### Replace the pictures
#### Replace line 10  ```content_path``` with the image you want to transform
#### Replace line 11 ```style_path``` with the style picture you want to transfer

### Run the file!
#### ```python aiart.py```
