<h1 align="center">AI-Art-Generator</h1>

[![made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18nLCUAQZJ-vuOIn04IrBMubqsV6VO_9j?usp=sharing)[![Made with Tensorflow](https://aleen42.github.io/badges/src/tensorflow.svg)](https://www.tensorflow.org/)

<h2 align="center">Overview</h2>

## Simple Art style transferer
You give a style image and the model learns the features and transfers it to the content image
Best recommended to run with a GPU for the fastest result

## How it works?

Neural style transfer is an optimization technique that involves creating a new image that merges the content of one image with the style of another. Here's a high-level overview of how the AI-Art-Generator works:

    - Load the content image and the style image.
    - Preprocess the images by resizing and normalizing them.
    - Create a model that combines a pre-trained convolutional neural network (CNN) with the VGG19 architecture.
    - Define loss functions that measure the content loss and style loss between the generated image and the target image.
    - Set up the optimization process using gradient descent to minimize the total loss.
    - Iterate the optimization process to update the generated image and minimize the loss.
    - Generate the final stylized image.

By minimizing the content loss, the generated image retains the content of the original content image. By minimizing the style loss, the generated image captures the artistic style of the style image. The balance between the content and style losses can be adjusted to control the final result.

## Further reading

[Gatys’ paper](https://arxiv.org/abs/1508.06576)\
[Gradient descent](https://developers.google.com/machine-learning/crash-course/reducing-loss/gradient-descent)

## Credits
[Tensorflow article](https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398)
(It was for version TensorFlow V1)


## Some examples of the project

#### Content and style images

![image](https://cdn.discordapp.com/attachments/748848099891347498/794168270831353856/tRe7lwtniHiKzxOK0pl2g5HA6HwFwXCRc6dDhcDgcDofjIuESLYfD4XA4HI6LhEu0HA6HwFwOC4SLtFyOBwOh8PhuEi4RMvhcDgc.png)

### Output

![image](https://cdn.discordapp.com/attachments/748848099891347498/794168176110731264/uNsabtFDjw5F7SPtB5ZrBdeNPfbuXaH96JOWTIkCF3KLtdQhkyZMiQIffDA18yJAhQ5QhgYZMiQIXcoQwMfMmTIkDuUoYEPGTJky.png)

## How to run locally?

#### Clone the repo.
``` sh
git clone https://github.com/Rohith04MVK/AI-Art-Generator
```

#### Setup conda.
```sh
conda create -n AI-Art-Generator python=3.8.5
```
```sh
conda activate AI-Art-Generator
```


#### Install dependencies.

```sh
pip install -r requirement.txt
```
#### OR
```sh
conda install --file requirement.txt
```

#### Replace the pictures.
Replace line 10  `content_path` with the image you want to transform
Replace line 11 `style_path` with the style picture you want to transfer

#### Run the file!
```sh
python aiart.py
```
