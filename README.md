# image-scrambler
Undetectable adversarial image generation

This adds noise to an image and makes it undetectable by normal CV algorithms, without actually altering the underlying image to the human eye. There is a loop to test this noise and see at what minimum epsilon threshold it's effective (consistently misclassified by an ImageNet model). The images will be stored in an images/ folder and the output should be put in an output/ folder. You should also display the images side by side to let me see if they're that different.
