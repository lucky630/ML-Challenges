The Approach which i have taken for this challenge is as follow:

- Used the learned weights from places365 dataset.used Resnet50,resnet18,Alexnet pretrained weights.densenet weights didn’t used because it’s not compatible with pytorch 1.0. Link to the pretrained weights are below:
https://github.com/CSAILVision/places365/

- Used imagenet pretrained weights avaiable in Cadene.here model used are Resnet152,densenet169,Resnext.link to the pretrained weights are below:
https://github.com/Cadene/pretrained-models.pytorch

- These 6-7 different model have been trained for different sizes that is 128,150(Default),300 and also with different Augmentation.link to the transformations are below:
https://docs.fast.ai/vision.transform.html

- Weighted Average of probability of each class have given me the final solution.

- Last but not the least have Created oof for all the models and do ensembling by training decision classifier model.

I have started to work on this problem from last 4 days only.link to the challenge problem is below:
https://datahack.analyticsvidhya.com/contest/practice-problem-intel-scene-classification-challe/
