# Accenture Digital Hack Up
First Position solution for Accenture Digital Hack Up machine Learning challenge
https://www.hackerearth.com/challenges/competitive/Accenture-ml/leaderboard/

## About 
In this competition we were chellenged to build a model that can predict scores of comments based upon The parent comment to which sarcastic comments are made and the Reply to a parent comment.

Train Having 45000 comment replies and the test set having 30000 rows of comments.Here Task was to build a model that can predict scores of comments present in the test dataset.


![alt text](https://github.com/lucky630/ML-Challenges/blob/master/Accenture%20Digital%20Hack%20Up/data/CommentScorerDiagram.png)

1. Used tfidf vectorization for both word and character level to convert the comments into vector form.
2. Generate new features like comment length,sentiment value of comment & profanity value of comment.
3. Concatenate both and apply the lightgbmRegressor model for 6 folds to get the final prediction
4. lighgbmRegressor was tuned using the bayesian optimization.

## Team Member
- [utsav aggarwal](https://github.com/utsav1)
- [Arjun Rana](https://github.com/monsterspy)
