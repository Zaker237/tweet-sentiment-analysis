# Tweet Sentiments Analysis

This is a small project i made with the aim to build a Model for Sentiment analysis for Tweets. The model must receive a tweet as input and be able to predict whether the tweet has a positive, negative or neutral impression.

The model is builded using (python lightning)[https://www.pytorchlightning.ai/] which is an ultimate PyTorch research framework that scale models without the boilerplate.

For the training of our model, we use the (following)[https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis?resource=download] dataset foung on (Kaggle)[https://www.kaggle.com/].

Beside our model, we will also build a small web server to deploy our application through REST API. We used (Flask)[https://flask.palletsprojects.com/en/2.2.x/] to build our server and deploy it as a docker container.


### Todo

* write more details about the model
* explain the step for data clean
* provide more hyperparameters for reproductibility.
