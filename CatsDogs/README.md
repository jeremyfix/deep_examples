Log in to Kaggle : https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition and get the data train.zip, test.zip sample_submission.csv
Put and unzip everything in raw_data/
You should get :
- raw_data/sample_submission.csv
- raw_data/test
- raw_data/train

The provided datasets are :
- train : 25K images of dogs and cats
- test : 12.5K images

The requested output is a probability that the image is a dog (1=dog, 0=cat)


We then split the data in train, valid and keep a small subset into sample/train, sample/valid for prototyping purposes. To do so, just run the "split_data.py" script. Parameters can be found in the beginning of the script

