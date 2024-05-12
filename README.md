# Product-Review-Rating-Prediction

## Task
Using BERT to predict review rating of product
| [kaggle link](https://www.kaggle.com/competitions/2024-data-mining-hw2/)

## Dataset 

#### Columns
- rating: Consumer ratings of this product (the label that need to predict)
  - The actual ratings in this dataset are whole numbers without decimal points, and ratings should be outputted directly as integers ranging from 1 to 5
- title: title of the review
- text: content of the review
- verified_purchase: A review is verified_purchase if
  - Bought the item on this site
  - Paid a price available to most shoppers.
- helpful_vote: the number of people that found this review helpful

#### Link
- [Training Data]([https://drive.google.com/file/d/1Ly8FfrUSgOTNA3xhsbaJaLS3KYsc3OpJ/view?usp=sharing](https://drive.google.com/file/d/1AgXJOjuVwHOJiMxuHpVC_47JKUMXu3VT/view?usp=sharing))

- [Testing Data]([https://drive.google.com/file/d/1nVts8Hcx4iFplVeNRYyEdsUaGfMxGNdE/view?usp=sharing](https://drive.google.com/file/d/1AhayduaMDuI6S_eztNqdLupo8KD-m8b9/view?usp=sharing))

## Implementation

#### Feature Selection

Initially, I tried removing `verified_purchase=False` from the training data, but the results were not better than when I did not remove it. Therefore, in this assignment, I did not use `helpful_vote` and `verified_purchase`. Instead, I used `title` and `text` as inputs and `rating` as labels. Since there are five categories (`num_labels=5`), I used a list to represent the labels. For example, `rating=1` is represented as `[0, 1, 0, 0, 0]`.

#### Preprocessing

Perform a series of preprocessing steps using utilities from the gensim library.
- convert text to lowercase and a consistent unicode format
- remove leading or trailing whitespaces
- remove any HTML or XML tags
- remove all punctuation marks
- replace occurrences of multiple white spaces with a single space
- remove numbers
- remove stopwords
- remove short words which include less meaningful words in the data

#### Tokenize

First, I combine title and text. Next, this string is processed using a BERT tokenizer. The tokenizer is configured to ensure uniform input length through padding (padding=True) and to limit the sequence length to a maximum of 128 tokens (max_length=128).
