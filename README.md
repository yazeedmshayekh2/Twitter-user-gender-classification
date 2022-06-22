# Twitter-user-gender-classification

In the beginning, we define the problem that needs to be solved in this project, that is, we have a data set with columns by setting goals that will help us solve it. Therefore, we started to preprocess the available data to get information about these missing values or by finding the percentage of these values and then using one of the two options to drop the columns or fill them with fabricated values and in the other stages we replace the unknown values with nan
Cloud-based machine learning helped me solve the problem with easy access and high flexibility.

1: Methodology 
1.1: Model description:
Classification is defined as the process of recognition, understanding, and grouping of objects and ideas into preset categories a.k.a sub - populations.
With the help of these pre-categorized training datasets, classification in machine learning programs leverage a wide range of algorithms to classify future datasets into respective and relevant categories, In the classification algorithms used for machine learning, we use input training data for the purpose of predicting the probability that the data will fall into one of the pre-defined categories.

First, we scaled the (x) with a standard scaler this will improve the model’s performance, we used (train_test_split) function from sklearn, then we split each of our inputs (Data_9, desc, tweets) and y in the same manner to eight new sets (testing, training), we put the train size 70% and the test will be 30%. Then, we starting to build our model, 
we begin by defining our inputs (X, desc, tweets), so we’re going to create three different inputs (X_inputs, desc_inputs, tweet_inputs), X_inputs is going to be a vector of length 18(X.shape[1]), desc_inputs is going to be a vector of length 62(desc.shape[1]) each sequence is 62 long and tweet_inputs is going to be a vector of length 104(tweets.shape[1]). 

Now, we have three inputs, we’re going to process each one separately. Let’s begin with X, first we’re going to define two dense layers (X_dense1, X_dense2) and this is just going to be a standard neural network section, and we found 256 activations works and we a ‘relu’ activation function, for the first dense layer we pass (X_inputs) and for the second one we pass (X_dense1).

Now let’s work with description (desc), we’re going to embed the sequences, the embedding layer is going take each sequence and it is going to send each word to a new location in a high dimensional vector space, the embedding layer will take three arguments:

Input dimension(input_dim): this is a from sparse encoding (one hot encoder) to a dense encoding, if we look to description which is our sequences, each one of the 20000 total unique words is going to be mapped to a new location, here we map that vector of length 20000.

Output dimension(output_dim): here we’re going to mapping the vector to a new dimensional space of our choosing and the 256 will work, but this what will do is send each one of these words to a new location in (256) dimensional vector space and the model will actually learn where to send them. (Note: now each word can be represented as a vector of length 256 which is a lot better than represented by a vector of length 20000)
Note: generally, the more intricate the connection between the words the higher dimension space you need.

Input length(input_length): the length of the given sequence.

Because the embedding layer is two-dimensional, we’re going to have some a hard time feeding in the later parts of the model (there will be some errors and some codes will not run), so we’re going to flatten it by take all the rows and put them side by side (will have one long vector).

So, we're not only going to send the embedded words to the final prediction, we’re also going to run the embedded words through GRU (Gated Recurrent Units) which is form long-short-term memory network layer, which is a form of RNN (Recurrent Neural Networks). RNN is uses the previous information in the sequence to produce the current output. the purpose of GRU is to capture time dependencies between the words, in RNN takes as input a given word but also the previous one, so each time it sees a new word, it considers (the past as well as the present).

desc embedding is now being passed into the GRU but also into the flattened layer, so we're going to concatenate the (desc_gru) with (desc_faltten), because each one of them returns a single vector, so we're going to take these two vectors and put them side by side on a new layer (desc_concat), which will contain all the information from the description.

We’re going to do the same steps we do to description inputs on the tweets inputs.

before the end, we’re going to concatenate all three layers 
(X_dense2, desc_concat, tweets_concat) in a new concatenation layer (concat).

Finally, the outputs will be a dense layer with three probability values of each class (male, female, brand), and a soft max activation function will ensure that their values between 0 and 1, so they can be used as a probability estimates.

This is the summary of our model:
![image](https://user-images.githubusercontent.com/102586302/175090232-051c22e1-fe54-4afb-b7e9-3b79a678e30a.png)

 







Our model looks like this:

 

As we see here, we have three different inputs, the left one for the descriptions, the right one for tweets, the middle one for the regular x values.
In the Middle: The regular x values is getting passed through two dense layers, then sent it to the concatenation layer.
In the Right and Left: both being embedded, the embeddings are being sent to a GRU and to a flattened, and the two outputs from GRU and flatten are getting concatenated back together.
The Fifth Row: the final three outputs get concatenated to one big vector>
The Final Row: the concatenated vector sent to a dense layer to give our final three values.


Now for the training, we’re going to compile the model, first we used “adam” optimizer, which (Adaptive Moment Estimation) is an algorithm for optimization technique for gradient descent. The method is really efficient when working with large problem involving a lot of data or parameters. It requires less memory and is efficient. Intuitively, it is a combination of the ‘gradient descent with momentum’ algorithm and the ‘RMSP’ algorithm. 
Then, we used Loss function(sparse_categorical_crossentropy) to find error or deviation in the learning process. Finally, we used 'accuracy' Metrics to evaluate the performance of our model. Then we fit the model by used [X_train, desc_train, tweets_train], y_train, and we decide to put 3 epochs because when we use gru's, we only have to train for a very small number of epochs, usually it is fit best by the first or second epoch.

1.2 Dataset Description:
info about the columns in the dataset:
1. *unitid*: a unique id for the user
2. *golden*: whether the user was included in the gold standard for the model; TRUE or FALSE
3. *trustedjudgments*: number of trusted judgments (int); always 3 for non-golden, and what may be a unique id for gold standard observations
4. *lastjudgment_at*: date and time of last contributor judgment; blank for gold standard observations
5. *gender*: one of male, female, or brand (for non-human profiles)
6. *gender:confidence*: a float representing confidence in the provided gender
7. *profile_yn*: “no” here seems to mean that the profile was meant to be part of the dataset but was not available when contributors went to judge it
8. *profile_yn: confidence*: confidence in the existence/non-existence of the profile
9. *created*: date and time when the profile was created
10. *description*: the user’s profile description
11. *fav_number*: number of tweets the user has favourited
12. *gender_gold*: if the profile is golden, what is the gender?
13. *link_color*: the link colour on the profile, as a hex value
14. *name*: the user’s name
15. *profileyngold*: whether the profile y/n value is golden
16. *profileimage*: a link to the profile image
17. *retweet_count*: number of times the user has retweeted (or possibly, been retweeted)
18. *sidebar_color*: color of the profile sidebar, as a hex value
19. *text*: text of a random one of the user’s tweets
20. *tweet_coord*: if the user has location turned on, the coordinates as a string with the format “[latitude, longitude]”
21. *tweet_count*: number of tweets that the user has posted
22. *tweet_created*: when the random tweet (in the text column) was created
23. *tweet_id*: the tweet id of the random tweet
24. *tweet_location*: location of the tweet; seems to not be particularly normalized
25. *user_timezone*: the timezone of the user
26. *the third columnn*(unitstate: state of the observation; one of finalized (for contributor-judged) or golden (for gold standard observations)
Dataset Domain: Twitter User Gender Classification.
Reason for selection, this dataset contains a lot of categorical columns such (text, description), and we want to learn how to handle, cleaning, training, testing this type of columns, specially (text, description) and the date/time columns like (created, tweet_created).
The data was processed through several steps:
At first, checking if there are null values, we see that there are columns with a high percentage of null values.
We check for unnecessary columns that contain a large number of odd values. we find that, there are many unnecessary columns.
Before dropping the columns, we look at the gender column. We see that there are unknown values in the gender columns, which are not useful, so the unknown will be replaced by Nan using Np.NaN, and then the drop columns
We use a string of true and false missing values in the gender column as a frame index for the original dataframe 
We see that we have varying lengths but it's not a great way to get them into the model so we're going to include all the sequences, so they take the same length by adding zeros to the end 
after drop the unnecessary columns with high number of unique values we now have 21 columns  
Take a look at the unique values in the purposeful columnb
Then we see the replacement of the unknown values by nan. Now we will drop the missing target values, after the drop columns with than 30% missing values, we have 16 columns and 18836 rows, now let’s check the remaining missing values 
We only have two columns with remaining missing values 
1. _last_judgment_at
2. description
we must understand what kind of data in these two columns.  
There are only 50 remaining missing values in the _last_judgment_at column, so let's drop those rows.
We only have the missing values in the description column, and due to its importance, we encode the values in it as empty strings.
There is still a lot of data that we deal with on behalf of the vertical columns, and they are ready to be entered into the form after scaling them. As for the categorical values, they must be converted in some way to enter the form.
Now we get rid of the missing values and left with 12 new columns of information and then we deal with the description and the text by feeding the pieces of this information separately into a series of distinct words.
We drop the text and description columns from the original df, and store it with a new form in the desc, tweets vars.

2: Results and discussion
First, we evaluate the model on the test set [X_test, desc_test, tweets_test, y_test], then, we display the model accuracy which was 65.19%, which is actually a very decent result, it is quite good based. 
Then, we define (y) predict, which was the index of the highest probability from the three different probability values which will be our actual classification.
Then we used a confusion matrix and classification report as evaluation measures for our model, to know our model performance.
Now, we visualize our final output to know our model performance:
 
we can see here the number of predicted values for each category and compared with number of actual values for each category.
1104: the number of predicted values in female’s category, which was the correct predicted values (Female).
393: the number of predicted values in female’s category, which wasn’t the correct predicted values, the model predicted here the value is (Male).
104: the number of predicted values in female’s category, which wasn’t the correct predicted values, the model predicted here the value is (Brand).
__________________________________________________________________
696: the number of predicted values in male’s category, which wasn’t the correct predicted values, the model predicted here the value is (Female).
1182: the number of predicted values in male’s category, which was the correct predicted values (Male).
361: the number of predicted values in male’s category, which wasn’t the correct predicted values, the model predicted here the value is (Brand).
__________________________________________________________________
204: the number of predicted values in Brand’s category, which wasn’t the correct predicted values, the model predicted here the value is (Male).
204: the number of predicted values in Brand’s category, which wasn’t the correct predicted values, the model predicted here the value is (Female).
1388: the number of predicted values in female’s category, which was the correct predicted values (Brand).

3: Conclusions
Our classification report was:
	precision    	Recall	F1-score	Support
Female	0.69	0.55	0.61	2004
Male	0.53	0.66	0.59	 1779
Brand	0.77	0.75	0.76	1853
Accuracy			0.65	5636
Macro avg	0.66	0.65	0.65	5636
Weighted avg	0.67	0.65	0.65	5636
As we see, the most easily classified category was a 
Brand: because precision = 0.77 which means that, 77% of our values were correct in the predictions. Recall = 0.75 which means out of all actual Brand values 75% were correct. Which means it is the easiest for the model to classify.
Female: precision = 0.69 which means that, 69% of our values were correct in the predictions. Recall = 0.55 which means out of all actual female values 55% were correct.
Male: precision = 0.53 which means that, 53% of our values were correct in the predictions. Recall = 0.66 which means out of all actual female values 66% were correct. Which looks like a male category probably our worst category. Which means it is the hardest for the model to classify.

Finally, we want to display some important values:
Model running time was: 505. 65333342552185s
Model Accuracy: 65.19% which is good
Disk space: 38.40 GB
Macro avg(f1-score): 65% / Macro avg(precision): 67% / Macro avg(recall): 65%
weighted avg(f1-score): 65% / weighted avg(precision): 66% / weighted avg(recall): 65%
