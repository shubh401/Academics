# Statistical Natural Language Processing - Summer, 2019 (Course Instructor - Prof. Dr. Dietrich Klakow)

#### The course project dealt with the classification of tweets as benign or abusive using natural language processing and analytical techniques and involved key functionalities as follows:

1. Text pre-processing - Tokenization, vectorization, handling special characters and emojis, etc.

2. Feature extraction.

3. Classification using traditional ML algorithms - Naive Bayes & SVM classifiers.

4. Comparison of results obtained from above-mentioned classifiers with the result obtained from neural network-based classification.  


Few implementation-specific details are as follows:

1. SNLP_Project.py is the main code file while others are dependencies include. The code can be executed by running the main python file.

2. The datasets are also available in the same directory and are read from the same directory.

3. Executing the code would generate prediction.txt for the test dataset. An already generated version is available in the same directory.

4. mismatchedTweets.txt is also included which contains the mispredicted tweets of the reported values and tweets in the main report for verification. 

5. Complete execution of code may take time ranging from 15 minutes to 1 hour depending upon system and resources available.

6. In the report, it is mentioned that both linear & RBF kernel of SVM is implemented. The code is not separately written. But, it can be verified by just changing the kernel value in SVMClassifier() function.