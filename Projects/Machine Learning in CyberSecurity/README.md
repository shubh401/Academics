# Machine Learning in Cyber Security - Winter, 2019-20 (Course Instructor - Prof. Dr. Mario Fritz)

#### The objective of this project is two-fold as follows:

1. Assessment of different adversarial attacks on machine learning model used for critical functionalities such as the diagnosis of medical diseases.

2. Generate new artificial data points for the selected dataset using *Generative Adversarial Networks (GAN)* and further assess its similarity to the actual dataset and the impact of the adversarial attacks as above.

The implementation details are as follows:

1. Dataset - [Diabetic Retinopathy Detection, Kaggle](https://www.kaggle.com/c/diabetic-retinopathy-detection/data). It contains data points divided into five distinct classes (0 - 4) representing the stage of the disease in ascending order. 

2. Multi-class classification using CNN over the entire dataset.

3. Binary Classification over the entire dataset, i.e., No Disease Stage vs Diseased Stage or (Class 0 vs Class {1,2,3,4}).

4. Five different adversarial attacks on both the above classifiers and comparison of the performance of the results in both cases.

5. DC-GAN used to generate new data points indpendently for each class.

5. Assessment of the impact of the previously considered adversarial attacks on the synthetic dataset generated in the above step.