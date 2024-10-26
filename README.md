# Fashion-MNIST Dataset👕
Two machine learning models were evaluated: Multi-layer Perceptron (MLP) and Convolutional Neural Networks (CNN).

1. Multi-layer Perceptron (MLP):
The MLP model, trained for ten epochs using Keras, achieved a test accuracy of 88.16% and indicated no overfitting, with close training and validation accuracies.

2. Convolutional Neural Networks (CNN):
The CNN model outperformed the MLP, achieving a test accuracy of 91.96% after five epochs. Similar to the MLP, it showed no signs of overfitting, and batch normalization was excluded due to its adverse effects on performance.

Overall, the CNN demonstrated superior classification performance on the Fashion-MNIST dataset.

# Iris Dataset🌸
Three machine learning models were evaluated: Naïve Bayes Classifier, Support Vector Machine (SVM), and Decision Trees.

1. Naïve Bayes Classifier:
The Naïve Bayes model achieved a test accuracy of 93.33%. It demonstrated strong performance, particularly with the Setosa class, which had a perfect precision, recall, and F1-score of 1.00. The model misclassified 4 out of 60 samples.

2. Support Vector Machine (SVM):
The SVM model outperformed the Naïve Bayes classifier, achieving a test accuracy of 95%. Like the Naïve Bayes model, it excelled in classifying Setosa, while also showing high precision and recall for Versicolor and Virginica. The SVM misclassified 3 out of 60 samples.

3. Decision Trees:
The Decision Tree model demonstrated variable performance based on its max depth. The highest test accuracy of 98.33% was observed at a max depth of 6. However, the model exhibited signs of overfitting, with perfect training accuracy but a decline in test accuracy beyond a max depth of 6. It misclassified 3 out of 60 samples, similar to the SVM.

Overall, the SVM model showed the best classification performance on the Iris dataset, followed closely by the Decision Trees, while the Naïve Bayes Classifier provided solid but comparatively lower accuracy.

# Wine Quality Dataset🍷
Several machine learning methods were applied for both classification and regression tasks, yielding various results.

1. Decision Trees: 
The Decision Tree classifier was trained with varying maximum depths. After splitting the dataset into features (X) and target variable (y), and defining a range of max_depth values, the following accuracies were observed:

Max Depth 1: Train Accuracy: 57.25%, Test Accuracy: 51.41%
Max Depth 2: Train Accuracy: 58.19%, Test Accuracy: 52.97%
Max Depth 3: Train Accuracy: 61.42%, Test Accuracy: 56.56%
Max Depth 4: Train Accuracy: 65.90%, Test Accuracy: 59.22%
Max Depth 5: Train Accuracy: 69.13%, Test Accuracy: 56.25%
Max Depth 6: Train Accuracy: 73.93%, Test Accuracy: 57.97%
Max Depth 7: Train Accuracy: 78.62%, Test Accuracy: 57.97%
Max Depth 8: Train Accuracy: 83.11%, Test Accuracy: 56.72%
Max Depth 9: Train Accuracy: 87.07%, Test Accuracy: 56.25%
Max Depth 10: Train Accuracy: 90.41%, Test Accuracy: 56.88%

The results indicate that while training accuracy increased with depth, the test accuracy plateaued or declined after a depth of 5 or 6, indicating overfitting. The overall test accuracy was 57%, with 276 out of 640 points mislabeled.

2. Support Vector Machine (SVM): 
For SVM classification, the dataset was standardized using StandardScaler. The SVM model achieved the following evaluation metrics:

Accuracy: 60%
Precision: 0.59
Recall: 0.60
This method performed moderately in terms of classification accuracy.

3. Multi-layer Perceptron (MLP) - Classification: 
The MLP model was constructed and trained with the quality column converted into binary labels. After training for 10 epochs, the model showed significant improvement:

Test Accuracy: 87.34%
Test Loss: 0.30
The model's performance improved steadily throughout the training, achieving an accuracy of around 89% on the training data, indicating that MLP is a strong choice for classification on this dataset.

4. Multi-layer Perceptron (MLP) - Regression: 
An MLP was also employed for regression on the Wine Quality dataset. The model's evaluation metrics were:

Mean Squared Error: 0.47
R-squared: 0.28
The plot of actual vs. predicted wine quality suggested a positive correlation between predicted and actual values.

5. Linear Regression: 
Linear regression was applied with similar preprocessing steps as in the MLP regression. The model's performance metrics were:

Mean Squared Error: 0.39
R-squared: 0.40
Scatter plots revealed insights into the relationships between alcohol content and wine quality, as well as between volatile acidity and quality.

In summary, the MLP classifier demonstrated the best accuracy for the Wine Quality dataset, outperforming both the Decision Tree and SVM methods. For regression tasks, while both MLP and linear regression were applied, the linear regression model performed slightly better based on lower mean squared error and higher R-squared values.
