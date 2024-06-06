import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# load dataset
data = pd.read_csv('spam.csv', encoding='latin-1')

# keep only necessary columns
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

# encode labels: 'ham' as 0 and 'spam' as 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# initialize CountVectorizer
vectorizer = CountVectorizer()

# fit and transform the training data
X_train_counts = vectorizer.fit_transform(X_train)

# transform the testing data
X_test_counts = vectorizer.transform(X_test)

# initialize the model
model = MultinomialNB()

# train the model
model.fit(X_train_counts, y_train)

# predict the labels for the test set
y_pred = model.predict(X_test_counts)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# classification report
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)

# function to predict if an email is spam or not
def predict_spam(email):
    email_counts = vectorizer.transform([email])
    prediction = model.predict(email_counts)
    return 'Spam' if prediction[0] == 1 else 'Not Spam'

# test
test_email = "Congratulations! You've won a $1000 gift card. Click here to claim your prize."
print(predict_spam(test_email))
