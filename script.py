
from scipy.spatial import distance #import the distance module from scipy's spatial package

# function to return the euclidian distance between to lists a and b (i.e. forEa point in each list for n dimensions)
def euc(a,b):
    return distance.euclidean(a,b)
# knn class
class knn():
    # fit method of knn: fits data (i.e. adds it to the class)
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    # predict method for knn: predicts something based on features from test data (X_test)
    def predict(self, X_test):
        # init predictions list
        predictions = []
        # loop through every list of features (obs)
        for row in X_test:
            # forEa row call the closest function and save it in variable label
            label = self.closest(row)
            # append the label to the list of predictions (one element represents the prediction for a given row of features -- obs)
            predictions.append(label)
        return predictions

    # closest method for knn: finds the closes point based on euc distance
    def closest(self, row):
        # init the shortest distance as dist between current obs and 1st obs in training
        best_dist = euc(row, self.X_train[0])
        # init bindex of shortest dist as 0
        best_index = 0
        # loop from 1 to length of training observations
        for i in range(1, len(self.X_train)):
            # forEa: set dist to distance between observation(input) and ith obs in training
            dist = euc(row, self.X_train[i])
            #if dist is less than the init best_dist then set best_dist as current dist
            if dist < best_dist:
                best_dist = dist
                # set best _index as the index of the shortest index
                best_index = i
        #return the label from the training data with the best_index index
        return self.Y_train[best_index]

# get user information for features of the data
def get_user_feature():
    user_observation = []
    for feature in feature_names:
        feature_input = input(f"Pass me some data for {feature}:")
        user_observation.append(float(feature_input))
    return list(user_observation)

# make a prediction from the user's feature input
def make_user_pred():
    user_pred_data = list([get_user_feature()])
    magical_prediction = clf.predict(user_pred_data)
    print(f"Looks like it's a {magical_prediction}")

#load pandas
import pandas as pd

# ask the user what the path to the data set is
data_path = input("What is the path to your labeled dataset? The data set (1) must be a csv (2) be numerical.")

#read in data from user's input for the pathname
data_path = str(data_path)
data_set = pd.read_csv(str(data_path))

#ask the user for th ename of the labeles column
target_name = input("What is the name of the labels (target/outcome) column? (case & white-space sensitive!)")

# creare a dataframe of just the features
features_data = data_set.drop([target_name], axis=1)
# save column names of features
feature_names = list(features_data.columns.values)
# save features data as a list of lists where each row (observation) is a list
X = features_data.values.tolist()
#save the labels as a list
Y = data_set[target_name].tolist()

#split the data into training and testing features and labels
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .5)

# set clf = knn() algorithm and fit the X and Y values for training
clf = knn()
clf.fit(X_train, Y_train)

# save the predictions from the features in the test data
predictions = clf.predict(X_test)

#save the accuracy of our model
from sklearn.metrics import accuracy_score
acc_score = accuracy_score(Y_test, predictions)

#ask the user whether they want to make a prediction from their own data
make_pred = input(f"The model is {acc_score * 100}% accurate, would you like to make a prediction? (y/n)")

# if user said yes, make a prediction
if make_pred == "y":
    make_user_pred()
