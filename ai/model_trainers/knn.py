from base import BaseStrategyModelTrainer
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import joblib


class KNNStrategyModelTrainer(BaseStrategyModelTrainer):

    def train(self):
        dataframe = self.load_data_frame()
        training_df = self.populate_features(dataframe)

        # Split the data into training and test datasets
        x = training_df[['feature_1', 'feature_2']]
        y = training_df['label']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Train and Test the KNN classifier/Model
        knn = KNeighborsClassifier(n_neighbors=1000)
        knn.fit(x_train, y_train)
        y_pred_knn = knn.predict(x_test)

        # Print classification report for KNN
        print("KNN: \n", classification_report(y_test, y_pred_knn))
        # Save Models
        joblib.dump(knn, 'knn_model.pkl')
