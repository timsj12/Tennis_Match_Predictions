from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import ML_Models
import pandas as pd
import seaborn as sns
import s3
sns.set()


def visualize_predictions():

    X_train = s3.load_file('X_train.pkl', '/Data')
    X_test = s3.load_file('X_test.pkl', '/Data')
    y_train = s3.load_file('y_train.pkl', '/Data')
    y_test = s3.load_file('y_test.pkl', '/Data')

    for ml_model in ML_Models.getModels():

        ml_model.model.fit(X_train, y_train)

        y_model = ml_model.model.predict(X_test)
        print(ml_model.name)
        fig1 = ConfusionMatrixDisplay.from_predictions(y_test, y_model)
        image_predictions = fig1.figure_
        image_name = ml_model.name + '_Results_Matrix.png'
        plt.title(ml_model.name)
        s3.save_plot(image_predictions, image_name, '/Models/')

        if ml_model.name == "Random Forest":
            feature_names = X_train.columns

            # Get feature importances
            importances = ml_model.model.feature_importances_

            # Create a DataFrame to associate feature names with importances
            feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

            # Sort the features by importance (descending)
            feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

            # Select the top 20 important features
            top_10_features = feature_importances.head(20)
            print(top_10_features)

            # Increase the figure size
            fig3 = plt.figure(figsize=(12, 8))

            # Create a bar plot of the top 10 important features with a color palette
            sns.barplot(x='Importance', y='Feature', data=top_10_features, palette='viridis')

            # Rotate the y-axis labels for better readability
            plt.yticks(rotation=0)

            plt.title('Top 10 Feature Importances')
            image_name = 'RF_Top_Features.png'
            s3.save_plot(fig3, image_name, '/Models/', False)
