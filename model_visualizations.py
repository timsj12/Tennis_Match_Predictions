from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import ML_Models
import pandas as pd
import seaborn as sns
import s3
sns.set()


def visualize_predictions(**kwargs):

    ti = kwargs['ti']
    datasets = ti.xcom_pull(task_ids='set_data', key='datasets')

    for dataset in datasets:

        tour = dataset[:3]
        pre_post = dataset[4:]

        X_train = s3.load_file(f'X_{tour}_{pre_post}_train.pkl', f'/{tour.upper()}/{pre_post.upper()}_MATCH')
        X_test = s3.load_file(f'X_{tour}_{pre_post}_test.pkl', f'/{tour.upper()}/{pre_post.upper()}_MATCH')
        y_train = s3.load_file(f'y_{tour}_{pre_post}_train.pkl', f'/{tour.upper()}/{pre_post.upper()}_MATCH')
        y_test = s3.load_file(f'y_{tour}_{pre_post}_test.pkl', f'/{tour.upper()}/{pre_post.upper()}_MATCH')

        for ml_model in ML_Models.getModels():

            ml_model.model.fit(X_train, y_train)

            y_model = ml_model.model.predict(X_test)
            print(ml_model.name)
            fig1 = ConfusionMatrixDisplay.from_predictions(y_test, y_model)
            image_predictions = fig1.figure_
            image_name = f'{tour}_{pre_post}_{ml_model.name}_Results_Matrix.png'
            plt.title(f'{tour.upper()} {pre_post.upper()}-Match {ml_model.name}')
            s3.save_plot(image_predictions, image_name, f'/{tour.upper()}/{pre_post.upper()}_MATCH/Models/')

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

                plt.title(f'{tour.upper()} {pre_post.upper()}_match_Top 10 Feature Importances')
                image_name = f'{tour}_{pre_post}_match_RF_Top_Features.png'
                s3.save_plot(fig3, image_name, f'/{tour.upper()}/{pre_post.upper()}_MATCH/Models/', False)
