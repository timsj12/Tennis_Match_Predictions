from matplotlib import pyplot as plt
import seaborn as sns
import s3
sns.set()

def graph_metrics():

    df = s3.load_file('atp_ml_model_metrics.pkl', '/Metrics')

    value_color_mapping = {
        'Multi-Layer Perceptron': 'tomato',
        'Support Vector': 'green',
        'Gradient Boost': 'lightskyblue',
        'Logistic Regression': 'rosybrown',
        'Random Forest': 'lightsalmon',
        'Decision Tree': 'cyan',
        'Gaussian Naive Bayes': 'brown',
        'Bernoulli Naive Bayes': 'pink',
    }

    for i in range(1, len(df.columns)):
        plt.xticks(rotation=45)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        # Bar plot with varying colors for each 'Model'

        bars = plt.bar(df['Model'], df.iloc[:, i], color=df['Model'].map(value_color_mapping).fillna('gray'))
        plt.xlabel('Model')
        plt.title(df.columns[i])
        plt.tight_layout()

        if df.columns[i] != 'Time':
            plt.ylim(0, 1)

        image_name = df.columns[i] + '.png'

        # Add labels and a title
        s3.save_plot(bars, image_name, '/Metrics/', False)