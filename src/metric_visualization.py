from matplotlib import pyplot as plt
import seaborn as sns
import s3
sns.set()

def graph_metrics(**kwargs):
    ti = kwargs['ti']
    datasets = ti.xcom_pull(task_ids='set_data', key='datasets')

    for dataset in datasets:
        tour = dataset[:3]
        pre_post = dataset[4:]

        df = s3.load_file(f'{tour}_{pre_post}_match_ml_model_metrics.pkl', f'/{tour.upper()}/{pre_post.upper()}_MATCH/METRICS')

        value_color_mapping = {
            'Multi-Layer Perceptron': 'tomato',
            'Gradient Boost': 'lightskyblue',
            'Logistic Regression': 'rosybrown',
            'Random Forest': 'lightsalmon',
            'Decision Tree': 'cyan',
            'Gaussian Naive Bayes': 'brown',
            'Bernoulli Naive Bayes': 'pink',
        }

        for i in range(1, len(df.columns)):
            plt.xticks(rotation=45, ha='right', fontsize=10)
            # Bar plot with varying colors for each 'Model'

            bars = plt.bar(df['Model'], df.iloc[:, i], color=df['Model'].map(value_color_mapping).fillna('gray'))
            plt.xlabel('Model')
            plt.title(f'{tour.upper()} {pre_post.upper()}-Match - {df.columns[i]}')
            plt.tight_layout()

            if df.columns[i] != 'Time':
                plt.ylim(0, 1)
                plt.ylabel('Score')
            else:
                plt.ylabel('Seconds (s)')

            image_name = f'{tour}_{pre_post}_match_{df.columns[i]}.png'

            # Add labels and a title
            s3.save_plot(bars, image_name, f'/{tour.upper()}/{pre_post.upper()}_MATCH/METRICS/VISUALIZATIONS/', False)

            plt.close()