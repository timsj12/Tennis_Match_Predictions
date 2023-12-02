from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import s3
sns.set()


def pre_post_compare(**kwargs):
    ti = kwargs['ti']
    datasets = ti.xcom_pull(task_ids='set_data', key='datasets')

    prefix = {'atp': 0, 'wta': 0 }

    for dataset in datasets:
        tour = dataset[:3]
        prefix[tour] += 1

    print(prefix)

    for tour in prefix.keys():

        if prefix[tour] != 2:
            continue

        df = s3.load_file(f'{tour}_post_match_ml_model_metrics.pkl', f'/{tour.upper()}/POST_MATCH/METRICS')
        df = df.sort_index()

        df2 = s3.load_file(f'{tour}_pre_match_ml_model_metrics.pkl', f'/{tour.upper()}/PRE_MATCH/METRICS')
        df2 = df2.sort_index()

        for i in range(1, len(df.columns)):

            # Create bar plot
            fig, ax = plt.subplots()

            x = np.arange(len(df['Model']))

            bar_width = 0.25

            # Plot bars for df
            bar1 = plt.bar(x - bar_width/2, df.iloc[:, i], width = bar_width, color='blue',label='Post Match')

            # Plot bars for df2
            bar2 = plt.bar(x + bar_width/2, df2.iloc[:, i], width = bar_width, color='orange',label='Pre Match')

            # bars = plt.bar(df['Model'], df.iloc[:, i], color=df['Model'].map(value_color_mapping).fillna('gray'))
            ax.set_xticks(x, df['Model'])
            ax.set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=10)
            # ax.set.xlabel('Model')
            ax.set_title(f'{tour.upper()} {df.columns[i]}')

            if df.columns[i] != 'Time':
                ax.set_ylim(0, 1)
                ax.set_ylabel('Score')
            else:
                ax.set_ylabel('Seconds (s)')

            ax.set_xlabel('Model')

            plt.tight_layout()
            plt.legend()

            image_name = f'{tour}_metric_pre_post_comparison_{df.columns[i]}.png'

            s3.save_plot(bar2, image_name, f'/{tour.upper()}/PRE_POST_COMPARISON/', False)

            plt.close()
