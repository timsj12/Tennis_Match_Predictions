import logging
import pandas as pd
import s3


def ingest_data(**kwargs):

    ti = kwargs['ti']
    datasets = ti.xcom_pull(task_ids='set_data', key='datasets')

    print(datasets)

    unique_prefix = set()

    for dataset in datasets:
        subset = dataset[:3]
        if subset not in unique_prefix:
            unique_prefix.add(subset)

    for tour in unique_prefix:
        df = pd.DataFrame()

        print(tour)

        for i in range(2023, 1989, -1):

            csv_url = f'https://raw.githubusercontent.com/JeffSackmann/tennis_{tour}/master/{tour}_matches_{i}.csv'
            csv = pd.read_csv(csv_url, on_bad_lines='skip')
            df = pd.concat([df, csv])

        logging.info(f'DataFrame Created; Shape {df.shape}')

        s3.write_file(df,f'{tour}_raw_data.pkl', f'/{tour.upper()}')
