import logging
import pandas as pd
import s3


def ingest_data():
    def get_base_dir(**kwargs):
        ti = kwargs['ti']
        BASE_DIR = ti.xcom_pull(key='BASE_DIR', task_ids='set_initial_inputs')
        return BASE_DIR

    def get_base_bucket(**kwargs):
        ti = kwargs['ti']
        BUCKET_DIR = ti.xcom_pull(key='BUCKET_DIR', task_ids='set_initial_inputs')
        return BUCKET_DIR

    BASE_DIR = get_base_dir()
    BUCKET_DIR = get_base_bucket()

    df = pd.DataFrame()

    for i in range(2023, 1989, -1):

        csv_url = f'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{i}.csv'
        csv = pd.read_csv(csv_url, on_bad_lines='skip')
        df = pd.concat([df, csv])

    logging.info(f'DataFrame Created; Shape {df.shape}')

    s3.write_file(df,'atp_raw_data.pkl', '/Data')

