import logging
import pandas as pd
import s3


def ingest_data(**kwargs):
    ti = kwargs['ti']
    datasets = ti.xcom_pull(task_ids='set_data', key='datasets')

    # Check if the pulled value is not None before using it
    if datasets is not None:
        logging.info(f"Pulled datasets from XCom: {datasets}")
        # Rest of your logic using the pulled value
    else:
        logging.warning("No datasets found in XCom.")

    df = pd.DataFrame()

    for i in range(2023, 1989, -1):

        csv_url = f'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{i}.csv'
        csv = pd.read_csv(csv_url, on_bad_lines='skip')
        df = pd.concat([df, csv])

    logging.info(f'DataFrame Created; Shape {df.shape}')

    s3.write_file(df,'atp_raw_data.pkl', '/Data')
