import logging
import pandas as pd
import s3


def ingest_data():
    df = pd.DataFrame()

    for i in range(2023, 1989, -1):

        csv_url = f'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{i}.csv'
        csv = pd.read_csv(csv_url, on_bad_lines='skip')
        df = pd.concat([df, csv])

    logging.info(f'DataFrame Created; Shape {df.shape}')

    s3.write_file(df,'atp_raw_data.pkl', '/Data')
