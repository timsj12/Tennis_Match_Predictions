import io
import boto3
import s3fs
from matplotlib import pyplot as plt
from s3fs.core import S3FileSystem
import pickle
import numpy as np
import tempfile

def get_base_dir(**kwargs):
    ti = kwargs['ti']
    BASE_DIR = ti.xcom_pull(key='BASE_DIR', task_ids='task_producer')
    return BASE_DIR

def get_base_bucket(**kwargs):
    ti = kwargs['ti']
    BUCKET_DIR = ti.xcom_pull(key='BUCKET_DIR', task_ids='task_producer')
    return BUCKET_DIR

BASE_DIR = get_base_dir()
BUCKET_DIR = get_base_bucket()

def write_file(df, file, sub=''):
    s3 = S3FileSystem()
    DIR = BASE_DIR + sub
    # Push data to S3 bucket as a pickle file
    with s3.open('{}/{}'.format(DIR, file), 'wb') as f:
        f.write(pickle.dumps(df))

def load_file(file, sub=''):
    s3 = S3FileSystem()
    DIR = BASE_DIR + sub                    # Insert here
    # Get data from S3 bucket as a pickle file
    df = np.load(s3.open('{}/{}'.format(DIR, file)), allow_pickle=True)
    return df

def write_ml_model(model, sub=''):
    s3 = S3FileSystem()
    with tempfile.TemporaryDirectory() as tempdir:
        model.save(f"{tempdir}/{model.name}.h5")
        DIR = BASE_DIR + sub
        # Push saved temporary model to S3
        s3.put(f"{tempdir}/{model.name}", f"{DIR}/{model.name}.h5")

def save_plot(fig, image_name, sub='', grid=True):

    if grid:
        plt.grid(False)

    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)

    DIR = BUCKET_DIR
    print(DIR)

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(DIR)
    bucket.put_object(Body=img_data, ContentType='image/png', Key='Tennis_Historical_Match_Investigation' + sub + image_name)



