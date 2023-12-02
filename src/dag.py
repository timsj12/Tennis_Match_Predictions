from datetime import timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime
from pipeline_inputs import set_dataset_inputs
from batch_ingest import ingest_data
from transform import transform_data
from model_metrics import compare_models
from metric_visualization import graph_metrics
from metric_comparison import pre_post_compare
from final_data_train import test_train_data
from model_visualizations import visualize_predictions

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 10, 12),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

dag = DAG(
    'batch_ingest_dag',
    default_args=default_args,
    description='ingest tennis data',
    schedule_interval=timedelta(days=1),
)

inputs_etl = PythonOperator(
    task_id='set_data',
    python_callable=set_dataset_inputs,
    provide_context=True,
    dag=dag,
)

ingest_etl = PythonOperator(
    task_id='ingest_data',
    python_callable=ingest_data,
    provide_context=True,
    dag=dag,
)

transform_etl = PythonOperator(
    task_id='transform_dataset',
    python_callable=transform_data,
    provide_context=True,
    dag=dag,
)

compare_etl = PythonOperator(
    task_id='machine_learning_metrics',
    python_callable=compare_models,
    provide_context=True,
    dag=dag,
)

graph_metrics_etl = PythonOperator(
    task_id='machine_learning_metrics_graphs',
    python_callable=graph_metrics,
    provide_context=True,
    dag=dag,
)

pre_post_graph_etl = PythonOperator(
    task_id='pre_post_graphs',
    python_callable=pre_post_compare,
    provide_context=True,
    dag=dag,
)

final_data_etl = PythonOperator(
    task_id='create_final_test_train_dataset',
    python_callable=test_train_data,
    provide_context=True,
    dag=dag,
)

visualizations_etl = PythonOperator(
    task_id='create_model_plots',
    python_callable=visualize_predictions,
    provide_context=True,
    dag=dag,
)



inputs_etl >> ingest_etl >> transform_etl >> compare_etl >> graph_metrics_etl >> pre_post_graph_etl >> final_data_etl >> visualizations_etl