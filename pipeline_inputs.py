

"""
Comment out data sets that are not desired to be examined below.
Maximum of four pipelilines to be created

Men's pre match data
Men's post match data

Women's pre match data
Women's post match data
"""


def set_dataset_inputs(**kwargs):

    predictions = []

    # Look at Men's Tennis Data and make predictions on post match data
    predictions.append('atp_post')

    # Look at Men's Tennis Data and make predictions on matchup only and no match metrics
    predictions.append('atp_pre')

    # Look at Men's Tennis Data and make predictions on post match data
    predictions.append('wta_post')

    # Look at Men's Tennis Data and make predictions on matchup only and no match metrics
    predictions.append('wta_pre')

    print(predictions)

    ti = kwargs['ti']
    ti.xcom_push(key='datasets', value=predictions)

