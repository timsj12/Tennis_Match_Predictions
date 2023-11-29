import logging
import pandas as pd
import numpy as np
import s3
from sklearn import preprocessing

def transform_data(**kwargs):
    ti = kwargs['ti']
    datasets = ti.xcom_pull(task_ids='set_data', key='datasets')

    for dataset in datasets:

        tour = dataset[:3]
        pre_post = dataset[4:]

        df = s3.load_file(f'{tour}_raw_data.pkl', f'/{tour.upper()}')

        df.drop_duplicates()

        if pre_post == 'post':

            df.drop(columns=['tourney_id', 'tourney_name', 'tourney_level', 'match_num', 'winner_entry', 'winner_seed','winner_ioc',
                             'loser_ioc', 'loser_seed', 'loser_entry', 'score', 'winner_rank', 'l_bpFaced', 'w_bpFaced','loser_rank',
                             'tourney_date', 'winner_name', 'loser_name'], inplace=True)

            df.dropna(
                subset=['w_ace', 'w_df', 'loser_hand', 'winner_hand', 'winner_rank_points', 'loser_rank_points', 'loser_age', 'winner_age',
                         'l_ace', 'l_df','w_bpSaved', 'w_svpt', 'w_SvGms', 'l_svpt', 'l_SvGms','l_bpSaved'],
                inplace=True)

            df['minutes'].fillna(df['minutes'].mean(), inplace=True)

        else:
            df.drop(columns=['tourney_id', 'tourney_name', 'tourney_level', 'match_num', 'winner_entry', 'winner_seed', 'winner_ioc',
                             'loser_ioc', 'loser_seed', 'loser_entry', 'score','w_bpFaced', 'l_bpFaced','w_1stWon', 'l_1stWon', 'winner_rank', 'loser_rank',
                             'tourney_date', 'winner_name', 'loser_name', 'minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn','w_2ndWon', 'w_SvGms',
                             'w_bpSaved', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_2ndWon', 'l_SvGms', 'l_bpSaved'], inplace=True)

            df.dropna(subset=[ 'loser_hand', 'winner_hand', 'winner_rank_points', 'loser_rank_points', 'loser_age', 'winner_age'], inplace=True)

        df['winner_ht'].fillna(df['winner_ht'].mean(), inplace=True)
        df['loser_ht'].fillna(df['loser_ht'].mean(), inplace=True)

        df.reset_index(drop=True, inplace=True)
        random_sample = df.sample(frac=0.5)
        random_index = random_sample.index.to_numpy()

        all_index = np.arange(df.shape[0])
        other_index = np.setdiff1d(all_index, random_index)

        # create new columns and assign values of NA
        for column in df.columns:
            if column[0:7] == 'winner_':
                new_col = 'p0' + column[6:]
                df[new_col] = pd.NA
            elif column[0:2] == 'w_':
                new_col = 'p0' + column[1:]
                df[new_col] = pd.NA
            elif column[0:6] == 'loser_':
                new_col = 'p1' + column[5:]
                df[new_col] = pd.NA
            elif column[0:2] == 'l_':
                new_col = 'p1' + column[1:]
                df[new_col] = pd.NA

        df['match_winner'] = pd.NA

        """
        Take half the the random indexes
        assign winner to player 0
        assign loser to player 1
        copy data to new columns
        """
        logging.info(f'Winner = Player 0 Start')
        df = create_player(df, random_index, 0)
        logging.info(f'Winner = Player 0 Complete')

        """
        Take other half the the random indexes
        assign winner to player 1
        assign loser to player 0
    
        copy data to new columns
        """
        logging.info(f'Winner = Player 1 Start')
        df = create_player(df, other_index, 1)
        logging.info(f'Winner = Player 1 Complete')

        # drop winner and loser columns for ML analysis
        for column in df.columns:
            if column[0:7] == 'winner_':
                df.drop(columns=column, inplace=True)
            elif column[0:2] == 'w_':
                df.drop(columns=column, inplace=True)
            elif column[0:6] == 'loser_':
                df.drop(columns=column, inplace=True)
            elif column[0:2] == 'l_':
                df.drop(columns=column, inplace=True)

        # One Hot Encode Surface, Winner Hand, Loser Hand, Round columns
        ohe = preprocessing.OneHotEncoder(dtype=int, sparse=False, handle_unknown="ignore")
        ohe_data = ohe.fit_transform(df[['surface', 'draw_size', 'p0_hand', 'p1_hand', 'round']])
        ohe_df = pd.DataFrame(ohe_data, columns=ohe.get_feature_names_out())

        df.drop(columns=['surface', 'draw_size', 'p0_hand', 'p1_hand', 'round'], inplace=True)
        df = pd.concat([df, ohe_df], axis=1)

        s3.write_file(df,f'{tour}_{pre_post}_match_clean_data.pkl', f'/{tour.upper()}/{pre_post.upper()}_MATCH')



def create_player(data_frame, random_rows, winner):
    if winner == 0:
        loser = 1
    else:
        loser = 0

    for i in random_rows:
        for column in data_frame.columns:
            if column[0:7] == 'winner_':
                new_col = 'p' + str(winner) + column[6:]
                data_frame.loc[i, new_col] = data_frame.loc[i, column]
            elif column[0:2] == 'w_':
                new_col = 'p' + str(winner) + column[1:]
                data_frame.loc[i, new_col] = data_frame.loc[i, column]
            elif column[0:6] == 'loser_':
                new_col = 'p' + str(loser) + column[5:]
                data_frame.loc[i, new_col] = data_frame.loc[i, column]
            elif column[0:2] == 'l_':
                new_col = 'p' + str(loser) + column[1:]
                data_frame.loc[i, new_col] = data_frame.loc[i, column]

        # assign match winner to 0 (player 0)
        data_frame.loc[i, 'match_winner'] = winner

    return data_frame

