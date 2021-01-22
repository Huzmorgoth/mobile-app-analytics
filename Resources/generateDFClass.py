import pandas as pd
import numpy as np
import random


class GenerateDataset:
    def __init__(self):
        pass

    def build(self):
        # empty dataset with 150k rows
        df = pd.DataFrame({'user_id': 0,
                           'user_category': None,
                           'event': None,
                           'time': pd.NaT},
                          index=list(range(150000)))

        # generate 7k user_IDs randomly
        user_ids = np.arange(7000) + 1

        # user categories: users who installed the app on their own are organic
        # and the users who installed it through the campaign, advertisements
        # or for rewards, are non-organic users.
        user_categories = ['organic', 'non-organic']

        # create events list
        events = ['install', 'signup', 'click_other_content',
                  'create_content', 'create_team', 'create_colab_content',
                  'post_content', 'post_colab_content', 'delete_content']

        # create a date range
        dates = pd.date_range(start='2019-01-01', end='2020-12-31', freq='H')

        # populate the generated values to the empty dataset
        df.user_id = df.user_id.apply(lambda user: random.choice(user_ids))

        # assign user category to each user randomly
        user_cat_dict = {user_id: random.choice(user_categories) for user_id in df.user_id.unique()}
        df.user_category = df.user_id.map(user_cat_dict)

        # populate event and time columns
        # by randomly applying values from events and dates lists
        df.event = df.event.apply(lambda event: random.choice(events))
        df.time = df.time.apply(lambda time: random.choice(dates))

        return df
