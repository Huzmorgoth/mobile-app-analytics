import pandas as pd
import numpy as np
import streamlit as st

class UserAcquisition:
    def __init__(self):
        pass

    def user_acquisition(self, df, event):
        """
         The function  for identifying the acquisition time for each user
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError('"dataset" needs to be a pandas dataframe')

        if not isinstance(event, str):
            raise TypeError('"event" needs to be a string')

        if event not in df['event'].unique():
            raise ValueError('"event" have to be a valid event present in the dataset')

        # get the acquisition time for each user
        acquisition = df[df['event'] == event].sort_values(
            'time').drop_duplicates(
            subset='user_id', keep='first')[['user_id', 'time']]

        # convert df to a dictionary
        acquisition = dict(zip(acquisition['user_id'], acquisition['time']))

        return acquisition

    def cohort_events_acquisition(self, df, event, period='week', month_format='period'):
        """
        The function to add "cohort", "event_period", "user_active" and "user_returns" columns.
        "cohort" is the weekly/monthly period that the user generated a successful plan (user acquired).
        "event_period" is the cohort that any event belongs in.
        "user_active" is True if the event took place at or after the user's acquisition time, False otherwise.
        "user_returns" is True if the event took place during a period subsequent to the acquisition cohort,
        False otherwise.
        """
        assert period in ['day', 'week', 'month'], '"period" should be either "day", "week" or "month"'

        if month_format:
            assert month_format in ['period', 'datetime'], '"month_format" should be either "period" or "datetime"'

        # user acquisition dictionary of unqiue acquired users
        acquisition = self.user_acquisition(df, event)
        users = acquisition.keys()

        # filter dataframe for only acquired users
        events = df[df['user_id'].isin(users)].copy()

        # get acquisition time for each user and create a "cohort" column
        events['acquisition_time'] = events['user_id'].map(acquisition)

        # create the "cohort" and "event_period" columns, based on the period defined
        if period == 'day':
            events['cohort'] = events['acquisition_time'].dt.date
            events['event_period'] = events['time'].dt.date

        elif period == 'week':
            events['cohort'] = (events['acquisition_time']
                                - events['acquisition_time'].dt.weekday.astype(
                        'timedelta64[D]')).astype('datetime64[D]')

            events['event_period'] = (events['time']
                                      - events['time'].dt.weekday.astype(
                        'timedelta64[D]')).astype('datetime64[D]')

        else:
            # if monthly period, choose between pandas period type and datetime type
            # period type has a nice monthly format and is fine for aggregations
            # datetime would show up as first/last day of the month (yyyy-mm-dd)
            if month_format == 'period':
                events['cohort'] = events['acquisition_time'].dt.to_period('M')
                events['event_period'] = events['time'].dt.to_period('M')

            elif month_format == 'datetime':
                events['cohort'] = events['acquisition_time'].dt.date.astype('datetime64[M]')
                events['event_period'] = events['time'].dt.date.astype('datetime64[M]')

        # indicate if the user did any action at or after his/her acquisition time
        # if you do not want to count same-day activity replace following line with:
        # events['user_active'] = (events['time'].dt.date > events['acquisition_time'].dt.date)
        events['user_active'] = (events['time'] >= events['acquisition_time'])
        events['plan_user_active'] = (events['time'] > events['acquisition_time'])

        # indicate if the user returned in any period subsequent to his/her acquisition cohort
        events['user_returns'] = (events['event_period'] > events['cohort'])

        return events

    def users_per_period(self, df, event, user_category, period='week', month_format='period'):
        """
        The function to group new users into period cohorts.
        The first time a user generates a plan is treated as the acquisition time.
        """
        if user_category:
            assert hasattr(df, user_category), '"user_category" needs to be a column in the df dataset'

        # calculate the cohort for each user and period for each event
        events = self.cohort_events_acquisition(df, event, period=period, month_format=month_format)

        # will be used to rename the period column of each groupby result
        period_name = {'week': 'week_starting',
                       'month': "month"}

        # calculate size of each users cohort
        new_users = events.drop_duplicates(subset=['user_id', 'cohort']) \
            .groupby(['cohort']).size() \
            .reset_index() \
            .rename({0: 'new_users', 'cohort': period_name[period]}, axis=1) \
            .set_index(period_name[period])

        # break down new users into Organic/Non-organic
        if user_category:
            category = events[events['event'] == event] \
                .groupby(['cohort', 'user_category'])['user_id'].nunique() \
                .reset_index() \
                .rename({'user_id': 'new_users', 'cohort': period_name[period]}, axis=1) \
                .set_index(period_name[period])

            category = category.pivot(columns='user_category', values='new_users')[['organic', 'non-organic']] \
                .rename({'organic': 'new_organic_users', 'non-organic': 'new_non_organic_users'}, axis=1)

        # calculate number of active users per period
        active_users = events[events['user_active']] \
            .groupby(['event_period'])['user_id'].nunique() \
            .reset_index() \
            .rename({'user_id': 'active_users', 'event_period': period_name[period]}, axis=1) \
            .set_index(period_name[period])

        # calculate number of returning users per period
        returning_users = events[events['user_returns']] \
            .groupby(['event_period'])['user_id'].nunique() \
            .reset_index() \
            .rename({'user_id': 'returning_users', 'event_period': period_name[period]}, axis=1) \
            .set_index(period_name[period])

        # merge into a single dataframe
        if user_category:
            ds = new_users.join([category, active_users, returning_users], how='outer', sort=False).astype(
                'Int64').copy()
        else:
            ds = new_users.join([active_users, returning_users], how='outer', sort=False).astype('Int64').copy()
        ds.fillna(0, inplace=True)

        # calculate period-on-period growth
        ds['w/w_growth'] = ds['new_users'].pct_change().apply(lambda x: "{0:.2f}%".format(x * 100))
        ds['new/return_ratio'] = (ds['new_users'] / ds['returning_users']) \
            .fillna(0) \
            .replace(np.inf, np.nan) \
            .apply(lambda x: "{0:.1f}".format(x))

        return ds

    def create_funnel_df(self, df, steps, from_date=None, to_date=None, step_interval=pd.to_timedelta('0')):
        """
        Function used to create a dataframe that can be passed to functions for generating funnel plots
        """
        assert isinstance(steps, list), '"steps" should be a list of strings'

        if step_interval != 0:
            assert isinstance(step_interval, pd.Timedelta), \
                '"step_interval" should be a valid pd.Timedelta object. For more info visit:' \
                'https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.Timedelta.html'

        # filter df for only events in the steps list
        df = df[['user_id', 'event', 'time']]
        df = df[df['event'].isin(steps)]

        values = []
        # create a dict to hold the filtered dataframe of each step
        dfs = {}
        # for each step, create a df and filter only for that step
        for i, step in enumerate(steps):
            if i == 0:

                # filter for users that did the 1st event and find the minimum time
                dfs[step] = df[df['event'] == step] \
                    .sort_values(['user_id', 'time'], ascending=True) \
                    .drop_duplicates(subset=['user_id', 'event'], keep='first')

                # filter df of 1st step according to dates
                # this will allow the 1st step to have started during the defined period
                # but subsequent steps are allowed to occur at a later date so that the funnel
                # is not penalised unfairly
                if from_date:
                    dfs[step] = dfs[step][(dfs[step]['time'] >= from_date)]

                if to_date:
                    dfs[step] = dfs[step][(dfs[step]['time'] <= to_date)]

            else:
                # filter for specific event
                dfs[step] = df[df['event'] == step]

                # left join with previous step
                # this ensures only rows for which the distinct_ids appear in the previous step
                merged = pd.merge(dfs[steps[i - 1]], dfs[step], on='user_id', how='left')

                # keep only events that happened after previous step and sort by time
                merged = merged[merged['time_y'] >=
                                (merged['time_x'] + step_interval)].sort_values('time_y', ascending=True)

                # take the minimum time of the valid ones for each user
                merged = merged.drop_duplicates(subset=['user_id', 'event_x', 'event_y'], keep='first')

                # keep only the necessary columns and rename them to match the original structure
                merged = merged[['user_id', 'event_y', 'time_y']].rename({'event_y': 'event',
                                                                          'time_y': 'time'}, axis=1)

                # include the df in the df dictionary so that it can be joined to the next step's df
                dfs[step] = merged

                # append number of users to the "values" list
            values.append(len(dfs[step]))

        # create dataframe
        funnel_df = pd.DataFrame({'step': steps, 'val': values})

        return funnel_df

    def group_funnel_dfs(self, df, steps, column):
        """
        Function used to create a dict of funnel dataframes used to generate a stacked funnel plot
        """
        assert isinstance(df, pd.DataFrame), '"df" should be a pandas dataframe'
        assert isinstance(column, str), '"col" should be a string'
        assert hasattr(df, column), '"column" should be a column in "df"'

        dict_ = {}
        # get the distinct_ids for each property that we are grouping by
        ids = dict(df.groupby([column])['user_id'].apply(set))

        for i in df[column].dropna().unique():
            ids_list = ids[i]
            df_sub = df[df['user_id'].isin(ids_list)]
            if len(df_sub[df_sub['event'] == steps[0]]) > 0:
                dict_[i] = self.create_funnel_df(df_sub, steps)

        return dict_

    def cohort_period(self, df):
        """
        Creates a `cohort_period` column, which is the Nth period based on the user's acquisition date.
        """
        df['cohort_period'] = np.arange(len(df))
        return df

    def mask_retention_table(self, dim):
        """
        Function used to fill NaN values with 0 above the diagonal line of the retention table and force
        the rest to be NaN.
        """
        # create an array of the same shape as the df and assign all elements =True
        mask = np.full(dim, True)

        # assign False where period for each row would no exist
        # i.e. if we have 10 weeks, the 1st week would have data for the next 9 weeks but the 2nd week would
        # only have data for the next 8 weeks, etc...
        for row in range(mask.shape[0]):
            mask[row, :mask.shape[0] - row] = False

        return mask

    def retention_table(self, df, period='week', month_format='period', event_filter=None):
        """
        Function used to generate retention stats split into weekly cohorts
        """
        assert period in ['week', 'month'], '"period" should be either "week" or "month"'
        if event_filter:
            assert event_filter in df['event'].unique(), '"event_filter" should be a valid event present in "df"'

        # filter out internal testers and get acquisition time of each user
        # create an event_period column for each event
        # determine if each event happened at least 1 day after the user acquisition
        events = self.cohort_events_acquisition(df, event_filter, period=period, month_format=month_format)

        # calculate size of each users cohort
        cohort_sizes = events.drop_duplicates(subset=['user_id', 'cohort']).cohort.value_counts() \
            .to_frame() \
            .rename({'cohort': 'size'}, axis=1)
        cohort_sizes.index.rename('cohort', inplace=True)

        # filter only for events after acquisition date
        events = events[events['plan_user_active']]
        # filter for event of interest
        if event_filter:
            events = events[events['event'] == event_filter]

        grouped = events.groupby(['cohort', 'event_period'])

        # count the unique users per Group + Period
        cohorts = grouped.agg({'user_id': pd.Series.nunique})
        # reindex the "cohort" (and "event_period" columns) to avoid empty weeks causing misalignment
        # grab the minimum 'cohort' date and maximum 'event_period' date
        start, end = cohorts.index.get_level_values('cohort').min(), \
                     cohorts.index.get_level_values('event_period').max()

        # TODO: if more periods will be considered need to add more here
        if period == 'week':
            full_index = pd.date_range(start=start, end=end, freq='W-MON', name='cohort')
        elif period == 'month':
            if month_format == 'period':
                full_index = pd.date_range(start=start.to_timestamp(), end=end.to_timestamp(), freq='MS')
            elif month_format == 'datetime':
                full_index = pd.date_range(start=start, end=end, freq='MS')

        cohorts.reset_index(inplace=True)
        #print(cohorts.dtype())
        # create all possible combinations of possible date periods
        # date_period needs to be equal to or greater than cohort
        possible_dates = []
        for i in range(len(full_index)):
            for j in range(len(full_index)):
                if i <= j:
                    possible_dates.append((i, j))

        # fill in missing combinations of cohort and event_period
        # add a new row in the df for a combination of possible dates with value=0
        for combo in possible_dates:
            if len(cohorts[(cohorts['cohort'] == full_index[combo[0]]) &
                           (cohorts['event_period'] == full_index[combo[1]])]) < 1:

                cohorts = cohorts.append({'cohort': full_index[combo[0]],
                                          'event_period': full_index[combo[1]],
                                          'user_id': 0},
                                         ignore_index=True).sort_values(['cohort', 'event_period'])
        cohorts = cohorts.set_index(['cohort', 'event_period'])

        # create 'cohort_period' column
        cohorts = cohorts.astype(str).groupby(level=0).apply(self.cohort_period)

        # reindex the DataFrame
        cohorts.reset_index(inplace=True)
        cohorts.set_index(['cohort', 'cohort_period'], inplace=True)

        # create user_retention df
        user_retention = cohorts['user_id'].unstack(0).T
        # include the cohort size as a secondary index
        user_retention = user_retention.join(cohort_sizes, how='outer', sort=False)
        user_retention['size'].fillna(0, inplace=True)
        user_retention['size'] = user_retention['size'].astype(int)
        user_retention.set_index('size', append=True, inplace=True)
        user_retention.columns.name = 'cohort_period'
        # convert float to Int64
        user_retention = user_retention[user_retention.columns].replace('NaN', np.NaN) \
            .astype('float64')
        # .astype('Int64')

        # convert to percentages
        user_retention_pct = user_retention.divide(user_retention.index.get_level_values('size'), axis='rows')

        # fill NaNs with 0 where a value is possible to exist
        mask_array = self.mask_retention_table(user_retention.shape)
        user_retention = user_retention.fillna(0).mask(mask_array)
        user_retention_pct = user_retention_pct.fillna(0).mask(mask_array)
        return user_retention, user_retention_pct

    def filter_starting_step(self, x, starting_step, n_steps):
        """
        Function used to return the first n_steps for each user starting from the "starting_step".
        The function will be used to generate the event sequence journey for each user.
        """
        assert isinstance(x, (list, pd.Series)), '"x" should be a python list or pandas series containing event names'
        assert isinstance(starting_step, str), '"starting_step" should be a string resembling an event name'
        assert isinstance(n_steps, int), '"n_steps" should be an integer'

        starting_step_index = x.index(starting_step)

        return x[starting_step_index: starting_step_index + n_steps]

    def user_journey(self, df, starting_step, n_steps=3, events_per_step=5):
        """
        Function used to map out the journey for each user starting from the defined "starting_step" and count
        how many identical journeys exist across users.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError('"df" should be a dataframe')

        assert isinstance(events_per_step, int), '"events_per_step" should be an integer'
        if events_per_step < 1:
            raise ValueError('"events_per_step" should be equal or greater than 1')

        # sort df by time
        df = df.sort_values(['user_id', 'time'])
        # find the users that have performed the starting_step
        valid_ids = df[df['event'] == starting_step]['user_id'].unique()

        # plan out the journey per user, with each step in a separate column
        flow = df[df.user_id.isin(valid_ids)] \
            .groupby('user_id') \
            .event.agg(list) \
            .to_frame()['event'] \
            .apply(lambda x: self.filter_starting_step(x, starting_step=starting_step, n_steps=n_steps)) \
            .to_frame() \
            ['event'].apply(pd.Series)

        # fill NaNs with "End" to denote no further step by user; this will be filtered out later
        flow = flow.fillna('End')

        # add the step number as prefix to each step
        for i, col in enumerate(flow.columns):
            flow[col] = '{}: '.format(i + 1) + flow[col].astype(str)

        # replace events not in the top "events_per_step" most frequent list with the name "Other"
        # this is done to avoid having too many nodes in the sankey diagram
        for col in flow.columns:
            all_events = flow[col].value_counts().index.tolist()
            all_events = [e for e in all_events if e != (str(col + 1) + ': End')]
            top_events = all_events[:events_per_step]
            to_replace = list(set(all_events) - set(top_events))
            flow[col].replace(to_replace, [str(col + 1) + ': Other'] * len(to_replace), inplace=True)

        # count the number of identical journeys up the max step defined
        flow = flow.groupby(list(range(n_steps))) \
            .size() \
            .to_frame() \
            .rename({0: 'count'}, axis=1) \
            .reset_index()

        return flow

    def sankey_df(self, df, starting_step, n_steps=3, events_per_step=5):
        """
        Function used to generate the dataframe needed to be passed to the sankey generation function.
        "source" and "target" column pairs denote links that will be shown in the sankey diagram.
        """
        # generate the user user flow dataframe
        flow = self.user_journey(df, starting_step, n_steps, events_per_step)

        # create the nodes labels list
        label_list = []
        cat_cols = flow.columns[:-1].values.tolist()
        for cat_col in cat_cols:
            label_list_temp = list(set(flow[cat_col].values))
            label_list = label_list + label_list_temp

        # create a list of colours for the nodes
        # assign 'blue' to any node and 'grey' to "Other" nodes
        colors_list = ['blue' if i.find('Other') < 0 else 'grey' for i in label_list]

        # transform flow df into a source-target pair
        for i in range(len(cat_cols) - 1):
            if i == 0:
                source_target_df = flow[[cat_cols[i], cat_cols[i + 1], 'count']]
                source_target_df.columns = ['source', 'target', 'count']
            else:
                temp_df = flow[[cat_cols[i], cat_cols[i + 1], 'count']]
                temp_df.columns = ['source', 'target', 'count']
                source_target_df = pd.concat([source_target_df, temp_df])
            source_target_df = source_target_df.groupby(['source', 'target']).agg({'count': 'sum'}).reset_index()

        # add index for source-target pair
        source_target_df['source_id'] = source_target_df['source'].apply(lambda x: label_list.index(x))
        source_target_df['target_id'] = source_target_df['target'].apply(lambda x: label_list.index(x))

        # filter out the end step
        source_target_df = source_target_df[(~source_target_df['source'].str.contains('End')) &
                                            (~source_target_df['target'].str.contains('End'))]

        return label_list, colors_list, source_target_df
