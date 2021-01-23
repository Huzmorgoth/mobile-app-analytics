from plotly.offline import iplot
import streamlit as st

from Resources.generateDFClass import GenerateDataset
from Resources.AcquisitionClass import UserAcquisition
from Resources.VisualizationClass import Visualisations

#generate dataset

ds_obj = GenerateDataset()

df = ds_obj.build()
st.title('Mobile App Statistics')
st.header('User Events Dataset:')
st.dataframe(df)

#Extracting acquisition time, user cohorts, events period
acq_obj = UserAcquisition()

st.sidebar.header('Explore Data:')
event = st.sidebar.radio('select the event for calculating acquisition,'
                         'activity, and for visualizations:', ('install', 'signup'))

period = st.sidebar.radio('select the period:', ('week','month'))

side_bar_but = st.sidebar.button('User Acquisition')

if side_bar_but:
    st.subheader('User Acquisition')
    cohort = acq_obj.cohort_events_acquisition(df=df,
                                               event=str(event),
                                               period=str(period),
                                               month_format='datetime')
    st.dataframe(cohort)

#Activity statistics per period
side_bar_activ = st.sidebar.button('Activity stats')
if side_bar_activ:
    st.subheader('Activity Stats')
    activity_stats = acq_obj.users_per_period(df=df,
                                              event=str(event),
                                              user_category='user_category',
                                              period=str(period),
                                              month_format='datetime')
    st.dataframe(activity_stats)

#Visualize user growth per period
st.sidebar.header('Visualizations:')
vis_obj = Visualisations()
side_bar_growth = st.sidebar.button('Generate Growth Graph')
if side_bar_growth:
    st.subheader('Users Growth Per Period')
    fig_1 = vis_obj.growth(df=df, event=str(event),
                           user_category='user_category',
                           period=str(period),
                           month_format='period')
    st.plotly_chart(fig_1)

side_bar_reten = st.sidebar.button('Generate Retention Graph')
if side_bar_reten:
    st.subheader('Users Retention Graph')
    retention_df = acq_obj.retention_table(df,
                                           period=str(period),
                                           event_filter=str(event),
                                           month_format='datetime')
    # st.text('ret_pct:', retention_df.index.get_level_values(1))
    fig_2 = vis_obj.retention_heatmap(retention_df[1].apply(lambda x: x * 100))
    st.pyplot(fig_2[1])

side_bar_funnel = st.sidebar.button('Generate Funnel Graph')
if side_bar_funnel:
    st.subheader('Funnel Graph')
    steps = ['install', 'signup', 'create_content', 'post_content']
    fig_3 = vis_obj.plot_stacked_funnel(df=df, steps=steps, col='user_category')
    st.plotly_chart(fig_3)

side_bar_journey = st.sidebar.button('Generate User Journey Graph')
if side_bar_journey:
    st.subheader('Users Journey')
    fig_4 = vis_obj.plot_user_flow(df=df, starting_step='install')
    st.plotly_chart(fig_4)
