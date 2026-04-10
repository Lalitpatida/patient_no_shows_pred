import pandas as pd
import numpy as np

def feature_engineering(df):
    df = df.copy()

    # -------------------- Date Features --------------------
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

    df = df.sort_values(by=['PatientId', 'ScheduledDay'])

    # Lead Time (most important feature)
    df['LeadTime'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
    df['LeadTime_bin'] = pd.cut(df['LeadTime'], bins=[-1, 0, 7, 21, 100], 
                                labels=['same_day', '1week', '2-3weeks', 'long'])

    # -------------------- Patient History --------------------
    df['prev_no_show_count'] = df.groupby('PatientId')['No_show'] \
        .transform(lambda x: x.shift().fillna(0).cumsum())

    df['prev_show_count'] = df.groupby('PatientId')['No_show'] \
        .transform(lambda x: (1 - x).shift().fillna(0).cumsum())

    df['prev_total_visits'] = df['prev_no_show_count'] + df['prev_show_count']
    df['no_show_rate'] = df['prev_no_show_count'] / df['prev_total_visits'].replace(0, np.nan)
    df['no_show_rate'] = df['no_show_rate'].fillna(0).round(3)

    # Recency-weighted no-show rate (more recent behavior matters more)
    df['recency_weight'] = 1 / (df.groupby('PatientId').cumcount() + 1)

    cum_num = (df['No_show'] * df['recency_weight']).groupby(df['PatientId']).transform('cumsum')
    cum_denom = df['recency_weight'].groupby(df['PatientId']).transform('cumsum')

    # Corrected: subtract current row's contribution
    df['weighted_no_show_rate'] = (cum_num - df['No_show'] * df['recency_weight']) / \
                                  (cum_denom - df['recency_weight'] + 1e-8)
    df['weighted_no_show_rate'] = df['weighted_no_show_rate'].fillna(0).round(3)

    # -------------------- Behavioral & Demographic Features --------------------
    df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})
    df['is_weekend'] = df['AppointmentDayOfWeek'].isin(['Saturday', 'Sunday']).astype(int)
    df['is_old'] = (df['Age'] > 60).astype(int)
    df['long_wait'] = (df['waitDays'] > 7).astype(int)
    df['comorbidity_count'] = df[['Hypertension', 'Diabetes', 'Handicap']].sum(axis=1)

    df['sms_effectiveness'] = df['SMS_received'] * df['no_show_rate']
    df['sms_effectiveness'] = df['sms_effectiveness'].fillna(0).round(3)

    #df['neighbourhood_no_show_rate'] = df.groupby('Neighbourhood')['No_show'].transform('mean')

    # One-hot for categorical (LeadTime_bin is useful)
    df = pd.get_dummies(df, columns=['LeadTime_bin'], drop_first=True)

    # Drop columns
    drop_cols = ['ScheduledDay', 'AppointmentDay', 'AppointmentDayOfWeek', 
                 'Neighbourhood', 'AppointmentID', 'Alcoholism']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    df.to_csv("data/f_engineered.csv", index=False)
    print("Feature engineering completed. Shape:", df.shape)
    return df
