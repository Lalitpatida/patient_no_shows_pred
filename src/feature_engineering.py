# import pandas as pd

# def feature_engineering(df):
#     df = df.copy()

#     df.drop(columns=['PatientId', 'AppointmentID', 'Alcoholism'], inplace=True)
#     df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})
    
#     day_map = {
#         'Monday': 0,
#         'Tuesday': 1,
#         'Wednesday': 2,
#         'Thursday': 3,
#         'Friday': 4,
#         'Saturday': 5,
#         'Sunday': 6
#     }

#     df['day_of_week'] = df['AppointmentDayOfWeek'].map(day_map)

#     df['is_weekend'] = df['AppointmentDayOfWeek'].isin(['Saturday', 'Sunday']).astype(int)
#     df['is_old'] = (df['Age'] > 60).astype(int)
#     df['long_wait'] = (df['waitDays'] > 7).astype(int)

#     df = pd.get_dummies(df, columns=['Neighbourhood'], drop_first=True)

#     df.drop(columns=['ScheduledDay', 'AppointmentDay', 'AppointmentDayOfWeek'], inplace=True)

#     df.to_csv("data/f_engineered.csv", index=False)

#     return df






import pandas as pd

def feature_engineering(df):
    df = df.copy()

    # -------------------- Drop unnecessary columns --------------------
    df.drop(columns=['PatientId', 'AppointmentID', 'Alcoholism'], inplace=True)

    # -------------------- Binary Encoding --------------------
    df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})

    # -------------------- Manual Weekday Encoding --------------------
    day_map = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }

    df['day_of_week'] = df['AppointmentDayOfWeek'].map(day_map)

    # Handle any unexpected values
    df['day_of_week'].fillna(-1, inplace=True)

    # -------------------- New Features --------------------
    df['is_weekend'] = df['AppointmentDayOfWeek'].isin(['Saturday', 'Sunday']).astype(int)
    df['is_old'] = (df['Age'] > 60).astype(int)
    df['long_wait'] = (df['waitDays'] > 7).astype(int)

    # -------------------- One-Hot Encoding --------------------
    df = pd.get_dummies(df, columns=['Neighbourhood'], drop_first=True)

    # -------------------- Drop unused columns --------------------
    df.drop(columns=['ScheduledDay', 'AppointmentDay', 'AppointmentDayOfWeek'], inplace=True)

    # -------------------- Save --------------------
    df.to_csv("data/f_engineered.csv", index=False)

    return df