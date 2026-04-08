import pandas as pd

def fix_age(df):
    invalid = df[df['Age'] < 0].shape[0]
    df = df[df['Age'] >= 0].copy()
    print(f'Removed {invalid} rows with invalid age values.')
    return df

def rename_columns(df):
    return df.rename(columns={
        'Hipertension': 'Hypertension',
        'No-show': 'No_show',
        'Handcap': 'Handicap'
    })

def fix_patient_id(df):
    df['PatientId'] = df['PatientId'].astype(int).astype(str)
    df['AppointmentID'] = df['AppointmentID'].astype(int).astype(str)
    return df

def fix_dates(df):
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'], errors='coerce')
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'], errors='coerce')

    df['waitDays'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
    df['ScheduledMonth'] = df['ScheduledDay'].dt.month
    df['AppointmentMonth'] = df['AppointmentDay'].dt.month
    df['AppointmentDayOfWeek'] = df['AppointmentDay'].dt.day_name()

    invalid_dates = df[df['waitDays'] < 0].shape[0]
    df = df[df['waitDays'] >= 0].copy()

    print(f'Removed {invalid_dates} invalid date rows.')
    return df

def fix_no_show(df):
    df['No_show'] = df['No_show'].map({'Yes': 1, 'No': 0})
    return df

def clean_pipeline(df):
    df = rename_columns(df)
    df = fix_patient_id(df)
    df = fix_age(df)
    df = fix_dates(df)
    df = fix_no_show(df)

    print(f'Cleaning complete: {df.shape}')
    return df




# df['is_weekend'] = df['AppointmentDayOfWeek'].isin(['Saturday','Sunday']).astype(int)
# df['is_old'] = (df['Age'] > 60).astype(int)
# df['long_wait'] = (df['waitDays'] > 7).astype(int)