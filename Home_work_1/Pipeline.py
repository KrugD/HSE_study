import dill
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def transformtorque(df):
    """Преобразует колонку крутящего момента в отдельные колонки крутящего момента и оборота."""
    df_new = df.copy()

    def extract_numeric(str_num):
        """Извлекает числовые значения из строки крутящего момента."""
        if pd.isna(str_num) or str_num == 'NaN':
            return pd.Series([np.nan, np.nan])

        # Извлекаем момент и обороты
        parts = re.split('at|@', str_num)

        if len(parts) != 2:
            return pd.Series([np.nan, np.nan])

        Nm_part = parts[0].strip()
        rpm_part = parts[1].strip()

        # Обработка момента
        if 'kgm' in Nm_part.lower():
            if 'Nm' in (Nm_part.split('kgm')[0]):
                Nm_value = float(re.split('kgm|Nm|NM', Nm_part)[0])
            else:
                Nm_value = float(re.split('kgm|KGM', Nm_part)[0]) * 9.8
        else:
            Nm_value = float(re.split('Nm|nm|NM', Nm_part.replace('(', 'Nm'))[0].replace(',', ''))  # "4,500" -> "4500"

        # Обработка оборотов
        if '-' or '~' in rpm_part:  # интервал
            rpm_values = re.split('-|~', rpm_part)
            if len(rpm_values) == 2:
                rpm_value = (float(rpm_values[0].replace('rpm', '').replace('RPM', '').replace('+/', '').replace(',',
                                                                                                                 '').strip()) + float(
                    rpm_values[1].replace('rpm', '').replace('RPM', '').replace('+/', '').replace(',', '').strip())) / 2
            else:
                rpm_value = float(rpm_part.replace('rpm', '').replace('RPM', '').strip().replace(',', ''))
        else:
            rpm_value = float(rpm_part.replace('rpm', '').strip().replace(',', ''))

        return pd.Series([Nm_value, rpm_value])

    df_new[['torque_Nm', 'torque_rpm']] = df_new['torque'].apply(extract_numeric)
    return df_new

def data_colum_transform(df):
    """Преобразует различные столбцы в правильные числовые форматы."""
    df_new = df.copy()
    def extract_float(value):
        """Извлекает число с плавающей запятой из заданного значения."""
        try:
            return float(str(value).split()[0])
        except (ValueError, IndexError):
            return float('nan')

    # Преобразование пробега, двигателя и максимальной мощности в числовые значения
    df_new['mileage_kmpl'] = df_new['mileage'].apply(lambda x: float(str(x).split(' ')[0]))
    df_new['engine_CC'] = df_new['engine'].apply(lambda x: float(str(x).split(' ')[0]))
    df_new['max_power_bhp'] = df_new['max_power'].apply(extract_float)
    df_new['Car_Brand'] = df_new['name'].apply(lambda x: x.split(' ')[0])
    df_new.drop(['torque', 'mileage', 'engine', 'max_power', 'name'], axis= 1, inplace= True)
    return df_new

def data_type_transform(df):
    """Преобразует определенные столбцы в тип object."""
    df_new = df.copy()
    df_new['seats'] = df_new['seats'].astype(object)
    return df_new

def main():
    df_train = pd.read_csv(
        'https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
    df_train.drop('selling_price', axis=1, inplace=True)

    numerical_features = make_column_selector(
        dtype_include=[int, float])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_features = make_column_selector(
        dtype_include=[object])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'))
    ])
    preprocessor_1 = Pipeline(steps=[
        ('transformtorque', FunctionTransformer(transformtorque)),
        ('data_colum_transform', FunctionTransformer(data_colum_transform)),
        ('data_type_transform', FunctionTransformer(data_type_transform)),
    ])
    preprocessor_2 = ColumnTransformer(transformers=[
        ('categorical', categorical_transformer, categorical_features),
        ('numerical', numerical_transformer, numerical_features)
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor_1', preprocessor_1),
        ('preprocessor_2', preprocessor_2),
    ])

    pipeline.fit_transform(df_train)

    # Сохранение обученного pipeline в файл
    data_pipeline = f'model/cars_pipeline.pkl'
    with open(data_pipeline, 'wb') as file:
        dill.dump({
            'pipeline': pipeline,
        }, file)

if __name__ == '__main__':
    main()