import numpy as np
import pandas as pd

SPEC_FOLDER = "full_data"
EXCITE_WAVE_LENGTH = 350
PREDICT_IONS = ["Cr"]


def load_map(n_file, spec_file):
    """Loads 2D fluorescence map and returns as df.
    """
    df = pd.read_csv(spec_file + f'/{n_file}' + '.csv', index_col=0).iloc[1:, :-1].T

    # function for renaming labels and indexes in map
    vec_renaming = lambda vec_list: [float(name.split()[0]) for name in vec_list]

    df.columns = vec_renaming(df.columns)
    df.index = vec_renaming(df.index)

    return df


def get_wavelength(spec_file, n_file=1):
    """Loads array of wavelength.
    """
    df = pd.read_csv(spec_file + f'/{n_file}' + '.csv', index_col=0).iloc[1:, :-1].T

    # function for renaming labels and indexes in map
    vec_renaming = lambda vec_list: [float(name.split()[0]) for name in vec_list]

    l_wavelenth = np.array(vec_renaming(df.columns))

    return l_wavelenth


def get_x(wave_length, spec_file):
    """Forms and returns data of fluorescence - x data.
    """
    l_X = []
    for n_file in range(1, 1001):
        fl_map = load_map(n_file, spec_file)
        l_X.append(fl_map.loc[wave_length])
    X = pd.DataFrame(l_X, index = np.arange(1,1001), dtype=np.float64)

    return X


def get_y(l_ions, spec_file):
    """Forms and returns data of ions concentration - y data.
    """
    df = pd.read_excel(spec_file + f'/Y_answers.xlsx', index_col=0)

    y = df[l_ions].to_numpy(dtype=np.float64).ravel()

    return y
