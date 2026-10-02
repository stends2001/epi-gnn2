"""
Three types of transformation functions:
1. Parameter - computing functions
2. Transformation - application functions
3. Reverse transformation - application functions
"""


import pandas as pd
import numpy as np

from ...columnregistration import (
    LogParams, ZScoreParams, MinMaxParams
)

# Three types of Functions here:

# ======== 1. PARAMETERS - COMPUTING FUNCTIONS ========= #
def compute_zscore_params(train_df: pd.DataFrame, column: str) -> ZScoreParams:
    """
    Compute (and return) the parameters to perform a Zscore transformation with
    based on training (!) data
    
    Parameters
    ----------
    train_df: pd.DataFrame
        dataframe on which to zscore
    column: str
        name of the column to perform zscore on

    Returns
    -------
    ZScoreParams
    """
    mean = float(train_df[column].mean())
    std  = float(train_df[column].std())
    return ZScoreParams(mean=mean, std=std)

def compute_minmax_params(train_df: pd.DataFrame, column: str) -> MinMaxParams:
    """
    Compute (and return) the parameters to perform a minmax transformation with
    based on training (!) data
    
    Parameters
    ----------
    train_df: pd.DataFrame
        dataframe on which to zscore
    column: str
        name of the column to perform zscore on

    Returns
    -------
    MinMaxParams
    """    
    return MinMaxParams(
        min=float(train_df[column].min()),
        max=float(train_df[column].max())
    )

# ======== 2. TRANSFORMATION / NORMALIZATION - APPLYING FUNCTIONS ========= #
def apply_log(df: pd.DataFrame, column: str, params: LogParams) -> pd.DataFrame:
    """ 
    Apply logarithm as dictated by parameters on a df's column

    Parameters
    ----------
    df: pd.DataFrame
        dataframe in which the column should be log-transformed
    column: str
        name of the column to be log-transformed
    params: LogParams
        The parameters that dictate the log-transform

    Returns
    -------
    pd.DataFrame with log-transformed column
    """
    out         = df.copy()
    out[column] = np.log(df[column] + params.shift)
    return out

def apply_zscore(df: pd.DataFrame, column: str, params: ZScoreParams) -> pd.DataFrame:
    """ 
    Apply zscore as dictated by parameters on a df's column

    Parameters
    ----------
    df: pd.DataFrame
        dataframe in which the column should be zscore-normalized
    column: str
        name of the column to be zscore-normalized
    params: ZScoreParams
        The parameters that dictate the zscore-normalized

    Returns
    -------
    pd.DataFrame with zscore-normalized column
    """    
    out         = df.copy()
    out[column] = 0.0 if params.std == 0 else (df[column] - params.mean) / params.std
    return out

def apply_minmax(df: pd.DataFrame, column: str, params: MinMaxParams) -> pd.DataFrame:
    """ 
    Apply minmax as dictated by parameters on a df's column

    Parameters
    ----------
    df: pd.DataFrame
        dataframe in which the column should be minmax-normalized
    column: str
        name of the column to be minmax-normalized
    params: MinMaxParams
        The parameters that dictate the minmax-normalized

    Returns
    -------
    pd.DataFrame with minmax-normalized column
    """       
    out = df.copy()
    rng = params.max - params.min
    out[column] = 0.0 if rng == 0 else (df[column] - params.min) / rng
    return out

# ======== 3. TRANSFORMATION / NORMALIZATION - REVERSE APPLYING FUNCTIONS ========= #
def reverse_log(df: pd.DataFrame, column: str, params: LogParams):
    """ 
    Reverse log-transform dictated in params on specified column in df

    Parameters
    ----------
    df: pd.DataFrame
        dataframe in which the column should be reverse-log-transformed
    column: str
        name of the column to be reverse-log-transformed
    params: LogParams
        The parameters that dictate the reverse-log-transformed

    Returns
    -------
    pd.DataFrame with reverse-log-transformed column
    """        
    out         = df.copy()
    out[column] = np.exp(df[column]) - params.shift
    return out

def reverse_zscore(df: pd.DataFrame, column: str, params: ZScoreParams) -> pd.DataFrame:
    """ 
    Reverse zscore-normalization dictated in params on specified column in df

    Parameters
    ----------
    df: pd.DataFrame
        dataframe in which the column should be reverse-zscore-normalized
    column: str
        name of the column to be reverse-zscore-normalized
    params: ZScoreParams
        The parameters that dictate the reverse-zscore-normalized

    Returns
    -------
    pd.DataFrame with reverse-zscore-normalized column
    """            
    out = df.copy()
    out[column] = df[column] * params.std + params.mean
    return out

def reverse_minmax(df: pd.DataFrame, column: str, params: MinMaxParams) -> pd.DataFrame:
    """ 
    Reverse minmax-normalization dictated in params on specified column in df

    Parameters
    ----------
    df: pd.DataFrame
        dataframe in which the column should be reverse-minmax-normalized
    column: str
        name of the column to be reverse-minmax-normalized
    params: MinMaxParams
        The parameters that dictate the reverse-minmax-normalized

    Returns
    -------
    pd.DataFrame with reverse-minmax-normalized column
    """     
    out = df.copy()
    out[column] = df[column] * (params.max - params.min) + params.min
    return out