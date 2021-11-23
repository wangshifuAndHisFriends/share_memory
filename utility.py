import pandas as pd 
import os
import numpy as np


def index2int64(data):
    # make index to int64
    index = data.index.to_frame(index=False)
    index.iloc[:, 0] = index.iloc[:, 0].view('int64')
    index.iloc[:, 1] = index.iloc[:, 1].str[:6].astype('int64')
    data.index = pd.MultiIndex.from_frame(index)
    return data


def load_and_write_candles(
    data_path='/dataDisk2/alphaData/eastmoney_data/extra_data/dailybar/ohlcv_stk.pkl',
    start_date = '2016-01-01',
    end_date = '2017-12-29',
    write_dir = './data/'
    ):    
    # load candles and save it in current directory
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    data = pd.read_pickle(data_path)
    data = data.loc[pd.IndexSlice[start_date:end_date, :], :]
    
    # make index of data into int64
    data = index2int64(data)
    
    # swaplevel and sort_index
    data = data.swaplevel().sort_index()

    # save data into write dir
    if not os.path.isdir(write_dir):
        os.mkdir(write_dir)
    
    data.to_parquet(os.path.join(write_dir, 'data.parquet'), engine='pyarrow')

    return data


def get_indptr(data):
    gsize = data.groupby(level=0).size()
    indptr = np.zeros(shape=len(gsize)+1,dtype='int64')
    indptr[1:] = np.cumsum(gsize.values)
    return indptr


def get_start_and_end_position(data):
    indptr = get_indptr(data)
    positions = np.empty(shape=(len(indptr)-1, 2), dtype='int64')
    positions[:,0] = indptr[:-1]
    positions[:,1] = indptr[1:]

    return positions
    

if __name__ == '__main__':
    data = pd.read_parquet('./data/data.parquet')
    print(get_indptr(data))
    print(get_start_and_end_position(data))
