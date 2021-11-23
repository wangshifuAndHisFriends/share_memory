import pandas as pd
import numpy as np
import multiprocessing.shared_memory as shared_memory


class SharedArray:
    def __init__(self, arr: np.ndarray):
        """
        put np.ndarray into shared memory
        """
        self.dtype = arr.dtype
        self.num_byte = arr.nbytes
        self.shape = arr.shape
        self.order = None

        if arr.flags['C_CONTIGUOUS']:
            self.order = 'C'
        elif arr.flags['F_CONTIGUOUS']:
            self.order = 'F'
        else:
            raise ValueError('arr must be stored coninuously')
        
        # allocate memory
        self.shm = shared_memory.SharedMemory(create=True, size=self.num_byte)
        self.shm_name = self.shm.name 

        # create memory view from shared memory
        view = np.ndarray(shape=self.shape, dtype=self.dtype, buffer=self.shm.buf, order=self.order)
        view[:] = arr 
    
    def read(self):
        """
        read from shared memory
        """
        #shm = shared_memory.SharedMemory(name=self.shm_name)
        view = np.ndarray(shape=self.shape, dtype=self.dtype, buffer=self.shm.buf, order=self.order)
        view.flags['WRITEABLE'] = False 
        return view
    
    def close(self):
        """
        release data in shared memory
        """    
        self.shm.unlink()
    
        
class OHLCV:
    def __init__(self, 
        ohlcv: np.ndarray,
        index: np.ndarray,
        symbol_indptr: np.ndarray,
        symbol_range: np.ndarray,
        columns: pd.Index
    ):
        """
        ohlcv: float64 array of shape [n, d] containing values of each candle bars. 
        index: int64 array of shape [n,2], index of ohlcv
        symbol_indptr: int64 index pointer array of shape (num_symbols+1,). The candle 
            bars of symbol[i] will be stored from symbol_indptr[i] to symbol_indptr[i+1]-1.
        symbol_range: int64 range array of shape (num_symbols, 2). Currently, the valid
            candle bars of symbol[i] are stores from (symbol_range[i, 0] to symbol_range[i, 1])
            with symbol_range[i,0] == symbol_indptr[i] and symbol_range[i,1] < symbol_indptr[i+1]
        """
        # store ohlcv into shared memory
        # critical region of shared variables
        self.ohlcv = SharedArray(ohlcv) 
        self.index = SharedArray(index)
        self.symbol_indptr = SharedArray(symbol_indptr)
        self.symbol_range = SharedArray(symbol_range)

        # private variable
        self.columns = columns

    def read(self):
        """
        return data stored in shared memory
        """
        ohlcv = self.ohlcv.read()
        index = self.index.read()
        symbol_indptr = self.symbol_indptr.read()
        symbol_range = self.symbol_range.read()

        data = {
            'ohlcv': ohlcv,
            'index': index,
            'symbol_indptr': symbol_indptr,
            'symbol_range': symbol_range
        }

        return data 


    def close(self):
        self.ohlcv.close()
        self.index.close()
        self.symbol_range.close()
        self.symbol_indptr.close()



if __name__ == '__main__':
    from utility import get_indptr, get_start_and_end_position
    from multiprocessing import Process
    import os
    import time 

    def thread(ohlcv: OHLCV):
        start = time.time()
        data = ohlcv.read()
        end = time.time()
        print('pid = %s, read data takes %e second' % (os.getpid(), end-start))
        for key in data:
            print("===========%s===========" % key)
            print(data[key])

    data = pd.read_parquet('./data/data.parquet')
    symbol_indptr = get_indptr(data)
    symbol_range = get_start_and_end_position(data)

    start = time.time()
    ohlcv = OHLCV(data.values, data.index.to_frame(index=False).values, symbol_indptr, symbol_range, data.columns)
    end = time.time()
    print('write data into share mem takes %e second' % (end-start))

    process = Process(target=thread, args=(ohlcv,))
    process.start()
    process.join()

    ohlcv.close()
