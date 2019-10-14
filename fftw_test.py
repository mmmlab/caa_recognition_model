import pylab as pl
import pyfftw


NR_THREADS = 1
INPUT_ARRAY = None
OUTPUT_ARRAY = None
FFT  = None
IFFT = None
INPUT_ARRAY_2D = None
OUTPUT_ARRAY_2D = None
FFT2  = None
IFFT2 = None

pyfftw.interfaces.cache.enable() # enable fftw_wisdom caching

def fftw_setup(arr,nr_threads=NR_THREADS):
    global INPUT_ARRAY
    global OUTPUT_ARRAY
    global FFT
    global IFFT
    global INPUT_ARRAY_2D
    global OUTPUT_ARRAY_2D
    global FFT2
    global IFFT2
    arr_size = pl.shape(arr)[0]
    # define 1D FFTs
    INPUT_ARRAY = pyfftw.n_byte_align_empty(pl.shape(arr), 16, 'complex128')
    OUTPUT_ARRAY = pyfftw.n_byte_align_empty(pl.shape(arr), 16, 'complex128')
    INPUT_ARRAY[:] = pl.zeros(pl.shape(arr))
    OUTPUT_ARRAY[:] = pl.zeros(pl.shape(arr))+0j
    FFT = pyfftw.FFTW(INPUT_ARRAY, OUTPUT_ARRAY,threads=nr_threads)
    IFFT = pyfftw.FFTW(OUTPUT_ARRAY,INPUT_ARRAY,direction='FFTW_BACKWARD',threads=nr_threads)
    # define 2D FFTs (assuming square arrays)
    INPUT_ARRAY_2D = pyfftw.n_byte_align_empty((arr_size,arr_size), 16, 'complex128')
    OUTPUT_ARRAY_2D = pyfftw.n_byte_align_empty((arr_size,arr_size), 16, 'complex128')
    INPUT_ARRAY_2D[:,:] = pl.zeros((arr_size,arr_size))
    OUTPUT_ARRAY_2D[:,:] = pl.zeros((arr_size,arr_size))+0j
    # execute over both axes
    FFT2 = pyfftw.FFTW(INPUT_ARRAY_2D,OUTPUT_ARRAY_2D,axes=(0,1),threads=nr_threads)
    IFFT2 = pyfftw.FFTW(OUTPUT_ARRAY_2D,INPUT_ARRAY_2D,axes=(0,1),direction='FFTW_BACKWARD',threads=nr_threads)
    
    
def fft(arr):
    INPUT_ARRAY[:] = arr+0j
    return pl.array(FFT())

def ifft(arr):
    OUTPUT_ARRAY[:]=arr
    return IFFT().real

def fft2(arr):
    INPUT_ARRAY_2D[:,:] = arr+0j
    return pl.array(FFT2())

def ifft2(arr):
    OUTPUT_ARRAY_2D[:,:]=arr
    return IFFT2().real
