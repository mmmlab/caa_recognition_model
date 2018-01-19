import pylab as pl
import pyfftw


NR_THREADS = 1;
INPUT_ARRAY = None;
OUTPUT_ARRAY = None;
FFT  = None;
IFFT = None;

pyfftw.interfaces.cache.enable(); # enable fftw_wisdom caching

def fftw_setup(arr,nr_threads=NR_THREADS):
    global INPUT_ARRAY;
    global OUTPUT_ARRAY;
    global FFT;
    global IFFT;
    INPUT_ARRAY = pyfftw.n_byte_align_empty(pl.shape(arr), 16, 'complex128');
#    INPUT_ARRAY = pyfftw.n_byte_align_empty(shape(arr), 16, 'float64');
    OUTPUT_ARRAY = pyfftw.n_byte_align_empty(pl.shape(arr), 16, 'complex128');
    INPUT_ARRAY[:] = pl.zeros(pl.shape(arr));
    OUTPUT_ARRAY[:] = pl.zeros(pl.shape(arr))+0j;
    FFT = pyfftw.FFTW(INPUT_ARRAY, OUTPUT_ARRAY,threads=nr_threads);
    IFFT = pyfftw.FFTW(OUTPUT_ARRAY,INPUT_ARRAY,direction='FFTW_BACKWARD',threads=nr_threads);
    
    
def fft(arr):
    INPUT_ARRAY[:] = arr+0j;
#    INPUT_ARRAY[:] = arr;
    return pl.array(FFT());

def ifft(arr):
    OUTPUT_ARRAY[:]=arr;
    return IFFT().real;
#    return IFFT();
