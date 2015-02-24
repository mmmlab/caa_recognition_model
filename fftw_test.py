from pylab import *
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
    INPUT_ARRAY = pyfftw.n_byte_align_empty(shape(arr), 16, 'complex128');
    OUTPUT_ARRAY = pyfftw.n_byte_align_empty(shape(arr), 16, 'complex128');
    INPUT_ARRAY[:] = zeros(shape(arr));
    OUTPUT_ARRAY[:] = zeros(shape(arr))+0j;
    FFT = pyfftw.FFTW(INPUT_ARRAY, OUTPUT_ARRAY,threads=2);
    IFFT = pyfftw.FFTW(OUTPUT_ARRAY,INPUT_ARRAY,direction='FFTW_BACKWARD',threads=nr_threads);
    
    
def fft(arr):
    INPUT_ARRAY[:] = arr+0j;
    return array(FFT());

def ifft(arr):
    OUTPUT_ARRAY[:]=arr;
    return IFFT().real;