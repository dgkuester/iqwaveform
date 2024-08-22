import cupy as cp

# context manager for evaluating abs^2 at the output of the fft
apply_abs2_in_fft = cp.fft.config.set_cufft_callbacks(
    cb_store=r"""
__device__ void abs2(
    void *dataOut, 
    size_t offset,
    cufftComplex element, 
    void *callerInfo, 
    void *sharedPtr
)
{
    ((cufftComplex *) dataOut)[offset].x = element.x*element.x + element.y*element.y;
}

__device__ cufftCallbackStoreC d_storeCallbackPtr = abs2;
"""
)


def build():
    """pre-build CUDA kernels"""
    with apply_abs2_in_fft:
        pass
