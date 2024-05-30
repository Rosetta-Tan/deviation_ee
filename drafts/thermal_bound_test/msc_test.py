import dynamite
from dynamite.operators import zero, identity, sigmax, sigmay, sigmaz
import numpy as np

dnm_int_t = np.int32

msc_dtype = np.dtype([('masks', dnm_int_t),
                      ('signs', dnm_int_t),
                      ('coeffs', np.complex128)])

def serialize(msc):
    '''
    Take an MSC representation and spin chain length and serialize it into a
    byte string.

    The format is
    `nterms int_size masks signs coefficients`
    where `nterms`, and `int_size` are utf-8 text, including newlines, and the others
    are each just a binary blob, one after the other. `int_size` is an integer representing
    the size of the int data type used (32 or 64 bits).

    Binary values are saved in big-endian format, to be compatible with PETSc defaults.

    Parameters
    ----------
    MSC : np.array
        The MSC representation

    Returns
    -------
    bytes
        A byte string containing the serialized operator.
    '''

    rtn = b''

    rtn += (str(msc.size)+'\n').encode('utf-8')
    rtn += (str(msc.dtype['masks'].itemsize*8)+'\n').encode('utf-8')

    int_t = msc.dtype[0].newbyteorder('B')
    cplx_t = np.dtype(np.complex128).newbyteorder('B')
    rtn += msc['masks'].astype(int_t, casting='equiv', copy=False).tobytes()
    rtn += msc['signs'].astype(int_t, casting='equiv', copy=False).tobytes()
    rtn += msc['coeffs'].astype(cplx_t, casting='equiv', copy=False).tobytes()

def deserialize(data):
    '''
    Reverse the serialize operation.

    Parameters
    ----------
    data : bytes
        The byte string containing the serialized data.

    Returns
    -------
    tuple(int, np.ndarray)
        A tuple of the form (L, MSC)
    '''

    start = 0
    stop = data.find(b'\n')
    msc_size = int(data[start:stop])

    start = stop + 1
    stop = data.find(b'\n', start)
    int_size = int(data[start:stop])
    if int_size == 32:
        int_t = np.int32
    elif int_size == 64:
        int_t = np.int64
    else:
        raise ValueError('Invalid int_size. Perhaps file is corrupt.')

    msc = np.ndarray(msc_size, dtype=msc_dtype)

    mv = memoryview(data)
    start = stop + 1
    int_msc_bytes = msc_size * int_size // 8

    masks = np.frombuffer(mv[start:start+int_msc_bytes],
                          dtype=np.dtype(int_t).newbyteorder('B'))

    # operator was saved using 64 bit dynamite, but loaded using 32
    if int_size == 64 and msc_dtype['masks'].itemsize == 4:
        if np.count_nonzero(masks >> 31):
            raise ValueError('dynamite must be built with 64-bit indices'
                             'to load operator on more than 31 spins.')

    msc['masks'] = masks
    start += int_msc_bytes
    msc['signs'] = np.frombuffer(mv[start:start+int_msc_bytes],
                                 dtype=np.dtype(int_t).newbyteorder('B'))
    start += int_msc_bytes
    msc['coeffs'] = np.frombuffer(mv[start:],
                                  dtype=np.dtype(np.complex128).newbyteorder('B'))

    return msc

if __name__ == '__main__':
    filename = 'test.msc'
    from dynamite import config
    from itertools import combinations
    config.L = 6
    op = zero()
    itertator = list(combinations(range(6), 4))
    for i in range(6):
        op += sigmax(i)*sigmax((i+1)%6)
    print(op._string_rep)
        