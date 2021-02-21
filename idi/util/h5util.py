import numpy as _np
import h5py


def appenddata(file, key, data, chunks=None, compression='lzf'):
    data = _np.atleast_1d(_np.array(data))
    if _np.array(data).dtype.kind == 'U':
        data = data.astype(h5py.string_dtype(encoding='ascii'))
    if key not in file.keys():
        file.create_dataset(key, chunks=chunks or data.shape, compression=compression, shuffle=True if compression else None, data=data, maxshape=(None, *data.shape[1:]))
    else:
        file[key].resize((file[key].shape[0] + data.shape[0]), axis=0)
        file[key][-data.shape[0]:] = data


def overwritedata(file, key, data, chunks=None, compression='lzf'):
    data = _np.atleast_1d(_np.array(data))
    if _np.array(data).dtype.kind == 'U':
        data = data.astype(h5py.string_dtype(encoding='ascii'))
    if key in file.keys():
        del file[key]
    file.create_dataset(key, chunks=chunks or data.shape, compression=compression, shuffle=True if compression else None, data=data, maxshape=(None, *data.shape[1:]))

    
def shrink(file, key, n):
    file[key].resize(file[key].shape[0] - n, axis=0)


def list2array(li):
    maxlen = _np.max([len(e) for e in li])
    return _np.array([_np.pad(e, (0, maxlen - len(e)), 'constant') for e in li])


def copymasked(src, dst, mask):
    def func(name, obj):
        if isinstance(obj, h5py._hl.dataset.Dataset):
            dst[name] = obj[mask, ...]

    if isinstance(src, h5py._hl.dataset.Dataset):
        name = src.name.split('/')[-1]
        dst[name] = src[mask]
    elif isinstance(src, h5py._hl.group.Group) or isinstance(h5py._hl.files.File):
        src.visititems(func)
    else:
        raise TypeError


def chunkediter(dataset, sel=slice(None), readsize=16, outsize=1):
    r = range(*sel.indices(len(dataset)))
    for i in range(0, len(r), readsize):
        ids = r[i: i + readsize]
        tmp = _np.array(dataset[ids.start: ids.stop: ids.step])
        for j in range(0, len(tmp), outsize):
            yield _np.squeeze(tmp[j: j + outsize])
