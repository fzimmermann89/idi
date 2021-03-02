import multiprocessing as _mp
import ctypes as _ctypes
import numpy as _np
import h5py as _h5py
import contextlib as _contextlib


# _mp = _multiprocessing.get_context('spawn')


class datasetreader:
    """
    reads in multiple datasets of an h5 in seperate background processes
    """

    def __init__(self, datasets, file=None, sizecache=20, willread=None):
        """
        reader for multiple datasets of same shape and dtype of an h5 file

        :param datasets: Iterable of Datasets or Strings
        :param file: None (if datasets are h5 Datasets) or String, h5py File or Group, List of Strings (same length as datasets)
        :param sizecache: int, size of the cache
        :param willread: bool array, indices where willread is false are promised to never be read, or None
        """
        ds_names = list()
        filenames = list()
        for i, d in enumerate(datasets):
            if isinstance(d, _h5py.Dataset):
                ds_names.append(d.name)
                filenames.append(d.file.filename)
            elif isinstance(d, str):
                ds_names.append(d)
                if isinstance(file, str):
                    filenames.append(file)
                elif isinstance(file, list) and len(file) == len(datasets):
                    filenames.append(file[i])
                elif isinstance(file, _h5py.File):
                    filenames.append(file.filename)
                elif isinstance(file, _h5py.Group):
                    filenames.append(file.file.filename)
                    ds_names[i] = file.file.name + ds_names[i]
                else:
                    raise TypeError('if datasets are not h5py datasets, a file must be specified as h5py.File or filename(s)')
            else:
                raise TypeError('datasets should be list of strings or datasets')

        with _h5py.File(filenames[0], 'r') as h5file:
            shape = h5file[ds_names[0]].shape
            self._dtype = h5file[ds_names[0]].dtype
            ctype = _np.ctypeslib.as_ctypes_type(self._dtype)

        for d, filename in zip(ds_names[1:], filenames[1:]):
            with _h5py.File(filename, 'r') as h5file:
                if h5file[d].shape != shape:
                    raise ValueError('should have same shape')
                if _np.ctypeslib.as_ctypes_type(h5file[d].dtype) is not ctype:
                    raise TypeError('should have same dtype')

        nds = len(ds_names)
        self._nds = nds
        self._n = shape[0]
        self._lastread = _mp.Value('i', -1)
        if willread is not None:
            if not isinstance(willread, _np.ndarray) or not willread.shape[0] == shape[0] or not willread.dtype == bool:
                raise ValueError('willread must be a boolean array of same length as the datasets or None')
            if _np.all(willread):
                willread = None
            else:
                self._lastread.value = _np.argmax(willread) - 1
        shapecache = (sizecache, *shape[1:])
        self._rindices = [_mp.RawArray(_ctypes.c_int64, int(sizecache)) for i in range(nds)]
        self._indices = [_np.frombuffer(r, _np.int64) for r in self._rindices]
        for i in self._indices:
            i[:] = -5
        self._rcache = [_mp.RawArray(ctype, int(_np.prod(shapecache))) for i in range(nds)]
        self._willread = willread
        self._cache = [_np.frombuffer(r, self._dtype).reshape(shapecache) for r in self._rcache]

        self._readevents = [_mp.Event() for i in range(nds)]
        self._writeevents = [_mp.Event() for i in range(nds)]
        self._readerprocess = [
            _mp.Process(
                name=f'reader_process_{i}',
                target=self._reader,
                args=(
                    self._rindices[i],
                    self._rcache[i],
                    shapecache,
                    self._lastread,
                    self._readevents[i],
                    self._writeevents[i],
                    filenames[i],
                    ds_names[i],
                    willread,
                ),
            )
            for i in range(nds)
        ]
        for p in self._readerprocess:
            p.start()

    @staticmethod
    def _reader(rindices, rcache, shape, lastread, readevent, writeevent, filename, dsname, willread):
        """
        this function will run in separete processes to read the data into the cache
        """
        import mkl

        mkl.set_num_threads_local(1)

        if willread is not None:
            readall = False
            readindices = _np.nonzero(willread)[0]
            inextwrite = 0
        else:
            readall = True
            nextwrite = 0

        with _h5py.File(filename, 'r') as file:
            dataset = file[dsname]
            n = dataset.shape[0]
            indices = _np.frombuffer(rindices, _np.int64)
            cache = _np.frombuffer(rcache, dataset.dtype).reshape(shape)
            while True:
                if lastread.value > n:
                    return
                empty = _np.where(indices < (lastread.value - 1))[0]
                nempty = len(empty)
                if nempty == 0:
                    if not readevent.wait(0.1):
                        writeevent.set()
                    readevent.clear()
                    continue
                if readall:
                    start = max(nextwrite, lastread.value)
                    if start >= n:
                        return
                    if start + nempty > n - 1:
                        nempty = n - start
                        empty = empty[:nempty]
                    nextwrite = start + nempty
                    read = dataset[start:nextwrite, ...]
                    cache[empty, ...] = read
                    indices[empty] = _np.arange(start, nextwrite)
                    writeevent.set()
                else:
                    if readindices[inextwrite] < lastread.value:
                        inextwrite = _np.argmax(readindices > lastread.value)
                    if inextwrite + nempty > readindices.shape[0]:
                        nempty = readindices.shape[0] - inextwrite
                        empty = empty[:nempty]
                    readids = readindices[inextwrite : inextwrite + nempty]
                    read = dataset[readids, ...]
                    cache[empty, ...] = read
                    indices[empty] = readids
                    inextwrite = inextwrite + nempty
                    writeevent.set()
                    if inextwrite == readindices.shape[0]:
                        return

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._lastread.value = self._n + 1
        for ev in self._readevents:
            ev.set()
        del self._cache
        del self._rcache
        for p in self._readerprocess:
            p.join(100)
            p.kill()
        del self._readerprocess

    def __getitem__(self, i):
        innerind = tuple()
        if isinstance(i, int):
            n = i
            t = slice(None, None, None)
        elif isinstance(i, tuple):
            n, t = i[:2]
            if len(i) >= 2:
                innerind = i[2:]
            if isinstance(t, int):
                t = slice(t, t + 1, None)
            if not isinstance(n, int):
                raise IndexError('image index must be int')
        else:
            raise IndexError('must give two indices, image (int) and tile (int or slice).')
        if n < 0 or n >= self._n:
            raise IndexError(f'image index must be between 0 and {self._n - 1}')
        try:
            r = range(self._nds)[t]
        except IndexError:
            raise IndexError('tile index invalid')
        if self._lastread.value > n:
            raise IndexError('must request images monotonically increasing')
        if self._willread is not None and not self._willread[n]:
            raise IndexError(f'willread {n} was set to false, but trying to read!')
        self._lastread.value = n
        ret = []
        for ct in r:
            self._readevents[ct].set()
            while True:
                index = _np.where(self._indices[ct] == n)[0]
                if len(index) == 1:
                    ind = (index[0], *innerind)
                    ret.append(self._cache[ct][ind])
                    break
                self._writeevents[ct].wait()
                self._writeevents[ct].clear()

        return _np.squeeze(_np.asarray(ret))

    def __len__(self):
        return self._n

    @property
    def shape(self):
        return (self._n, self._nds)

    @property
    def dtype(self):
        return self._dtype


@_contextlib.contextmanager
def arrayreader(array, *args, **kwargs):
    """
    dummy context manager around an array
    """
    yield array
