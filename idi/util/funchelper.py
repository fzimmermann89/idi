from functools import wraps as _wraps, reduce as _reduce


def aslist(fn=None):
    def aslist_return(fn):
        @_wraps(fn)
        def aslist_helper(*args, **kw):
            return list(fn(*args, **kw))

        return aslist_helper

    if fn is None:
        return aslist_return
    return aslist_return(fn)


def asgen(fn=None):
    def asgen_return(fn):
        @_wraps(fn)
        def asgen_helper(arg, *args, **kw):
            for item in arg:
                yield fn(item, *args, **kw)

        return asgen_helper

    if fn is None:
        return asgen_return
    return asgen_return(fn)


# def aslengen(fn=None):
#     class lengen(object):
#         def __init__(self, gen, length):
#             self.gen = gen
#             self.length = length
#
#         def __len__(self):
#             return self.length
#
#         def __iter__(self):
#             return self.gen


def parallel(fn=None):
    from collections import deque
    import time
    try:
        from pathos.multiprocessing import Pool
    except ImportError:
        import warnings
        warnings.warn('no pathos available, be careful with parallel decorator and pickling errors.')
        from multiprocessing import Pool

    def parallel_return(fn):
        @_wraps(fn)
        def parallel_helper(arg, *args, **kw):
            with Pool(4) as p:
                q = deque()
                for item in arg:
                    if len(q) > 4 and q[0].ready():
                        yield q.popleft().get()
                    while len(q) > 8 and not q[0].ready():
                        time.sleep(0.01)
                    q.append(p.apply_async(fn, (item,) + args, kw))
                for r in q:
                    yield r.get()

        return parallel_helper

    if fn is None:
        return parallel_return
    return parallel_return(fn)


def chain(*fns):
    return _reduce(lambda f, g: lambda x: f(g(x)), fns)


def group(iterable, n):
    return zip(*([iter(iterable)] * n))
