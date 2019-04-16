import time
from contextlib import contextmanager


def timeit_fn(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('function [{:s}] took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap


@contextmanager
def timeit_context(name):
    time1 = time.time()
    yield
    time2 = time.time()
    print('[{}] took {:.3f} ms'.format(name, (time2-time1)*1000.0))
