import random
from collections import deque, namedtuple

class Memory:
    def __init__(self, mem_size, *mem_fields):
        """
        mem_size: used to limit number datapoints in memory
        name_components: name of the different components that will be in memory
        --
        e.g.
        >>> memory = Memory(100, 'state_start', 'state_stop', 'reward')
        >>> memory = Memory(100, 'state_start', 'state_stop', 'done')
        """
        self.queue = deque(maxlen=mem_size)
        self.new_data_type = namedtuple('data', mem_fields)

    def add_data(self, **kwargs):
        self.queue.append(self.new_data_type(**kwargs))

    def __len__(self):
        return len(self.queue)

    def __repr__(self):
        return '{} item in memory (max {}) - {}'.format(
            len(self), self.queue.maxlen, self.new_data_type._fields
        )

    def sample_data(self, num_items):
        """
        randomly sample number of items from memory
        converts the list of data_type to data_type of list usage as a batch
        """
        num_items = min(num_items, len(self))
        data = random.sample(self.queue, num_items)
        return self.new_data_type(*zip(*data))
