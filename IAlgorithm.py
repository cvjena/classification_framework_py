from Attributes import *
from Blob import Blob
import queue
import threading
import multiprocessing

__author__ = 'simon'



class IAlgorithm(object, metaclass=NonOverrideable):
    def __init__(self):
        self.use_cache = False
        self._cached_blobs = []

    def init_threading(self, num_threads = 4):
        self.manager = multiprocessing.Manager()
        self.num_threads = num_threads
        self.out_queue = multiprocessing.Queue()
        #self.read_thread = threading.Thread(target=self.read_thread)
        #self.read_thread.daemon = True
        self.cur_generator = None
        self.cur_generator_lock = multiprocessing.Lock()
        self.cur_generator_finished = True
        
        self.threading_init_finished = True

    def calculate_thread(self):
        finished = False
        while not finished:
            try:
                with self.cur_generator_lock:
                    item = next(self.cur_generator)
                self.out_queue.put(next(self._compute([item])))
            except StopIteration:
                self.cur_generator_finished = True
                finished = True

    def read_thread(self):
        finished = False
        while not finished:
            next_item = next(self.cur_generator)
            if next_item is not None:
                self.in_queue.put(next_item)    

    @non_overridable
    def compute(self, blob_generator):
        logging.debug("Using " + str(type(self)) + " compute function.")
        #        if self.use_cache:
        #            if len(self._cached_blobs)>0:
        #                return self._cached_blobs
        #            else:
        #                for blob in self._compute(blob_generator):
        #                    self._cached_blobs.append(blob)
        #                    yield blob
        #        else:
        if 'threading_init_finished' in self.__dict__ and self.threading_init_finished:
            self.cur_generator = blob_generator
            calc_threads = []
            for i in range(self.num_threads):
                t = multiprocessing.Process(target=self.calculate_thread)
                calc_threads.append(t)
                t.daemon = True
                t.start()
                
            while all([t.isAlive() for t in calc_threads]):
                try:
                    yield self.out_queue.get(timeout=0.1)
                except queue.Empty:
                    pass
            [t.join() for t in calc_threads]
            # All calc thread terminated now, but there might be still something in the queue
            while not self.out_queue.empty():
                yield self.out_queue.get()
            print("Finished queue")
        else:
            for blob in self._compute(blob_generator):
                yield blob

    def _compute(self, blob_generator):
        raise NotImplementedError("Not implemented.")

    @non_overridable
    def train(self, blob_generator):
        return self._train(blob_generator)

    def _train(self, blob_generator):
        logging.debug("Using " + str(type(self)) + " training function.")
        return self.compute(blob_generator)