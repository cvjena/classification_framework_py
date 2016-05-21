from Attributes import *
from Blob import Blob
import queue
import threading
import multiprocessing

__author__ = 'simon'



def pool_worker(callable_blob):
    class_obj,blob = callable_blob
    return next(class_obj._compute([blob]))

class IAlgorithm(object, metaclass=NonOverrideable):
    def __init__(self):
        self.use_cache = False
        self._cached_blobs = []

    def init_threading(self, num_threads = 4):
        #self.manager = multiprocessing.Manager()
        self.num_threads = num_threads
        #self.reader = multiprocessing.Pool(processes=1)
        #self.in_queue = multiprocessing.JoinableQueue(2*self.num_threads)
        #self.out_queue = multiprocessing.JoinableQueue()
        ##self.read_thread = threading.Thread(target=self.read_thread)
        ##self.read_thread.daemon = True
        #self.cur_generator = None
        #self.cur_generator_lock = multiprocessing.Lock()
        #self.cur_generator_finished = True

        self.threading_init_finished = True


    def read_thread(in_generator, in_queue):
        try:
            while True:
                in_queue.put(next(in_generator))    
        except StopIteration:
            pass

    def work_thread(in_queue, out_queue):
        finished = False
        while not finished:
            try:
                item = in_queue.get()
                out_queue.put(next(self._compute([item])))
                in_queue.task_done()
            except StopIteration:
                self.cur_generator_finished = True
                finished = True  

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
            # Start working threads
            workers = multiprocessing.Pool(processes=self.num_threads)
            for blob in workers.imap_unordered(pool_worker, ((self,blob) for blob in blob_generator)):
                yield blob
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