import concurrent
import concurrent.futures
import logging
import threading
from AlgorithmPipeline import AlgorithmPipeline
from IAlgorithm import IAlgorithm
from Blob import Blob
from numpy import hstack
import copy

__author__ = 'simon'


class ParallelAlgorithm(IAlgorithm):
    def __init__(self):
        self.pipelines = []

    # def _compute(self, blob: Blob):
    #     executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.pipelines))
    #     # Start the load operations and mark each future with its URL
    #     futures = [executor.submit(pipeline.compute, blob) for pipeline in self.pipelines]
    #     blob.data = [out_blobs.data.flatten() for out_blobs in concurrent.futures.as_completed(futures)]
    #     blob.data = hstack(blob.data)
    #     return blob

    def add_pipeline(self, pipeline):
        self.pipelines.append(pipeline)


    def _compute(self, blob_generator):
        return self._apply_fun(blob_generator, AlgorithmPipeline.compute)

    def _train(self, blob_generator):
        return self._apply_fun(blob_generator, AlgorithmPipeline.train)

    def _apply_fun(self, blob_generator, fun):
        threadsafe_generator = ThreadsafeIter(blob_generator,len(self.pipelines))
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.pipelines))
        # Start the load operations and mark each future with its URL
        futures = [executor.submit(fun, pipeline, threadsafe_generator) for pipeline in self.pipelines]
        generators = [gen for gen in concurrent.futures.as_completed(futures)]
        has_more_blobs = True
        while has_more_blobs:
            b = Blob()
            out_blobs = [next(gen.result()) for gen in generators]
            blob_uuids = [blob.meta.uuid for blob in out_blobs]
            # If not all UUIDs are equal or num outputs is not the same as the number of pipelines
            if not out_blobs or blob_uuids.count(blob_uuids[0]) != len(blob_uuids) or len(out_blobs) != len(self.pipelines):
                logging.error("Number of elements changed within ParallelAlgorithm pipelines. This is not allowed!")
                raise Exception("Error")

            # If there are no more blobs, we are done
            if len(out_blobs) == 0:
                has_more_blobs = False
            else:
                b.data = [blob.data.ravel() for blob in out_blobs]
                b.data = hstack(b.data)
                b.meta = out_blobs[0].meta
                yield b
        logging.info("Finished training in ParallelAlgorithm")


class ThreadsafeIter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it, thread_count):
        self.it = it
        self.consumer_count = thread_count
        self.barrier = threading.Barrier(thread_count)
        self.next_item = None
        self.has_next_item = False
        self.get_next_item_lock = threading.Lock()
        self.consume_count = 0

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        # Wait for all threads to finish the last object
        # This does not cause any additional computation time since the total computation time is bound
        # by the slowest thread times the number of objects anyway. This is correct if you can assume that one
        # thread is the slowest for all input elements. This usually holds true for all architectures in
        # a CV Classification pipeline
        logging.debug("In ThreadsafeIter next")
        #self.barrier.wait()

        # TODO: This is buggy as a fast thread can get the same item multiple times
        with self.get_next_item_lock:
            if self.next_item is None or self.consume_count >= self.consumer_count:
                self.next_item = next(self.it)
                logging.debug("In ThreadsafeIter: Fetching next item " + self.next_item.meta.imagepath)
                self.consume_count = 0
            self.consume_count = self.consume_count + 1

        # We need a shallow copy to avoid problems with overwriting blob.data and affecting the other one at the same time
        return copy.copy(self.next_item)



        # Now set a flag that we did not got the next item yet. This needs to be done with a barrier to make
        # sure that there is no thread overriding self.has_next_item after it has been set to True in the next step
        #self.has_next_item = False
        #self.barrier.wait()

        # with self.get_next_item_lock:
        #     if not self.has_next_item:
        #         self.has_next_item = True
        # return self.next_item