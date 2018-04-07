import logging
import os
from multiprocessing import Manager,Queue
from .batch_filter import BatchFilter
from gunpowder.batch_request import BatchRequest
from gunpowder.ext import h5py
from gunpowder.coordinate import Coordinate

logger = logging.getLogger(__name__)

from collections import namedtuple
from threading import Thread, Event

WriteRequest=namedtuple("WriteRequest","dset position data")

def BG_Write_Thread_func(file_name, request_queue, stopEvent):
        logger.info("Thread started with fn %s"%file_name)
        _file = h5py.File(file_name, 'a')

        # keep looping until we are supposed to stop
        while not stopEvent.isSet():
            if not request_queue.empty():
                logger.info("Emptying Queue")
                while not request_queue.empty():
                    req = request_queue.get()
                    _file[req.dset][req.position] = req.data
        # event was set
        # make sure there was no race condition and no stragglers
        while not request_queue.empty():
            logger.info("A few left in Queue")
            req = request_queue.get()
            _file[req.dset][req.position] = req.data

        _file.close()



class Hdf5Write(BatchFilter):
    '''Assemble arrays of passing batches in one HDF5 file. This is useful to
    store chunks produced by :class:`Scan` on disk without keeping the larger
    array in memory. The ROIs of the passing arrays will be used to determine
    the position where to store the data in the dataset.

    Args:

        dataset_names (dict): A dictionary from :class:`ArrayKey` to names of
            the datasets to store them in.

        output_dir (string): The directory to save the HDF5 file. Will be
            created, if it does not exist.

        output_filename (string): The output filename.

        compression_type (string or int): Compression strategy.  Legal values
            are 'gzip', 'szip', 'lzf'.  If an integer in range(10), this
            indicates gzip compression level. Otherwise, an integer indicates
            the number of a dynamically loaded compression filter. (See
            h5py.groups.create_dataset())

        dataset_dtypes (dict): A dictionary from :class:`ArrayKey` to datatype
            (eg. np.int8). Array to store is copied and casted to the specified type.
             Original array within the pipeline remains unchanged.
        '''

    def __init__(
            self,
            dataset_names,
            output_dir='.',
            output_filename='output.hdf',
            compression_type=None,
            dataset_dtypes=None):
        self.dataset_names = dataset_names
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.compression_type = compression_type
        if dataset_dtypes is None:
            self.dataset_dtypes = {}
        else:
            self.dataset_dtypes = dataset_dtypes

        self.dataset_shapes = {}

        self.bg_thread=None
        self.queue = Queue(maxsize=50)
        self.stopEvent=Event()

    def create_output_file(self):
        try:
            os.makedirs(self.output_dir)
        except:
            pass
        _file = h5py.File(os.path.join(self.output_dir, self.output_filename), 'a')
        for (array_key, dataset_name) in self.dataset_names.items():

            logger.debug("Create dataset for %s", array_key)

            total_roi = self.spec[array_key].roi
            dims = total_roi.dims()

            # extends of spatial dimensions
            data_shape = total_roi.get_shape()//self.spec[array_key].voxel_size
            logger.debug("Shape in voxels: %s", data_shape)
            # add channel dimensions (HACK: Unsure how to get channels)
            data_shape = Coordinate([3])[:] + data_shape
            logger.debug("Shape with channel dimensions: %s", data_shape)

            if array_key in self.dataset_dtypes:
                dtype = self.dataset_dtypes[array_key]
            else:
                dtype = self.spec[array_key].dtype
            dataset = _file.create_dataset(
                    name=dataset_name,
                    shape=data_shape,
                    compression=self.compression_type,
                    dtype=dtype)
            self.dataset_shapes[dataset_name] = data_shape
            dataset.attrs['offset'] = total_roi.get_offset()
            dataset.attrs['resolution'] = self.spec[array_key].voxel_size
        _file.close()

    def setup(self):
        fn = os.path.join(self.output_dir, self.output_filename)
        if not os.path.exists(fn):
            self.create_output_file()
        self.stopEvent.clear()
        logger.info("starting thread")
        self.bg_thread = Thread(target=BG_Write_Thread_func, args=(fn,self.queue,self.stopEvent))
        self.bg_thread.start()

    def teardown(self):
        self.stopEvent.set()
        self.bg_thread.join()


    def process(self, batch, request):
        for array_key, dataset_name in self.dataset_names.items():
            roi = batch.arrays[array_key].spec.roi
            data = batch.arrays[array_key].data
            total_roi = self.spec[array_key].roi

            assert total_roi.contains(roi), (
                "ROI %s of %s not in upstream provided ROI %s"%(
                    roi, array_key, total_roi))

            data_roi = (roi - total_roi.get_offset())//self.spec[array_key].voxel_size
            dims = data_roi.dims()
            channel_slices = (slice(None),)*max(0, len(self.dataset_shapes[dataset_name]) - dims)
            voxel_slices = data_roi.get_bounding_box()
            req = WriteRequest(dset=dataset_name,position=channel_slices + voxel_slices, data=batch.arrays[array_key].data)
            #this will hang if the queue is full
            self.queue.put(req)
