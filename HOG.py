import logging
from numpy.core.umath import arctan2, cos, sin
from skimage import color
from IAlgorithm import IAlgorithm
from Blob import Blob
import numpy as np
from scipy import sqrt
from scipy.constants import pi
from scipy.ndimage import uniform_filter
import cv2

__author__ = 'simon'


import multiprocessing
def fun(f,q_in,q_out):
        while True:
                i,x = q_in.get()
                if i is None:
                        break
                q_out.put((i,f(x)))

def parmap(f, X, nprocs = 2):
        q_in   = multiprocessing.Queue(2)
        q_out  = multiprocessing.Queue()

        proc = [multiprocessing.Process(target=fun,args=(f,q_in,q_out)) for _ in range(nprocs)]
        for p in proc:
                p.daemon = True
                p.start()
        sent = [q_in.put((i,x)) for i,x in enumerate(X)]
        [q_in.put((None,None)) for _ in range(nprocs)]
        res = [q_out.get() for _ in range(len(sent))]

        [p.join() for p in proc]

        return [x for i,x in sorted(res)]

class HOG(IAlgorithm):
        def __init__(self):
                # _winSize, _blockSize, _blockStride, _cellSize, _nbins
                self.hog = cv2.HOGDescriptor()#self.init_threading()
                self.dim_red_matrix = np.random.rand(self.hog.getDescriptorSize(), 50)
                
        def single_compute(self,blob):
                image = color.rgb2gray(blob.data)
                blob.data = self.cvhog(image)
                return blob
        
        def _compute(self, blob_generator):
                if False:
                        i = 0
                        for blob in parmap(self.single_compute, blob_generator):
                                yield blob
                else:
                        for blob in blob_generator:
                                yield self.single_compute(blob)



        def cvhog(self,img, orientations=9):
                img = (img*255).astype('uint8')
                nwindowsX = (img.shape[1] - self.hog.winSize[0])/self.hog.cellSize[0] + 1
                desc = self.hog.compute(img)
                desc = desc.reshape((-1,nwindowsX,self.hog.getDescriptorSize()))
                return np.dot(desc,self.dim_red_matrix)

def hog(image, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(3, 3), visualise=False, normalise=False):
        """Extract Histogram of Oriented Gradients (HOG) for a given image.
        Compute a Histogram of Oriented Gradients (HOG) by
            1. (optional) global image normalisation
            2. computing the gradient image in x and y
            3. computing gradient histograms
            4. normalising across blocks
            5. flattening into a feature vector
        Parameters
        ----------
        image : (M, N) ndarray
            Input image (greyscale).
        orientations : int
            Number of orientation bins.
        pixels_per_cell : 2 tuple (int, int)
            Size (in pixels) of a cell.
        cells_per_block  : 2 tuple (int,int)
            Number of cells in each block.
        visualise : bool, optional
            Also return an image of the HOG.
        normalise : bool, optional
            Apply power law compression to normalise the image before
            processing.
        Returns
        -------
        newarr : ndarray
            HOG for the image as a 1D (flattened) array.
        hog_image : ndarray (if visualise=True)
            A visualisation of the HOG image.
        References
        ----------
        * http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
        * Dalal, N and Triggs, B, Histograms of Oriented Gradients for
          Human Detection, IEEE Computer Society Conference on Computer
          Vision and Pattern Recognition 2005 San Diego, CA, USA
        """
        image = np.atleast_2d(image)

        """
    The first stage applies an optional global image normalisation
    equalisation that is designed to reduce the influence of illumination
    effects. In practice we use gamma (power law) compression, either
    computing the square root or the log of each colour channel.
    Image texture strength is typically proportional to the local surface
    illumination so this compression helps to reduce the effects of local
    shadowing and illumination variations.
    """

        assert_nD(image, 2)

        if normalise:
                image = sqrt(image)

        """
    The second stage computes first order image gradients. These capture
    contour, silhouette and some texture information, while providing
    further resistance to illumination variations. The locally dominant
    colour channel is used, which provides colour invariance to a large
    extent. Variant methods may also include second order image derivatives,
    which act as primitive bar detectors - a useful feature for capturing,
    e.g. bar like structures in bicycles and limbs in humans.
    """

        if image.dtype.kind == 'u':
                # convert uint image to float
                # to avoid problems with subtracting unsigned numbers in np.diff()
                image = image.astype('float')

        gx = np.empty(image.shape, dtype=np.double)
        gx[:, 0] = 0
        gx[:, -1] = 0
        gx[:, 1:-1] = image[:, 2:] - image[:, :-2]
        gy = np.empty(image.shape, dtype=np.double)
        gy[0, :] = 0
        gy[-1, :] = 0
        gy[1:-1, :] = image[2:, :] - image[:-2, :]

        """
    The third stage aims to produce an encoding that is sensitive to
    local image content while remaining resistant to small changes in
    pose or appearance. The adopted method pools gradient orientation
    information locally in the same way as the SIFT [Lowe 2004]
    feature. The image window is divided into small spatial regions,
    called "cells". For each cell we accumulate a local 1-D histogram
    of gradient or edge orientations over all the pixels in the
    cell. This combined cell-level 1-D histogram forms the basic
    "orientation histogram" representation. Each orientation histogram
    divides the gradient angle range into a fixed number of
    predetermined bins. The gradient magnitudes of the pixels in the
    cell are used to vote into the orientation histogram.
    """

        magnitude = sqrt(gx ** 2 + gy ** 2)
        orientation = arctan2(gy, gx) * (180 / pi) % 180

        sy, sx = image.shape
        cx, cy = pixels_per_cell
        bx, by = cells_per_block

        n_cellsx = int(np.floor(sx // cx))  # number of cells in x
        n_cellsy = int(np.floor(sy // cy))  # number of cells in y

        # compute orientations integral images
        orientation_histogram = np.zeros((n_cellsy, n_cellsx, orientations))
        subsample = np.index_exp[cy // 2:cy * n_cellsy:cy,
                                 cx // 2:cx * n_cellsx:cx]
        for i in range(orientations):
                # create new integral image for this orientation
                # isolate orientations in this range

                temp_ori = np.where(orientation < 180.0 / orientations * (i + 1),
                                    orientation, -1)
                temp_ori = np.where(orientation >= 180.0 / orientations * i,
                                    temp_ori, -1)
                # select magnitudes for those orientations
                cond2 = temp_ori > -1
                temp_mag = np.where(cond2, magnitude, 0)

                temp_filt = uniform_filter(temp_mag, size=(cy, cx))
                orientation_histogram[:, :, i] = temp_filt[subsample]

        # now for each cell, compute the histogram
        hog_image = None

        if visualise:
                from .. import draw

                radius = min(cx, cy) // 2 - 1
                hog_image = np.zeros((sy, sx), dtype=float)
                for x in range(n_cellsx):
                        for y in range(n_cellsy):
                                for o in range(orientations):
                                        centre = tuple([y * cy + cy // 2, x * cx + cx // 2])
                                        dx = radius * cos(float(o) / orientations * np.pi)
                                        dy = radius * sin(float(o) / orientations * np.pi)
                                        rr, cc = draw.line(int(centre[0] - dx),
                                                           int(centre[1] + dy),
                                                           int(centre[0] + dx),
                                                           int(centre[1] - dy))
                                        hog_image[rr, cc] += orientation_histogram[y, x, o]

        """
    The fourth stage computes normalisation, which takes local groups of
    cells and contrast normalises their overall responses before passing
    to next stage. Normalisation introduces better invariance to illumination,
    shadowing, and edge contrast. It is performed by accumulating a measure
    of local histogram "energy" over local groups of cells that we call
    "blocks". The result is used to normalise each cell in the block.
    Typically each individual cell is shared between several blocks, but
    its normalisations are block dependent and thus different. The cell
    thus appears several times in the final output vector with different
    normalisations. This may seem redundant but it improves the performance.
    We refer to the normalised block descriptors as Histogram of Oriented
    Gradient (HOG) descriptors.
    """

        n_blocksx = (n_cellsx - bx) + 1
        n_blocksy = (n_cellsy - by) + 1
        normalised_blocks = np.zeros((n_blocksy, n_blocksx,
                                      by, bx, orientations))

        for x in range(n_blocksx):
                for y in range(n_blocksy):
                        block = orientation_histogram[y:y + by, x:x + bx, :]
                        eps = 1e-5
                        normalised_blocks[y, x, :] = block / sqrt(block.sum() ** 2 + eps)

        """
    The final step collects the HOG descriptors from all blocks of a dense
    overlapping grid of blocks covering the detection window into a combined
    feature vector for use in the window classifier.
    """

        if visualise:
                return normalised_blocks, hog_image
        else:
                return normalised_blocks.squeeze()


def assert_nD(array, ndim, arg_name='image'):
        """
        Verify an array meets the desired ndims.
        Parameters
        ----------
        array : array-like
            Input array to be validated
        ndim : int or iterable of ints
            Allowable ndim or ndims for the array.
        arg_name : str, optional
            The name of the array in the original function.
        """
        array = np.asanyarray(array)
        msg = "The parameter `%s` must be a %s-dimensional array"
        if isinstance(ndim, int):
                ndim = [ndim]
        if not array.ndim in ndim:
                raise ValueError(msg % (arg_name, '-or-'.join([str(n) for n in ndim])))