import pylab
import numpy as np

from skimage.io import imread
from skimage.util import img_as_float
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error


class ReduceColors:

    def get_train_data_fromImage(self, path):
        image = imread(path)
        image = img_as_float(image)
        m = image.shape[0]
        n = image.shape[1]
        X = np.reshape(image, (image.shape[0] * image.shape[1], image.shape[2]))
        return X, m, n

    def show_image(self, img, flat=True, **kwargs):
        if flat:
            m_ = kwargs.get('height')
            n_ = kwargs.get('width')
            img = np.reshape(img, (m_, n_, 3))
        pylab.imshow(img)
        pylab.show()
        pass

    def train_kmean(self, X, clusters=8):
        kmeans = KMeans(n_clusters=clusters, init='k-means++', random_state=241)
        kmeans.fit(X)
        return kmeans

    def shrink_colors(self, X_tr, kmeans_, method='mean'):
        labels = kmeans_.labels_
        X = np.copy(X_tr)
        k = np.amax(labels) + 1
        if method != 'median':
            shrink_method = np.mean
        else:
            shrink_method = np.median
        for i in range(k):
            cluster_idx = np.where(labels == i)
            X[cluster_idx] = shrink_method(X[cluster_idx], axis=0)
        return X

    def get_max_psnr(self, X_tr, X_mean, X_median):
        max_i = np.amax(X_tr)
        psnr_mean = 10 * np.log10(max_i / (mean_squared_error(X_tr, X_mean)))
        psnr_median = 10 * np.log10(max_i / (mean_squared_error(X_tr, X_median)))
        return psnr_mean if psnr_mean >= psnr_median else psnr_median

    def get_min_clusters_for_psnr(self, X_tr, max_psnr=20):
        min_clusters = -1
        for clusters in range(2, 21):
            kmean = self.train_kmean(X_tr, clusters)
            X_mean = self.shrink_colors(X_tr, kmean)
            X_median = self.shrink_colors(X_tr, kmean, method='median')
            psnr = self.get_max_psnr(X_tr, X_mean, X_median)
            if psnr > max_psnr:
                min_clusters = clusters
                break
        return min_clusters
