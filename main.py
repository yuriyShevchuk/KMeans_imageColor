from utils.imageProcessing import ReduceColors

imgProcessing = ReduceColors()
X_train, m, n = imgProcessing.get_train_data_fromImage('./data/parrots.jpg')
params = {'height': m, 'width': n}

# showing original picture and with shrinked colours to 8 with mean and median colour of a cluster
imgProcessing.show_image(X_train, flat=True, **params)
kmeans = imgProcessing.train_kmean(X_train)
X_shr_mean = imgProcessing.shrink_colors(X_train, kmeans)
imgProcessing.show_image(X_shr_mean, flat=True, **params)
X_shr_median = imgProcessing.shrink_colors(X_train, kmeans, method='median')
imgProcessing.show_image(X_shr_median, flat=True, **params)

#------- findind min number of clusters with more then 20 psnr error (mean and median colors)
min_clus = imgProcessing.get_min_clusters_for_psnr(X_train, 20)
print(f'min number of clusters is: {min_clus}')