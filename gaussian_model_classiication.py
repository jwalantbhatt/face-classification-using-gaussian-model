import cv2
import numpy as np
import os
import gc
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# initialize
gc.collect()
afpr = np.zeros(1000)
atpr = np.zeros(1000)
pca = PCA(n_components=50)

# define paths
train_path_faces = os.getcwd() + '\\faces_training'
train_path_non_faces = os.getcwd() + '\\non_faces_training'
test_path_faces = os.getcwd() + '\\faces_testing'
test_path_non_faces = os.getcwd() + '\\non_faces_testing'


# load an 2D array of image arrays
def load_array(path):
    images_flattened_list = []
    for filename in os.listdir(path):
        image = cv2.imread(path + '\\' + filename)
        # image_resize = cv2.resize(image,(60, 60))
        data = np.array(image)
        flattened = data.flatten()
        images_flattened_list.append(flattened)
    images_flattened_array = np.vstack(images_flattened_list)
    reduced_array = pca.fit_transform(images_flattened_array)
    return reduced_array


# learn mean and covariance from training data
def learn_parameters(array):
    mean = np.mean(array, axis=0)
    covariance = np.cov(array, rowvar=False)
    diagonal = np.diag(covariance)
    diag_matrix = np.diag(diagonal)
    return mean, diag_matrix


# find posterior probability
def get_posterior(p1, p2):
    p = p1/(p1 + p2)
    return p


# classify as positives and negatives with given threshold
def classify(thresh, post_array):
    pos = 0
    neg = 0
    for x in post_array:
        if x > thresh:
            pos += 1
        else:
            neg += 1
    return pos, neg


# calculate rates
def get_rates(r1, r2):
    r = float(r1) / float(r1 + r2)
    return r


# load data into arrays
face_images = load_array(train_path_faces)
non_face_images = load_array(train_path_non_faces)
face_test = load_array(test_path_faces)
non_face_test = load_array(test_path_non_faces)


# fitting parameters mean m and covariance c
mf, cf = learn_parameters(face_images)                        # face images
mnf, cnf = learn_parameters(non_face_images)                  # non-face images


# calculate likelihood on face test images
pw1 = multivariate_normal.pdf(face_test, mf, cf)              # world(w) is face
pw0 = multivariate_normal.pdf(face_test, mnf, cnf)            # world(w) is not face

# evaluate posterior probability for world state face
posterior_prob_face = get_posterior(pw1, pw0)

# calculate likelihood on non-face test images
pw1n = multivariate_normal.pdf(non_face_test, mf, cf)         # world(w) is face
pw0n = multivariate_normal.pdf(non_face_test, mnf, cnf)       # world(w) is not face

# evaluate posterior probability for world state face
posterior_prob_non_faces = get_posterior(pw1n, pw0n)

# populate confusion matrix variables
true_positive, false_negative = classify(0.5, posterior_prob_face)
false_positive, true_negative = classify(0.5, posterior_prob_non_faces)

# calculate false negative rate and false positive rate
fnr = get_rates(false_negative, true_positive)
fpr = get_rates(false_positive, true_negative)

# display
print "The False Negative Rate is: ", fnr
print "The False Positive Rate is: ", fpr

# array of rates
for j in range(0, 1000):
    a, b = classify(float(j)/1000, posterior_prob_face)
    c, d = classify((float(j)/1000), posterior_prob_non_faces)
    atpr[j] = get_rates(a, b)
    afpr[j] = get_rates(c, d)


# plot ROC
plt.plot(afpr, atpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve - model1")
plt.show()





