import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def kmeans_mnist():
    # Load MNIST dataset
    print("Load dataset")
    mnist = datasets.fetch_openml('mnist_784')
    data = mnist.data.astype(float)
    target = mnist.target.astype(int)

    # Split the dataset into training and testing sets (to be used for accuracy calculation)
    print("Split dataset")
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=42)

    # Reduce dimensionality with PCA (optional but can be useful for visualization)
    print("Start PCA")
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)

    # Apply k-means clustering
    print("Start training")
    kmeans = KMeans(n_clusters=10, random_state=42)
    kmeans.fit(data_train)

    # Predict cluster labels for the training set
    print("Start predicting")
    train_labels = kmeans.predict(data_train)

    # # Visualize the clusters (using PCA for 2D visualization)
    # print("Start visualizing")
    # plt.scatter(data_pca[:, 0], data_pca[:, 1], c=train_labels, cmap='viridis', s=50)
    # plt.title('K-Means Clustering of MNIST Training Data')
    # plt.show()

    print("Start labeling")
    # Assign predicted labels based on the majority true label in each cluster for training set
    cluster_to_label = {}
    for cluster in range(10):
        true_labels = target_train[train_labels == cluster]
        most_common_label = np.bincount(true_labels).argmax()
        cluster_to_label[cluster] = most_common_label

    # Predict cluster labels for the test set
    test_labels = kmeans.predict(data_test)

    # Assign predicted labels based on the majority true label in each cluster for test set
    predicted_labels = [cluster_to_label[cluster] for cluster in test_labels]

    # Calculate accuracy
    accuracy = metrics.accuracy_score(target_test, predicted_labels)
    print(f'Accuracy: {accuracy * 100:.2f}%')


if __name__ == "__main__":
    kmeans_mnist()
