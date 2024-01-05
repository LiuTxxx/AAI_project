import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

if __name__ == "__main__":
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load MNIST data
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=len(mnist_train), shuffle=True)

    # Extract features and labels
    data_iterator = iter(train_loader)
    images, labels = data_iterator.next()

    # Flatten the images
    images_flat = images.view(images.shape[0], -1)

    # Standardize the data
    data_standardized = StandardScaler().fit_transform(images_flat.numpy())

    # Reduce dimensionality using PCA for visualization purposes
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_standardized)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=10, random_state=42)
    predicted_labels = kmeans.fit_predict(data_standardized)

    # Visualize the results
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=predicted_labels, cmap='viridis', s=5)
    plt.title('K-Means Clustering of MNIST')
    plt.show()

    # Calculate the most common true label in each cluster
    cluster_assigned_labels = []
    for cluster in range(10):
        cluster_indices = np.where(predicted_labels == cluster)[0]
        cluster_true_labels = labels[cluster_indices]
        most_common_label = np.bincount(cluster_true_labels).argmax()
        cluster_assigned_labels.append(most_common_label)

    # Map predicted labels to the most common true labels
    mapped_labels = np.array([cluster_assigned_labels[label] for label in predicted_labels])

    # Calculate accuracy
    accuracy = np.sum(mapped_labels == labels) / len(labels)
    print(f"Accuracy: {accuracy}")