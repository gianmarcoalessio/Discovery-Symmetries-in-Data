import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torch import nn
import random
import numpy as np
from fast_soft_sort.pytorch_ops import soft_sort


def plot_images(images,label,n,columns,rows):
    # columns*rows must be equal to n
    # Prendi le prime 50 immagini del batch
    images = images[:n]

    # Crea una griglia di subplot
    fig, axs = plt.subplots(rows, columns, figsize=(columns, rows))

    for i, ax in enumerate(axs.flatten()):
        # Rimuovi la dimensione del canale per la visualizzazione
        image = images[i].squeeze()

        # Converti il tensore in un array NumPy
        image = image.numpy()

        # Visualizza l'immagine
        ax.imshow(image, cmap='gray')
        ax.axis('off')  # Rimuovi gli assi
        ax.set_title(label[i].numpy())

    plt.tight_layout()
    plt.show()

def rotate_images(images, labels, angle=3):
    rotated_images = []
    rotated_labels = []

    for i in range(len(images)):

        image = images[i]
        label = labels[i]

        # Choose a random rotation angle (multiple of 3 degrees)
        degree = random.choice(range(0, 360, angle))
        
        # Rotate the image
        rotated_image = TF.rotate(image, degree)
        rotated_images.append(rotated_image)
        rotated_labels.append(label)

    return torch.stack(rotated_images), torch.tensor(rotated_labels)

## Sbagliata conservo per il futuro
def rotate_images_old(images, labels, angle=3):
    rotated_images = []
    rotated_labels = []
    rotated_atoms = []
    cardinality = int(360/angle)


    for i in range(len(images)):

        image = images[i]
        label = labels[i]

        if i == 0 :
            for j in range(cardinality):

                rotated_atom  = TF.rotate(image,angle*j)
                rotated_atoms.append(rotated_atom)
                rotated_images.append(rotated_atom)
                rotated_labels.append(label)

        # Choose a random rotation angle (multiple of 3 degrees)
        degree = random.choice(range(0, 360, angle))
        
        # Rotate the image
        rotated_image = TF.rotate(image, degree)
        rotated_images.append(rotated_image)
        rotated_labels.append(label)

    rotated_atoms = torch.stack(rotated_atoms)
    rotated_atoms = rotated_atoms.view(cardinality, -1)

    return torch.stack(rotated_images), torch.tensor(rotated_labels), rotated_atoms, cardinality


def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def soft_sort_approx(input, beta=1.0):
    indices = torch.argsort(input)
    probabilities = torch.softmax(beta * input, dim=-1)
    return torch.sum(indices * probabilities, dim=-1)

def plot_images_with_predictions(images, true_labels, predicted_labels, n, rows, columns):
    # Ensure that columns*rows is equal to n
    if columns * rows != n:
        raise ValueError("The product of columns and rows must be equal to n")

    # Take the first n images from the batch
    images = images[:n]
    true_labels = true_labels[:n]
    predicted_labels = predicted_labels[:n]

    # Create a grid of subplots
    fig, axs = plt.subplots(rows, columns, figsize=(columns*2, rows*2))

    for i, ax in enumerate(axs.flatten()):

        # Reshape the image to 28x28 for visualization
        image = images[i].view(28, 28).numpy()

        # Display the image
        ax.imshow(image, cmap='gray')
        ax.axis('off')  # Remove the axes

        # Set the title with true and predicted labels
        title = f"True: {true_labels[i].item()}, Pred: {predicted_labels[i].item()}"
        ax.set_title(title)

    plt.tight_layout()
    plt.show()

def calculate_accuracy(model, data_loader):
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for data, labels in data_loader:

            data=data.view(data.shape[0], -1)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            plot_images_with_predictions(data, labels, predicted, 5, 1, 5)

            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy

def get_activations(_x, _model, _name):
    activations = {}

    def get_activation_hook(name):
        def hook(_model, _input, _output):
            _ = _model, _input
            activations[name] = _output.detach()

        return hook

    layer = getattr(_model, _name)
    layer.register_forward_hook(get_activation_hook("name"))
    _ = _model(_x)
    activation = activations["name"]
    return activation

def symm_loss(atoms, cardinality, sigma_squared=0.001):
    
    # Calcola la matrice dei prodotti scalari
    scalar_products = torch.matmul(atoms, atoms.t())
    
    # Calcola la differenza dei prodotti scalari (sfrutta la proprietà del broadcasting)
    diff_squared = scalar_products.unsqueeze(2) - scalar_products.unsqueeze(1)
    diff_squared = diff_squared.pow(2)
    
    # Crea la matrice delta_ij
    delta_ij = torch.eye(cardinality).unsqueeze(-1).unsqueeze(-1)
    delta_ij = delta_ij.to(atoms.device)  # Assicurati che delta_ij sia sullo stesso dispositivo degli atomi
    
    # Calcola la loss
    loss = (1 - delta_ij * cardinality) * torch.exp(-diff_squared / sigma_squared)
    loss = loss.sum()
    
    return loss

def comm_loss_function(parameters,train_images):

    mean_data =train_images - train_images.mean(dim=0)
    mean_paramters = parameters - parameters.mean(dim=0)

    data_cov = torch.mm(mean_data.t(), mean_data)/ (train_images.shape[0] - 1)
    parameters_cov = torch.mm(mean_paramters.t(), mean_paramters)/ (train_images.shape[0] - 1)
    commutator = torch.mm(data_cov, parameters_cov) - torch.mm(parameters_cov, data_cov)
    comm_loss = torch.norm(commutator)
    return comm_loss

def reg_loss_function(weights):

    gramian = torch.mm(weights, weights.t()) # first layer weights
    row_sorted = soft_sort(gramian, direction="DESCENDING", regularization_strength=1.0, regularization="l2")
    col_sorted  = soft_sort(gramian.t(), direction="DESCENDING", regularization_strength=1.0, regularization="l2")
    reg_loss = torch.sum((gramian - row_sorted)**2) + torch.sum((gramian - col_sorted)**2)

    return reg_loss