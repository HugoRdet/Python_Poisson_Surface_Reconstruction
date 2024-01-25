import torch
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

def sphere_points_and_normals_torch(radius, num_points_theta, num_points_phi):
    """
    Generate coordinates of points on the surface of a sphere and their normals using PyTorch.

    Parameters:
    radius (float): The radius of the sphere.
    num_points_theta (int): The number of points to generate along the azimuth angle (θ).
    num_points_phi (int): The number of points to generate along the polar angle (φ).

    Returns:
    torch.Tensor: A tensor of shape (num_points_theta * num_points_phi, 6) containing the coordinates of the points and their normals.
    """
    # Create tensors for azimuth and polar angles
    theta = torch.linspace(0, 2 * torch.pi, num_points_theta)
    phi = torch.linspace(0, torch.pi, num_points_phi)

    # Create a grid of angles
    theta, phi = torch.meshgrid(theta, phi)

    # Calculate the x, y, and z coordinates for each point
    x = radius * torch.sin(phi) * torch.cos(theta)
    y = radius * torch.sin(phi) * torch.sin(theta)
    z = radius * torch.cos(phi)

    # For a sphere centered at the origin, the normals are the same as the coordinates
    normals = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)

    # Normalize the normals
    normals = normals / torch.norm(normals, dim=1, keepdim=True)

    # Reshape and stack the coordinates and normals
    points_with_normals = torch.cat((x.flatten().unsqueeze(1), y.flatten().unsqueeze(1), z.flatten().unsqueeze(1), normals), dim=1)
    return points_with_normals

def plot_sphere_with_normals(points_with_normals, show_normals=True, normal_length=0.1):
    """
    Plot the points on a sphere and optionally their normals.

    Parameters:
    points_with_normals (torch.Tensor): A tensor of shape (n, 6) containing the coordinates and normals of the points.
    show_normals (bool): If True, plot the normals.
    normal_length (float): The length of the normal vectors in the plot.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract coordinates and normals
    x, y, z = points_with_normals[:, 0], points_with_normals[:, 1], points_with_normals[:, 2]
    nx, ny, nz = points_with_normals[:, 3], points_with_normals[:, 4], points_with_normals[:, 5]

    # Plot the points
    ax.scatter(x, y, z, color='b', s=5)  # s is the size of the points

    # Optionally plot the normals
    if show_normals:
        for i in range(len(x)):
            ax.quiver(x[i], y[i], z[i], nx[i], ny[i], nz[i], length=normal_length, color='r')

    # Set plot limits and labels
    ax.set_xlim([-0.7, 0.7])
    ax.set_ylim([-0.7, 0.7])
    ax.set_zlim([-0.7, 0.7])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.show()