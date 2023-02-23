import matplotlib.pyplot as plt
import image_processing
import BacteriaFeatures
import numpy as np


def central_moment(x_coordinates, y_coordinates, p, q):
    """
    Calculates the central moment for given x and y coordinates.

    @param x_coordinates list  x coordinates of image boundary
    @param y_coordinates list  y coordinates of image boundary
    @param p float order of x coordinate
    @param q float order of y coordinate
    """

    sum_y = 0
    m_pq = 0

    for y in y_coordinates:
        sum_y += (y ** q)

    for x in x_coordinates:
        m_pq += ((x ** p) * sum_y)

    return m_pq


def center_coordinate(x_coordinates, y_coordinates):
    """
    Calculates the centroid of given x and y coordinates.

    @param x_coordinates list x coordinates
    @param y_coordinates list y coordinates

    Returns:
        tuple: the x and y coordinates of the centroid
    """

    m_00 = central_moment(x_coordinates, y_coordinates, p=0, q=0)
    m_10 = central_moment(x_coordinates, y_coordinates, p=1, q=0)
    m_01 = central_moment(x_coordinates, y_coordinates, p=0, q=1)

    x_center = m_10 / m_00
    y_center = m_01 / m_00

    return x_center, y_center


def calc_radial_distance(x_coordinates, y_coordinates, x_center, y_center):
    """
    Calculates the radial distance of each point from the centroid.

    @param x_coordinates list x coordinates
    @param y_coordinates list  y coordinates
    @param x_center float x coordinate of the centroid
    @param y_center float y coordinate of the centroid

    Returns:
        numpy array: array of radial distances
    """

    dist = np.sqrt(np.power(x_coordinates - x_center, 2) + np.power(y_coordinates - y_center, 2))

    return dist


def calc_normalized_radial_distance(dist):

    """
    Calculates the normalized radial distance.
    equation: x / max

    @param dist numpy array of radial distances

    Returns:
        numpy array: array of normalized radial distances
    """

    normalized_radial_distance = dist / np.max(dist)

    return normalized_radial_distance


def discrete_fourier_transform(normalized_radial_distance):

    """
    Calculates the discrete Fourier transform of the normalized radial distances.

    @param normalized_radial_distance numpy  array of normalized radial distances

    Returns:
        float: the standard deviation of the Fourier transform
    """

    N = normalized_radial_distance.size
    output = []
    for u_param in range(N):

        exponential = np.exp((-1j * 2 * np.pi * np.arange(N) * u_param) / N)
        multiply = normalized_radial_distance * exponential

        output.append(np.average(multiply))

    return np.std(output)


def calc_fourier_descriptor(bacteria, fig_export_path, fig_name, img_dimension=[1200, 1200], skip=1):
    # https://en.wikipedia.org/wiki/Discrete_Fourier_transform
    """
    Calculates the Fourier descriptor for a colony / micro-colony.
    @param  fig_export_path directory to export images
    @param fig_name image name
    @return fourier transform
    """

    contours = image_processing.fill_contours(bacteria, img_dimension=None, um_pixel_ratio=0.144, margin=1)
    # save image
    if fig_export_path:
        image_processing.save_fig(contours, fig_export_path, fig_name)
    # external image
    boundary = image_processing.find_external_contours(contours, fig_export_path, fig_name + '_boundary')
    # convert pixel to coordinate
    x_coordinates, y_coordinates = image_processing.pixel_to_coordinate(boundary)
    x_coordinates, y_coordinates = image_processing.coordinate_sampling(x_coordinates, y_coordinates, skip=skip)
    # center position
    x_center, y_center = center_coordinate(x_coordinates, y_coordinates)
    if fig_export_path:
        fig, ax = plt.subplots()
        plt.scatter(x_coordinates, y_coordinates)
        plt.scatter(x_center, y_center)
        plt.savefig(fig_export_path + fig_name + '_coordinates.png')
    radial_distance = calc_radial_distance(x_coordinates, y_coordinates, x_center, y_center)
    normalized_radial_distance = calc_normalized_radial_distance(radial_distance)

    return discrete_fourier_transform(normalized_radial_distance)

