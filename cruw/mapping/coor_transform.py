import numpy as np


def pol2cart(rho, phi):
    """
    Transform from polar to cart coordinates
    :param rho: distance to origin
    :param phi: angle (rad)
    :return: x, y
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def cart2pol(x, y):
    """
    Transform from cart to polar coordinates
    :param x: x
    :param y: y
    :return: rho, phi (rad)
    """
    rho = (x * x + y * y) ** 0.5
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart_ramap(rho, phi):
    """
    Transform from polar to cart under RAMap coordinates
    :param rho: distance to origin
    :param phi: angle (rad) under RAMap coordinates
    :return: x, y
    """
    x = rho * np.sin(phi)
    y = rho * np.cos(phi)
    return x, y


def cart2pol_ramap(x, y):
    """
    Transform from cart to polar under RAMap coordinates
    :param x: x
    :param y: y
    :return: rho, phi (rad) under RAMap coordinates
    """
    rho = (x * x + y * y) ** 0.5
    phi = np.arctan2(x, y)
    return rho, phi
