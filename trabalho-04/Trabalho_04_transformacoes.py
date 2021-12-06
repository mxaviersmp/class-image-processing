#!/usr/bin/env python
# coding: utf-8

import cv2
import requests
import matplotlib.pyplot as plt

import numpy as np

from urllib.parse import urlparse
from pathlib import Path


def image_from_url_or_disk(path):
    if urlparse(path).scheme:
        print(f'baixando imagem de: {path}')
        resp = requests.get(path)
        image = np.asarray(bytearray(resp.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    elif Path(path).exists():
        print(f'abrindo imagem de: {path}')
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError('caminho deve ser url ou arquivo')
    return image


def show_images(grid, images, titles, filepath):
    plt.figure(figsize=(19, 10))
    for i, (img, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(*grid, i)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title(title)
        plt.axis('off')
    plt.suptitle(filepath)
    plt.show()


# ## Transformações Geométricas

def nearest_neighbor_interpolation(indices, image):
    indices = np.round(indices).astype(int)
    w, h = image.shape
    w, h = w - 1, h - 1

    scaled = image[np.clip(indices[0], 0, w), np.clip(indices[1], 0, h)]
    return scaled.astype(np.uint8)


def bilinear_interpolation(indices, image):
    dx, dy = indices - np.floor(indices)
    w, h = image.shape
    w, h = w - 1, h - 1

    x_l = np.floor(indices[0]).astype(int)
    y_l = np.floor(indices[1]).astype(int)

    f_x_y = image[np.clip(x_l, 0, w), np.clip(y_l, 0, h)]
    f_x1_y = image[np.clip(x_l + 1, 0, w), np.clip(y_l, 0, h)]
    f_x_y1 = image[np.clip(x_l, 0, w), np.clip(y_l + 1, 0, h)]
    f_x1_y1 = image[np.clip(x_l + 1, 0, w), np.clip(y_l + 1, 0, h)]

    scaled = (
        (1 - dx) * (1 - dy) * f_x_y
        + dx * (1 - dy) * f_x1_y
        + (1 - dx) * dy * f_x_y1
        + dx * dy * f_x1_y1
    )
    return scaled.astype(np.uint8)


def P(t):
    return np.maximum(t, 0)


def R(s):
    return 1/6 * (P(s + 2)**3 - 4*P(s + 1)**3 + 6*P(s)**3 - 4*P(s - 1)**3)


def bicubic_interpolation(indices, image):
    dx, dy = indices - np.floor(indices)
    w, h = image.shape
    w, h = w - 1, h - 1

    x_l = np.floor(indices[0]).astype(int)
    y_l = np.floor(indices[1]).astype(int)

    mn = [-1, 0, 1, 2]

    scaled = np.zeros_like(indices[0], dtype=float)
    for m in mn:
        for n in mn:
            f_x_y = image[np.clip(x_l + m, 0, w), np.clip(y_l + n, 0, h)]
            r_m = R(m - dx)
            r_n = R(dy - n)
            scaled += f_x_y * r_m * r_n
    return scaled.astype(np.uint8)


def L(n, dx, x, y, image):
    w, h = image.shape
    w, h = w - 1, h - 1

    return (
        (-dx * (dx - 1) * (dx - 2) * image[np.clip(x - 1, 0, w), np.clip(y + n - 2, 0, h)]) / 6
        + ((dx + 1) * (dx - 1) * (dx - 2) * image[np.clip(x, 0, w), np.clip(y + n - 2, 0, h)]) / 2
        + (-dx * (dx + 1) * (dx - 2) * image[np.clip(x + 1, 0, w), np.clip(y + n - 2, 0, h)]) / 2
        + (dx * (dx + 1) * (dx - 1) * image[np.clip(x + 2, 0, w), np.clip(y + n - 2, 0, h)]) / 6
    )


def lagrange_interpolation(indices, image):
    dx, dy = indices - np.floor(indices)
    w, h = image.shape
    w, h = w - 1, h - 1

    x_l = np.floor(indices[0]).astype(int)
    y_l = np.floor(indices[1]).astype(int)

    l_1 = L(1, dx, x_l, y_l, image)
    l_2 = L(2, dx, x_l, y_l, image)
    l_3 = L(3, dx, x_l, y_l, image)
    l_4 = L(4, dx, x_l, y_l, image)

    scaled = (
        (-dy * (dy - 1) * (dy - 2) * l_1) / 6
        + ((dy + 1) * (dy - 1) * (dy - 2) * l_2) / 2
        + (-dy * (dy + 1) * (dy - 2) * l_3) / 2
        + (dy * (dy + 1) * (dy - 1) * l_4) / 6
    )
    return scaled.astype(np.uint8)


def scale_image(sc, image, interpolation):
    img_height, img_width = image.shape
    if isinstance(sc, (list, tuple)):
        sc = [*map(float, sc)]
        sx, sy = np.divide(sc, image.shape)
    else:
        sx, sy = sc, sc

    new_height, new_width = int(img_height * sy), int(img_width * sx)

    sx = new_width / img_width
    sy = new_height / img_height

    indices = np.indices((new_width, new_height), dtype=float)
    indices[0] = indices[0] / sx
    indices[1] = indices[1] / sy
    return interpolation(indices, image)


def translation_matrix(tx, ty):
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=float)

def rotation_matrix(theta):
    theta = np.radians(theta)
    return np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ], dtype=float)

def transform_coords_matrix(indices):
    coords = indices.reshape(2, -1)
    return np.vstack((coords, np.ones_like(coords[1])))


def rotate_image(theta, image, interpolation_method):
    h, w = image.shape
    indices = np.indices((w, h))
    tx, ty = w // 2, h // 2

    T = translation_matrix(tx, ty)
    R = rotation_matrix(theta)

    A = T @ R @ np.linalg.inv(T)

    coords = transform_coords_matrix(indices)
    warp_coords = A @ coords

    x_, y_ = warp_coords[0, :], warp_coords[1, :]
    x_, y_ = x_.reshape((w, h)), y_.reshape((w, h))
    invalid_coords = np.logical_or(
        np.logical_or(x_ > w - 1, y_ > h - 1),
        np.logical_or(x_ < 0, y_ < 0)
    )
    x_ = np.clip(x_, 0, w - 1).round().astype(int)
    y_ = np.clip(y_, 0, h - 1).round().astype(int)

    indices[0] = x_
    indices[1] = y_
    rotated = interpolation_method(indices, image)
    rotated[invalid_coords] = 0

    return rotated


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Transformações geométricas em imagens.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'i', type=str,
        help='url ou caminho para a imagem'
    )
    parser.add_argument(
        'o', type=str,
        help='caminho para salvar a imagem transformada'
    )
    parser.add_argument(
        '-m',
        choices=['aproximacao', 'bilinear', 'bicubica', 'lagrange'],
        help='método de interpolação'
    )
    parser.add_argument(
        '-d', nargs=2,
        help='dimensão da imagem de saída'
    )
    parser.add_argument(
        '-e', type=float,
        help='fator de escala'
    )
    parser.add_argument(
        '-a', type=float,
        help='ângulo de rotação medido em graus no sentido anti-horário'
    )
    args = parser.parse_args()

    if (args.d or args.e) and args.a:
        raise parser.error('pode usar apenas um dos argumentos -d, -e ou -a')

    if not (args.d or args.e or args.a):
        raise parser.error('deve usar um dos argumentos -d, -e ou -a')

    interpolation_methods = {
        'aproximacao': nearest_neighbor_interpolation,
        'bilinear': bilinear_interpolation,
        'bicubica': bicubic_interpolation,
        'lagrange': lagrange_interpolation
    }

    image = image_from_url_or_disk(args.i)

    interpolation_method = interpolation_methods[args.m]


    sc = args.d or args.e
    theta = args.a
    if sc:
        img_transformed = scale_image(sc, image, interpolation_method)
    elif theta:
        img_transformed = rotate_image(theta, image, interpolation_method)
    else:
        raise

    plt.imshow(img_transformed, cmap='gray')
    plt.imsave(Path(args.o), img_transformed, cmap='gray')
