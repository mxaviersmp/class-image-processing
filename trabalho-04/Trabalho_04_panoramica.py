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


# ## Imagem panorâmica

class BRIEF_create:
    def __init__(self):
        # Initiate FAST detector
        self.star = cv2.xfeatures2d.StarDetector_create()
        # Initiate BRIEF extractor
        self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    def detectAndCompute(self, img, mask):
        # find the keypoints with STAR
        kp = self.star.detect(img, mask)
        # compute the descriptors with BRIEF
        kp, des = self.brief.compute(img, kp)
        return kp, des


def detect_keypoint_and_description(img, method='SIFT'):
    allowed_methods = {
        'SIFT': cv2.SIFT_create(),
        'SURF': cv2.xfeatures2d.SURF_create(),
        'BRIEF': BRIEF_create(),
        'ORB': cv2.ORB_create()
    }
    if method not in allowed_methods:
        raise ValueError(f'method must be one of {allowed_methods}')
    detector = allowed_methods[method]
    return detector.detectAndCompute(img, None)


def plot_features(img_a, kps_a, img_b, kps_b, method):
    plt.figure(figsize=(19, 10))
    for g, (i, k, n) in enumerate(zip([img_a, img_b], [kps_a, kps_b], ['A', 'B']), start=1):
        plt.subplot(1, 2, g)
        plt.imshow(cv2.drawKeypoints(i, k, None, color=(0,255,0)))
        plt.axis('off')
        plt.title(n)
    plt.suptitle(method)
    plt.show()


def match_correspondences(feat_a, feat_b):
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    best_matches = matcher.match(feat_a, feat_b)
    return sorted(best_matches, key = lambda x: x.distance)


def plot_correspondences(matches, img_a, kps_a, img_b, kps_b, method, n_points=100):
    plt.figure(figsize=(20,8))

    img = cv2.drawMatches(
        img_a, kps_a, img_b, kps_b, matches[:n_points],
        None,
        matchColor=(0,255,0), singlePointColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.imshow(img)
    plt.axis('off')
    plt.title(method)
    plt.show()


def homography_matrix(kps_a, kps_b, matches):
    kps_a = np.float32([kp.pt for kp in kps_a])
    kps_b = np.float32([kp.pt for kp in kps_b])

    matches_a = [match.queryIdx for match in matches]
    matches_b = [match.trainIdx for match in matches]

    points_a = kps_a[matches_a]
    points_b = kps_b[matches_b]

    H, _ = cv2.findHomography(points_a, points_b, cv2.RANSAC, 5)
    return H


def crop_empty_spaces(img):
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Cria imagem panoramica com 2 imagens utilizando detecção de pontos de interesse e descritores',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'a', type=str,
        help='url ou caminho para a imagem A'
    )
    parser.add_argument(
        'b', type=str,
        help='url ou caminho para a imagem B'
    )
    parser.add_argument(
        'o', type=str,
        help='caminho para salvar a imagem panorâmica'
    )
    parser.add_argument(
        '-m',
        choices=['SIFT', 'SURF', 'BRIEF', 'ORB'],
        help='método para pontos de interesse e descritores'
    )
    args = parser.parse_args()

    if not args.m:
        raise parser.error('escolha um dos métodos')

    method = args.m
    image_A = image_from_url_or_disk(args.a)
    image_B = image_from_url_or_disk(args.b)

    key_points_A, features_A = detect_keypoint_and_description(image_A, method)
    key_points_B, features_B = detect_keypoint_and_description(image_B, method)
    plot_features(image_A, key_points_A, image_B, key_points_B, method)

    best_matches = match_correspondences(features_A, features_B)
    plot_correspondences(best_matches, image_A, key_points_A, image_B, key_points_B, method, n_points=100)

    H = homography_matrix(key_points_A, key_points_B, best_matches)

    h, w = np.array(image_A.shape) + np.array(image_B.shape)

    panoramic = cv2.warpPerspective(image_A, H, (w, h))
    panoramic[0:image_B.shape[0], 0:image_B.shape[1]] = image_B
    panoramic = crop_empty_spaces(panoramic)

    plt.imshow(panoramic, cmap='gray')
    plt.imsave(Path(args.o), panoramic, cmap='gray')
