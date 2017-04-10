# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image, ImageDraw

import image_proc

q_mark_path = "assets/q_mark.png"

def draw_contours(im):
    """Draw gray contours around the letters

    :param im: PIL.Image
        original image representing letters
    :return: PIL.Image
        image with countours around each letter
    """
    cont = image_proc.get_contours(im)
    draw = ImageDraw.Draw(im)
    for i in range(cont.shape[0]):
        left = cont[i, 0]
        right = cont[i, 2]
        low = cont[i, 1]
        high = cont[i, 3]
        draw.line((left, low, right, low), fill=128)
        draw.line((left, high, right, high), fill=128)
        draw.line((left, low, left, high), fill=128)
        draw.line((right, low, right, high), fill=128)
    return im


def gray_out_letter(im, pos=[0], col=192):
    """Replace any non white color within letter contours with gray

    :param im: PIL.Image
    :param pos: int
        a list of positions of letters which are incorrect
    :param col: int
        the new color of the incorrect letter (0 = black, 255 = white)
    :return: PIL.Image
        the modified image with grayed out incorrect letters
    """

    contours = image_proc.get_contours(im)
    npim = np.array(im)
    for p in pos:
        left, low, right, high = contours[p, :]
        letter = npim[low:high + 1, left:right + 1]
        letter[letter < 255] = col
        npim[low:high + 1, left:right + 1] = letter
    im2 = Image.fromarray(npim)
    return im2


def flag_missing_letter(im, pos=[0], col=128):
    """Flags missing letters wherever they occur

    :param im: PIL.Image
    :param pos: list
        list of the positions which miss a letter
    :param col: int
        the color of the question mark (0 = black, 255 = white)
    :return: PIL.Image
        the modified image with question mark flagging missing letters
    """
    # Load question mark image
    with Image.open(q_mark_path) as image:
        q_mark_im = image.copy()
    q_mark_im = q_mark_im.convert("L")

    # Get positions of spaces between letters
    spaces = image_proc.get_spaces(im)
    npim = np.array(im)

    # Compute average letter width
    total_width = 0
    for i in range(1,spaces.shape[0]):
        total_width += spaces[i,0] - spaces[i-1,2]
    average_width = int(np.floor(total_width / (spaces.shape[0] - 1)))

    # Add question marks
    for i, p in enumerate(pos):
        left, low, right, high = spaces[p, :]
        if (right-left > average_width):
            right = left+average_width
        question_mark_im = q_mark_im.resize((right + 1 - left,high - low))
        question_mark = np.array(question_mark_im)
        question_mark[question_mark == 0] = col

        npim[low:high, left:right + 1] = question_mark
    im2 = Image.fromarray(npim)
    return im2


