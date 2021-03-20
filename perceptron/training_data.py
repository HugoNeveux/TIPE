#!/usr/bin/python3
# coding: utf8

import random

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
VOWELS = "aeiouy"


def random_word():
    """
    Create string containing random letters, length between 1 and 40

    :return: str
    """
    lst = [random.choice(ALPHABET) for _ in range(random.randint(1, 40))]
    return "".join(lst)


def get_words_from_dict(n: int):
    path = "../Data/francais.txt"  # Unix-systems only
    words = []
    with open(path, 'r') as f:
        w_tmp = f.readlines()
        for _ in range(n):
            words.append(random.choice(w_tmp))
    return words
