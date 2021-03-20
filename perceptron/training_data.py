#!/usr/bin/python3
# coding: utf8

import random

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
VOWELS = "aeiouy"


def random_word() -> str:
    """
    Create string containing random letters, length between 1 and 40

    :return: str
    """
    lst = [random.choice(ALPHABET) for _ in range(random.randint(1, 40))]
    return "".join(lst)


def get_words_from_dict(n: int, path="../data/francais-utf8.txt") -> list:
    """
    Get a random array of n words from dictionary

    Default path uses french dictionary

    :param n: int
    :param path: str
    :return: list
    """
    words = []
    with open(path, 'r', encoding="utf8") as f:
        w_tmp = f.readlines()
        for _ in range(n):
            words.append(random.choice(w_tmp))
    return words


def generate_dataset(length: int) -> list:
    """
    Generates dataset of chosen length as a shuffled list of tuples, containing random words and real words.

    Tuple composition is : ('word', 0 or 1). 0 stands for 'random word', 1 for 'real word'

    :param length: int
    :return: list
    """
    real_words = [(w, 1) for w in get_words_from_dict(int(length / 2))]
    random_words = [(random_word(), 0) for _ in range(int(length / 2))]

    res = real_words + random_words
    random.shuffle(res)

    return res


def vowel_rate(word: str) -> float:
    """
    Returns the vowel rate in the word given as parameter

    :param word:
    :return:
    """
    total = 0
    for char in word:
        if char in VOWELS:
            total += 1
    return total / len(word)