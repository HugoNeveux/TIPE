#!/usr/bin/python3
# coding: utf8

import random


def random_word():
    lst = [random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(random.randint(1, 26))]
    return "".join(lst)
