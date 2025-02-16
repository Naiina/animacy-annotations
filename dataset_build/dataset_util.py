import re

def create_sentence(tokens):

    sentence = " ".join(tokens)

    for mark in ['.', ',', '!', '?', ';', ':', ')', ']', '}', '...', '"', '\'',  '”']:
        sentence = sentence.replace(f" {mark}", mark)

    for mark in ['(', '[', '{', '"', '\'', '“']:
        sentence = sentence.replace(f"{mark} ", mark)

    sentence = re.sub(r" \b'\b", "'", sentence)
    sentence = re.sub(r"\s'\w'", "'", sentence)

    sentence = re.sub(r"(\w)\s*-\s*(\d)", r"\1-\2", sentence)
    sentence = re.sub(r"(\d)\s*-\s*(\w)", r"\1-\2", sentence)
    sentence = re.sub(r"(\d)\s*-\s*(\d)", r"\1-\2", sentence)
    sentence = re.sub(r'\s"\s', '"', sentence)
    sentence = re.sub(r"\s'\s", "'", sentence)

    return sentence


def load_wn_map(BN_MAP_PATH) -> dict:
    with open(BN_MAP_PATH, 'r', encoding='utf-8') as m:
        items = m.readlines()

        wn_map = dict()
        for map in items:
            map = map.strip().split('\t')

            bn = map[0]
            wn = map[1:]
            wn_map[bn] = wn

        return wn_map
