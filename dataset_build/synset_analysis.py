from collections.abc import Iterable
from nltk.corpus import wordnet as wn

human = [
    wn.synset('person.n.01'),
    wn.synset('operator.n.02'),
    wn.synset('teacher.n.02'),
    wn.synset('kin.n.02'),
    wn.synset('people.n.01'),
    wn.synset('enemy.n.01')
]

animate = [
    wn.synset('living_thing.n.01'),
    wn.synset('biological_group.n.01'),
    wn.synset('spiritual_being.n.01'),
    wn.synset('imaginary_being.n.01')
]

def is_syn_an(syn):
    paths = syn.hypernym_paths()

    path = sorted(paths, key=lambda p: [s.name() for s in p])[0]

    if wn.synset('genotype.n.01') in path:
        return False
    if any(s.name() in [h.name() for h in human] or s.name() in [a.name() for a in animate] for s in path):
        return True
    return False

def which_an(syn):
    paths = syn.hypernym_paths()
    path = sorted(paths, key=lambda p: [s.name() for s in p])[0]

    for s in path:
        if s == wn.synset('people.n.01') or s == wn.synset('priest.n.01'):
            return 'H'
        if s in animate:
            return 'A'
    return 'H'

def is_group(syn):
    paths = syn.hypernym_paths()
    path = sorted(paths, key=lambda p: [s.name() for s in p])[0]
    if any(s.name() == 'group.n.01' or s.name() == 'taxonomic_group.n.01' for s in path):
        return True
    return False



