from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset, Lemma

synsets = wn.synsets('company')

item:Synset = synsets[0]
print("name : " , item.name())
print("Part of speech : " , item.pos())
print("wordNet integer offset ID : ", item.offset())
print("definition : ", item.definition() )
print("example : " , item.examples())
print("verb frame IDs : ", item.frame_ids())
print(" ---------- Lemmas --------")
lemmas: list[Lemma] = item.lemmas()
print("List of lemma onjects for this synset ", item.lemmas())
print(lemmas[0].also_sees())