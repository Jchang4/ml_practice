import pickle
from nltk.corpus import gutenberg
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer

text = ""
for file_id in gutenberg.fileids():
    text += gutenberg.raw(file_id)
print(len(text)) # 11793318

trainer = PunktTrainer()
trainer.INCLUDE_ALL_COLLOCS = True
trainer.train(text)

tokenizer = PunktSentenceTokenizer(trainer.get_params())



pickle.dump(tokenizer, open('./data/sentence-tokenizer.pickle', 'wb'))