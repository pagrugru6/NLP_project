from nlp_lib import *
nlp_en = gen_bpe("train/train_en_sentences.txt", 2000)
nlp_en.save('bpe_model.pkl')
