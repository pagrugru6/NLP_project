from nlp_lib import *
print("Generating BPE models")
nlp_ru = gen_bpe("../train/train_ru_sentences.txt", 2000)
nlp_ru.save('../checkpoint/bpe_model_ru.pkl')
print("Saved ru")
nlp_fi = gen_bpe("../train/train_fi_sentences.txt", 2000)
nlp_fi.save('../checkpoint/bpe_model_fi.pkl')
print("Saved fi")
nlp_ja = gen_bpe("../train/train_ja_sentences.txt", 2000)
nlp_ja.save('../checkpoint/bpe_model_ja.pkl')
print("Saved ja")
# nlp_en = gen_bpe("train/train_en_sentences.txt", 2000)


# nlp_en.save(../checkpoint/'bpe_model_en.pkl')
