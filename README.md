# EVALM
Entropy-guided Vocabulary Augmentation of Multilingual Language Models for Low-resource Tasks

To generate vocabulary for augmentation of desired size use the below command:
"python3 evalm_vocab_selection.py <model \name> <training \file path> <minimum \word frequency to consider> <generated \vocab file> <Maximum \vocab size to augment> <training \file word translation dictionary> <entropy \reduction percentage threshold>"
ex:
python3 evalm_vocab_selection.py bert-base-multilingual-cased ./data/iitp_product_review_data/hi-train.csv 1 ./vocab_files/iitp_product_review_vocab_files/vocab_add_2000_evalm.txt 2000 ./vocab_files/iitp_product_review_vocab_files/iitp_word_trans.txt 25
