# EVALM
Entropy-guided Vocabulary Augmentation of Multilingual Language Models for Low-resource Tasks

#To generate vocabulary for augmentation of desired size use the below command:
python3 evalm_vocab_selection.py [model name] [training file path] [minimum word frequency to consider] [generated vocab file] [Maximum vocab size to augment] [training file word translation dictionary] [entropy reduction percentage threshold]

Example:
python3 evalm_vocab_selection.py bert-base-multilingual-cased ./data/iitp_product_review_data/hi-train.csv 1 ./vocab_files/iitp_product_review_vocab_files/vocab_add_2000_evalm.txt 2000 ./vocab_files/iitp_product_review_vocab_files/iitp_word_trans.txt 25

#To fine-tune model with generated vocabulary use the below command:

Example:
python3 evalm_finetune.py cuda:0 ./data/iitp_product_review_data/hi-train.csv ./data/iitp_product_review_data/hi-valid.csv ./data/iitp_product_review_data/hi-test.csv iitp_evalm_ft.pt iitp_evalm_predictions.txt 1 bert-base-multilingual-cased 5 ./vocab_files/iitp_product_review_vocab_files/vocab_add_2000_evalm.txt iitp_vocab_augment.tmp 0.5
