# EVALM
Entropy-guided Vocabulary Augmentation of Multilingual Language Models for Low-resource Tasks

##**To generate vocabulary for augmentation of desired size, use the below command:**


python3 evalm_vocab_selection.py [model name] [training file path] [minimum word frequency to consider] [generated vocab file] [Maximum vocab size to augment] [training file word translation dictionary] [entropy reduction percentage threshold]

Example:

python3 evalm_vocab_selection.py bert-base-multilingual-cased ./data/IITP_product_reviews/hi-train.csv 1 ./vocab_files/iitp_product_review_vocab_files/vocab_add_2000_evalm.txt 2000 ./vocab_files/iitp_product_review_vocab_files/iitp_word_trans.txt 25


##**To fine-tune the model with generated vocabulary, use the below command:**

python3 evalm_finetune.py [GPU] [train_file_path] [validation_file_path] [test_file_path] [saved_model_weight] [saved_model_prediction] [Seed] [model_name] [patience_count for early stopping] [vocab_file_to_augment] [final_vocab_file_after_augment] [embedding_initialization_technique,0:InitLRL,1:InitHRL,0.5:InitMix]


Example:

CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evalm_finetune.py cuda:0 ./data/IITP_product_reviews/hi-train.csv ./data/IITP_product_reviews/hi-valid.csv ./data/IITP_product_reviews/hi-test.csv iitp_evalm_ft.pt iitp_evalm_predictions.txt 1 bert-base-multilingual-cased 5 ./vocab_files/iitp_product_review_vocab_files/vocab_add_2000_evalm.txt iitp_vocab_augment.tmp 0.5


##**To fine-tune the model without vocabulary augmentation, use the below command:**

python3 normal_classification_finetune.py [GPU] [train_file_path] [validation_file_path] [test_file_path] [saved_model_weight] [saved_model_prediction] [Seed] [model_name] [patience_count for early stopping]


Example:

CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 normal_classification_finetune.py cuda:0 ./data/IITP_product_reviews/hi-train.csv ./data/IITP_product_reviews/hi-valid.csv ./data/IITP_product_reviews/hi-test.csv iitp_evalm_ft.pt iitp_evalm_predictions.txt 1 bert-base-multilingual-cased 5


##**To fine-tune the model without vocabulary augmentation & using flota, use the below command**

python3 flota_finetune.py [GPU] [train_file_path] [validation_file_path] [test_file_path] [saved_model_weight] [saved_model_prediction] [Seed] [model_name] [patience_count for early stopping]


Example:

CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 flota_finetune.py cuda:0 ./data/IITP_product_reviews/hi-train.csv ./data/IITP_product_reviews/hi-valid.csv ./data/IITP_product_reviews/hi-test.csv iitp_evalm_ft.pt iitp_evalm_predictions.txt 1 bert-base-multilingual-cased 5
