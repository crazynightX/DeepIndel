# DeepIndel
An interpretable deep learning approach for predictingCRISPR/Cas9-mediated editing outcomes

Below is the layout of the whole model.
![DeepIndel](https://github.com/user-attachments/assets/0c219c17-4f4d-4106-947e-fdbafc88101b)
# Enviroment
* Keras 2.4.3
* TensorFlow-GPU 2.5.0
* transformers 4.30.2
# Datasets
Include 3 datasets:
* K562
* HEK293t
* T cell
# File description
* model.py: The DeepIndel model with BERT-based module.
* model_train.py: Running this file to train the DeepIndel model. (5-fold cross-validation)
* model_test.py: Running this file to evaluate the DeepIndel model.
* vocab.txt: token vocabulary when encoding sequences.
