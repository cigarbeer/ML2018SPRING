import numpy as np 
import pandas as pd 
import sys 
from util import * 

if __name__ == '__main__':
    label_file_path = sys.argv[1] 
    unlabel_file_path = sys.argv[2] 
    wordvector_file_path = sys.argv[3] 
    wordvector = load_object(wordvector_file_path)
    label, ltexts = read_label_data(label_file_path) 
    ultexts = read_unlabel_data(unlabel_file_path) 
    texts = concat_data(ltexts, ultexts) 
    tokenizer = build_tokenizer(texts) 
    idseq = texts2idseq(ltexts, tokenizer) 
    idseqpad = pad_idseq(idseq) 
    m, max_document_size = idseqpad.shape 
    wv_rnn_model = wordvector_rnn_classifier(wordvector, tokenizer, max_document_size) 
    train(model=wv_rnn_model, X=idseqpad, y=label, batch_size=128, epochs=1, validation_split=0.1, save_model_path='./rnn.hdf5')
    print(label_file_path, unlabel_file_path)