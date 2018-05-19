import sys 
from util import * 

if __name__ == '__main__':
    label_file_path = sys.argv[1] 
    unlabel_file_path = sys.argv[2] 
    print('[instantiate hw5]') 
    hw5 = Hw5()
    print('[read label data]')
    label, ltexts = read_label_data(label_file_path) 
    print('[read unlabel data]')
    ultexts = read_unlabel_data(unlabel_file_path) 
    print('[concat texts]')
    texts = concat_data(ltexts, ultexts) 
    print('[convert texts to splitted corpus]')
    corpus = texts2corpus(texts) 
    
    print('[build tokenizer]')
    hw5.get_tokenizer(texts) 
    print('[save tokenizer]') 
    save_object(hw5, st.HW5_MODEL_PATH)  
    print('[translate texts to idx sequence]') 
    idseq = texts2idseq(ltexts, hw5.get_tokenizer()) 
    print('[padding to the same length]') 
    idseqpad = pad_idseq(idseq) 
    print('[get the max document size]') 
    m, max_document_size = idseqpad.shape 
    print('[save max document size]')
    hw5.max_document_size = max_document_size 
    save_object(hw5, st.HW5_MODEL_PATH) 
    print('[get bow]') 
    bow = hw5.get_tokenizer().texts_to_matrix(ltexts, mode='tfidf')
    print('[build bow model]') 
    wv_rnn_model = bow_classifier(input_shape=(hw5.max_document_size,)) 
    print('[start training]')
    train(model=wv_rnn_model, X=idseqpad, y=label, batch_size=st.BATCH_SIZE, epochs=st.EPOCHS, validation_split=st.VALIDATION_SPLIT, save_model_path=st.RNN_MODEL_CHECKPOINT_PATH)
    print('[done]') 
