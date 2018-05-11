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
    print('[train word2vec]') 
    wordvector = word2vec(corpus=corpus, dim=st.WORDVECTOR_DIM, window=st.WORDVECTOR_WINDOW, min_count=st.WORDVECTOR_MIN_COUNT, n_iter=st.WORDVECTOR_N_ITER)
    print('[save wordvector]') 
    hw5.wordvector = wordvector 
    save_object(hw5, st.HW5_MODEL_PATH)
    print('[build tokenizer]')
    tokenizer = build_tokenizer(texts) 
    print('[save tokenizer]') 
    hw5.tokenizer = tokenizer 
    save_object(hw5, st.HW5_MODEL_PATH)  
    print('[translate texts to idx sequence]') 
    idseq = texts2idseq(ltexts, tokenizer) 
    print('[padding to the same length]') 
    idseqpad = pad_idseq(idseq) 
    print('[get the max document size]') 
    m, max_document_size = idseqpad.shape 
    print('[save max document size]')
    hw5.max_document_size = max_document_size 
    save_object(hw5, st.HW5_MODEL_PATH)
    print('[build rnn model]') 
    wv_rnn_model = wordvector_rnn_classifier(wordvector, tokenizer, max_document_size) 
    print('[start training]')
    train(model=wv_rnn_model, X=idseqpad, y=label, batch_size=st.BATCH_SIZE, epochs=st.EPOCHS, validation_split=st.VALIDATION_SPLIT, save_model_path=st.RNN_MODEL_CHECKPOINT_PATH)
    print('[done]') 
