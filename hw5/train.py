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
    hw5.get_wordvector(corpus) 
    print('[test wordvector]') 
    hw5.get_wordvector().most_similar(positive=['woman', 'king'], negative=['man']) 
    hw5.get_wordvector().most_similar(positive=['girl', 'father'], negative=['boy']) 
    print('[save wordvector]') 
    save_object(hw5, st.HW5_MODEL_PATH)
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
    print('[build rnn model]') 
    wv_rnn_model = wordvector_rnn_classifier(hw5.get_wordvector(), hw5.get_tokenizer(), hw5.max_document_size) 
    print('[start training]')
    train(model=wv_rnn_model, X=idseqpad, y=label, batch_size=st.BATCH_SIZE, epochs=st.EPOCHS, validation_split=st.VALIDATION_SPLIT, save_model_path=st.RNN_MODEL_CHECKPOINT_PATH)
    print('[done]') 
