import sys 
import numpy as np 
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
    print(hw5.get_wordvector().most_similar(positive=['woman', 'king'], negative=['man'])) 
    print(hw5.get_wordvector().most_similar(positive=['girl', 'father'], negative=['boy'])) 
    print('[save wordvector]') 
    save_object(hw5, st.HW5_MODEL_PATH)
    print('[build tokenizer]')
    hw5.get_tokenizer(texts) 
    print('[save tokenizer]') 
    save_object(hw5, st.HW5_MODEL_PATH)  
    print('[translate label texts to idx sequence]') 
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
    print('[split validation data]') 
    (idseqpad, label), (val_idseqpad, val_label) = split_validation_set(X=idseqpad, y=label, rate=st.VALIDATION_SPLIT)
    print('[start training on label data]')
    train(model=wv_rnn_model, X=idseqpad, y=label, batch_size=st.BATCH_SIZE, epochs=st.EPOCHS, save_model_path=st.SEMISUPERVISED_RNN_MODEL_CHECKPOINT_PATH, validation_data=(val_idseqpad, val_label))
    print('[translate unlabel texts to idx sequence]')
    uidseq = texts2idseq(ultexts, hw5.get_tokenizer()) 
    print('[padding to the same length as labeled sequence]') 
    uidseqpad = pad_idseq(uidseq, max_len=hw5.max_document_size) 
    print('[semisupervised training]') 
    semimodel = None   
    for i in range(2): 
        print('[predict the label of unlabel data] iter: %d' % i) 
        semilabel, semiidseqpad = get_semisupervised_data(hw5.load_semisupervised_rnn_model(), uidseqpad, threshold=st.SEMISUPERVISED_THRESHOLD)
        print('[concatenate semisupervised data and label data] iter: %d' % i) 
        label_cat = np.concatenate((label, semilabel)) 
        idseqpad_cat = np.concatenate((idseqpad, semiidseqpad)) 
        print('[start training on semisupervised data] iter: %d' % i) 
        train(model=hw5.load_semisupervised_rnn_model(), X=idseqpad_cat, y=label_cat, batch_size=st.BATCH_SIZE, epochs=st.EPOCHS, save_model_path=st.SEMISUPERVISED_RNN_MODEL_CHECKPOINT_PATH, validation_data=(val_idseqpad, val_label))
    print('[done]') 
