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
    # print('[convert texts to splitted corpus]')
    # corpus = texts2corpus(texts) 
    
    print('[build tokenizer]')
    hw5.get_tokenizer(texts) 
    print('[save tokenizer]') 
    save_object(hw5, st.HW5_MODEL_PATH)  
    # print('[translate texts to idx sequence]') 
    # idseq = texts2idseq(ltexts, hw5.get_tokenizer()) 
    # print('[padding to the same length]') 
    # idseqpad = pad_idseq(idseq) 
    # print('[get the max document size]') 
    # m, max_document_size = idseqpad.shape 
    # print('[save max document size]')
    # hw5.max_document_size = max_document_size 
    # save_object(hw5, st.HW5_MODEL_PATH) 
    print('[get bow]') 
    tkn = hw5.get_tokenizer()
    tkn.num_words = st.BOW_MAX_NUM_WORDS 
    bow = tkn.texts_to_matrix(ltexts, mode='tfidf')
    m, n = bow.shape 
    print('[build bow model]') 
    bow_model = bow_classifier(input_shape=(n,)) 
    print('[start training]')
    train(model=bow_model, X=bow, y=label, batch_size=st.BATCH_SIZE, epochs=st.EPOCHS, validation_split=st.VALIDATION_SPLIT, save_model_path=st.BOW_MODEL_CHECKPOINT_PATH)
    print('[done]') 
