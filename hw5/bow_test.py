import sys 
from util import * 
if __name__ == '__main__':
    testing_data_path = sys.argv[1] 
    prediction_file_path = sys.argv[2] 
    print('[read testing data]')
    t_id, ttexts = read_testing_data(testing_data_path) 
    print('[load hw5]')
    hw5 = load_object(st.HW5_MODEL_PATH) 
    print('[get bow]') 
    tkn = hw5.get_tokenizer()
    tkn.num_words = st.BOW_MAX_NUM_WORDS 
    bow = tkn.texts_to_matrix(ttexts, mode='tfidf')
    m, n = bow.shape 
    # print('[translate texts to idx sequence]') 
    # idseq = texts2idseq(ttexts, hw5.get_tokenizer()) 
    # print('[padding to the same length as training data]') 
    # idseqpad = pad_idseq(idseq, max_len=hw5.max_document_size) 
    print('[make prediction]') 
    pred = predict(model=hw5.load_bow_model(), t=bow, batch_size=st.BATCH_SIZE) 
    print('[write prediction]') 
    write_prediction(pred, prediction_file_path)  
    print('[done]')
