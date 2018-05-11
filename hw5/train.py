import sys 
from util import * 

if __name__ == '__main__':
    label_file_path = sys.argv[1] 
    unlabel_file_path = sys.argv[2] 
    wordvector_file_path = sys.argv[3] 
    print('[load wordvector]')
    wordvector = load_object(wordvector_file_path) 
    print('[read label data]')
    label, ltexts = read_label_data(label_file_path) 
    print('[read unlabel data]')
    ultexts = read_unlabel_data(unlabel_file_path) 
    print('[concat texts]')
    texts = concat_data(ltexts, ultexts) 
    print('[build tokenizer]')
    tokenizer = build_tokenizer(texts) 
    print('[translate texts to idx sequence]') 
    idseq = texts2idseq(ltexts, tokenizer) 
    print('[padding to the same length]') 
    idseqpad = pad_idseq(idseq) 
    print('[get the max document size]') 
    m, max_document_size = idseqpad.shape
    print('[build rnn model]') 
    wv_rnn_model = wordvector_rnn_classifier(wordvector, tokenizer, max_document_size) 
    print('[start training]')
    train(model=wv_rnn_model, X=idseqpad, y=label, batch_size=128, epochs=200, validation_split=0.1, save_model_path=RNN_MODEL_CHECKPOINT_PATH)
    print('[save everything]') 
    hw5 = Hw5()
    hw5.max_document_size = max_document_size 
    hw5.wordvector = wordvector 
    hw5.tokenizer = tokenizer 
    hw5.rnn_model = load_model(RNN_MODEL_CHECKPOINT_PATH) 
    save_object(hw5, './hw5.pickle')
    
