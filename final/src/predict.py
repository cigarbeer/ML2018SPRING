import os 

def predict(model, generator): 
    prediction = model.predict_generator( 
        generator=generator, 
        steps=1, 
        max_queue_size=10, 
        workers=os.cpu_count(), 
        use_multiprocessing=True, 
        verbose=1 
    ) 
    return prediction 

