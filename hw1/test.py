import numpy as np 
import pandas as pd 
import sys 

from DataManager import DataManager 
import setting as st 

def predict(theta, instance):
    h_theta = np.dot(instance, theta)
    return h_theta


def make_predictions(theta, testX, mu, sigma):
    results = []
    for i, tdf in testX.items():
        tdf = tdf.iloc[-st.CHUNK_SIZE:, :]

        instance_n = (tdf.values.flatten() - mu) / sigma

        results.append((i, predict(theta, np.insert(instance_n.reshape((1, -1)), obj=0, values=1, axis=1))[0, 0]))
    return np.array(results)

def write_results(results, output_file):
    df = pd.DataFrame(results, columns=['id', 'value'])
    df.to_csv(output_file, index=False)
    return df 

def main(input_file, output_file):
    dm = DataManager()
    dm.read_training_data(st.TRAINING_SET)
    dm.read_testing_data(input_file)
    testX = dm.select_testing_feature(st.SELECTED_FEATURE)

    theta = np.load(st.MODEL_NAME)
    mu, sigma = np.load(st.STATISTICS)
    print('mu', mu, 'sigma', sigma)

    results = make_predictions(theta, testX, mu, sigma)

    write_results(results, output_file)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])