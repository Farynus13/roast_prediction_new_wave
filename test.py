from training import get_scaler,prepare_variable_length_dataset,LSTMWithProperAttention \
    ,test_variable_length_predictions
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
    #args on call
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="path to model")
    parser.add_argument("--plot", help="plot data", default=False)

    args = parser.parse_args()

    plot = args.plot
    model_path = args.model_path

    #if using data directly from workspace
    path = 'data.npy'
    data = np.load(path, allow_pickle=True)
    print(data.shape)

    #plot
    # if plot:
    #     for roast in data:
    #         plt.plot(roast)
    #     plt.show()

    #constants
    n_features = 3
    max_length = 800
    start_idx = 150+75
    scaler = get_scaler(data)
    #shuffle
    np.random.shuffle(data)
    train_test_data = data[:int(0.9*len(data))]
    validation_data = data[int(0.9*len(data)):]

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMWithProperAttention(input_size=3, hidden_size=128, output_size=2)
    model.load_state_dict(torch.load(model_path))
    test_variable_length_predictions(model,scaler,validation_data,max_length,min_idx=start_idx,eval_split=1.0,chunks=1,show_plot=plot)
