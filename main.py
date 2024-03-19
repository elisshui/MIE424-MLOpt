from utils import *
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from model import LSTM

def main():
    """
    Creates instance of model class, trains with different optimizers (Adam, Lookahead(Adam)), then displays results.
        - Example usage:
            model = Model()
            optimizer = optim.Adam(model.parameters())
            lookaheadArgs_1 = lookaheadArgs(lookahead=True)
            run_model_lookahead = runModel(model, optimizer, lookaheadArgs)
            run_model_lookahead.train()
            run_model_lookahead.plot_loss()
    """
    df = pd.read_csv("training_set.csv")
    data = torch.tensor(df.drop(columns=['filename', 'label']).values.astype(np.float32))
    le = preprocessing.LabelEncoder()
    label = le.fit_transform(df['label'].values)
    label = torch.nn.functional.one_hot(torch.tensor(label.astype(np.int64)))
    train_data, val_data, train_label, val_label = train_test_split(data, label, test_size=0.2, random_state=1, shuffle=True)
    train_dataset = TensorDataset(train_data, train_label)
    val_dataset = TensorDataset(val_data, val_label)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=True)

    #running the adam model
    model = LSTM()
    optimizer = optim.Adam(model.parameters())
    lookaheadArgs_1 = lookaheadArgs(lookahead=False)
    run_model = runModel(model, optimizer, lookaheadArgs_1)
    run_model.train(train_loader, val_loader)
    run_model.plot_loss() #generates graphs of both training and validation loss
    
    #running the lookahead model
    model2 = LSTM()
    lookaheadArgs_1 = lookaheadArgs(lookahead=True)
    run_model2 = runModel(model2, optimizer, lookaheadArgs_1)
    run_model2.train(train_loader, val_loader)
    run_model2.plot_loss() #generates graphs of both training and validation loss

if __name__ == "__main__":
    main()
