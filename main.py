from utils import *

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
    # reading the data
    df = pd.read_csv("data/training_set.csv")

    # processing the data
    data = torch.tensor(df.drop(columns=['filename', 'label']).values.astype(np.float32))
    le = preprocessing.LabelEncoder()
    label = le.fit_transform(df['label'].values)
    label = torch.nn.functional.one_hot(torch.tensor(label.astype(np.int64)))

    # splitting the data
    train_data, val_data, train_label, val_label = train_test_split(data, label, test_size=0.2, random_state=1, shuffle=True)
    train_dataset = TensorDataset(train_data, train_label)
    val_dataset = TensorDataset(val_data, val_label)

    # converting to DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=True)
   
    # -----------------EXPERIMENTS BEGIN HERE-----------------------------------

    learning_rate = 0.001

    # running the lookahead(Adam) model
    model_lh_adam = LSTM()
    optimizer_lh_adam = optim.Adam(model_lh_adam.parameters(), lr=learning_rate)
    lookaheadArgs_2 = lookaheadArgs(lookahead=True)

    run_model_lh_adam = runModel(model_lh_adam, optimizer_lh_adam, lookaheadArgs_2)
    run_model_lh_adam.train(train_loader, val_loader)
    run_model_lh_adam.save_experiement_data("results/experiment_LH.csv")

    # running the adam model
    model_adam = LSTM()
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=learning_rate)
    lookaheadArgs_1 = lookaheadArgs(lookahead=False)

    run_model_adam = runModel(model_adam, optimizer_adam, lookaheadArgs_1)
    run_model_adam.train(train_loader, val_loader)
    run_model_adam.save_experiement_data("results/experiment_adam.csv")

    # running the SGD model
    model_SGD = LSTM()
    optimizer_SGD = optim.SGD(model_SGD.parameters(), lr=learning_rate)
    lookaheadArgs_3 = lookaheadArgs(lookahead=False)

    run_model_SGD = runModel(model_SGD, optimizer_SGD, lookaheadArgs_3)
    run_model_SGD.train(train_loader, val_loader)
    run_model_SGD.save_experiement_data("results/experiment_SGD.csv")

    # running the Adadelta model
    model_adadelta = LSTM()
    optimizer_adadelta = optim.Adadelta(model_adadelta.parameters(), lr=learning_rate)
    lookaheadArgs_4 = lookaheadArgs(lookahead=False)

    run_model_adadelta = runModel(model_adadelta, optimizer_adadelta, lookaheadArgs_4)
    run_model_adadelta.train(train_loader, val_loader)
    run_model_adadelta.save_experiement_data("results/experiment_ADADelta.csv")

    # running the Adagrad model
    model_adagrad = LSTM()
    optimizer_adagrad = optim.Adagrad(model_adagrad.parameters(), lr=learning_rate)
    lookaheadArgs_5 = lookaheadArgs(lookahead=False)

    run_model_adagrad = runModel(model_adagrad, optimizer_adagrad, lookaheadArgs_5)
    run_model_adagrad.train(train_loader, val_loader)
    run_model_adagrad.save_experiement_data("results/experiment_ADAGrad.csv")

    # running the RMSprop model
    model_RMSprop = LSTM()
    optimizer_RMSprop = optim.RMSprop(model_RMSprop.parameters(), lr=learning_rate)
    lookaheadArgs_6 = lookaheadArgs(lookahead=False)

    run_model_RMSprop = runModel(model_RMSprop, optimizer_RMSprop, lookaheadArgs_6)
    run_model_RMSprop.train(train_loader, val_loader)
    run_model_RMSprop.save_experiement_data("results/experiment_RMSprop.csv")

    # plotting results
    run_model_adam.plot_train_loss((run_model_adam, "Adam Training Loss", "blue"),
                                   (run_model_lh_adam, "Lookahead(Adam) Training Loss", "red"),
                                   (run_model_SGD, "SGD Loss", "green"),
                                   (run_model_adadelta, "ADAdelta Training Loss", "purple"),
                                   (run_model_adagrad, "ADAgrad Training Loss", "orange"),
                                   (run_model_RMSprop, "RMSprop Training Loss", "pink"))
    
    run_model_adam.plot_val_loss((run_model_adam, "Adam Validation Loss", "blue"),
                                 (run_model_lh_adam, "Lookahead(Adam) Validation Loss", "red"),
                                 (run_model_SGD, "SGD Loss", "green"),
                                 (run_model_adadelta, "ADAdelta Validation Loss", "purple"),
                                 (run_model_adagrad, "ADAgrad Validation Loss", "orange"),
                                 (run_model_RMSprop, "RMSprop Validation Loss", "pink"))
    
    run_model_adam.plot_mem_cost((run_model_adam, "Adam Memory Cost", "blue"),
                                 (run_model_lh_adam, "Lookahead(Adam) Memory Cost", "red"),
                                 (run_model_SGD, "SGD Memory Cost", "green"),
                                 (run_model_adadelta, "ADAdelta Memory Cost", "purple"),
                                 (run_model_adagrad, "ADAgrad Memory Cost", "orange"),
                                 (run_model_RMSprop, "RMSprop Memory Cost", "pink"))
    
if __name__ == "__main__":
    main()