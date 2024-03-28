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
    df = pd.read_csv("training_set.csv")

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

    # running the adam model
    model_adam = LSTM()
    optimizer_1 = optim.Adam(model_adam.parameters(), lr=learning_rate)
    lookaheadArgs_1 = lookaheadArgs(lookahead=False)

    run_model_adam = runModel(model_adam, optimizer_1, lookaheadArgs_1)
    run_model_adam.train(train_loader, val_loader)
    run_model_adam.save_experiement_data("results/experiment_adam.csv")
    
    # running the lookahead model (pullback_momentum="None")
    model_look_none = LSTM()
    optimizer_2 = optim.Adam(model_look_none.parameters(), lr=learning_rate)
    lookaheadArgs_2 = lookaheadArgs(lookahead=True)

    run_model_look_none = runModel(model_look_none, optimizer_2, lookaheadArgs_2)
    run_model_look_none.train(train_loader, val_loader)
    run_model_look_none.save_experiement_data("results/experiment_LH_none.csv")

    # running the lookahead model (pullback_momentum="reset")
    model_look_reset = LSTM()
    optimizer_3 = optim.Adam(model_look_reset.parameters(), lr=learning_rate)
    lookaheadArgs_3 = lookaheadArgs(lookahead=True, pullback_momentum="reset")

    run_model_look_reset = runModel(model_look_reset, optimizer_3, lookaheadArgs_3)
    run_model_look_reset.train(train_loader, val_loader)
    run_model_look_reset.save_experiement_data("results/experiment_LH_reset.csv")


    # running the lookahead model (pullback_momentum="pullback")
    model_look_pull = LSTM()
    optimizer_4 = optim.Adam(model_look_pull.parameters(), lr=learning_rate)
    lookaheadArgs_4 = lookaheadArgs(lookahead=True, pullback_momentum="pullback")

    run_model_look_pull = runModel(model_look_pull, optimizer_4, lookaheadArgs_4)
    # run_model_look_pull.train(train_loader, val_loader)
    # run_model_look_reset.save_experiement_data("results/experiment_LH_pullback.csv")

    # plotting results
    run_model_adam.plot_train_loss((run_model_adam, "Adam Training Loss"),
                                   (run_model_look_none, "Lookahead Training Loss (pullback_momentum='None')"),
                                   (run_model_look_reset, "Lookahead Training Loss (pullback_momentum='reset')"),
                                   (run_model_look_pull, "Lookahead Training Loss (pullback_momentum='pullback')"))
    
    run_model_adam.plot_val_loss((run_model_adam, "Adam Validation Loss"),
                                 (run_model_look_none, "Lookahead Validation Loss (pullback_momentum='None')"),
                                (run_model_look_reset, "Lookahead Validation Loss (pullback_momentum='reset')"),
                                (run_model_look_pull, "Lookahead Validation Loss (pullback_momentum='pullback')"))
    
    run_model_adam.plot_mem_cost((run_model_adam, "Adam Memory Cost"),
                                 (run_model_look_none, "Lookahead Memory Cost (pullback_momentum='None')"),
                                 (run_model_look_reset, "Lookahead Memory Cost (pullback_momentum='reset')"),
                                 (run_model_look_pull, "Lookahead Memory Cost (pullback_momentum='pullback')"))
    
if __name__ == "__main__":
    main()