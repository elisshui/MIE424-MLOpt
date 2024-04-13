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

    # running the lookahead(Adam) model with weight decay1 0.5
    model_lh_adam_wd1 = LSTM()
    optimizer_lh_adam_wd1 = optim.Adam(model_lh_adam_wd1.parameters(), lr=0.001, weight_decay=0.001)
    lookaheadArgs_wd1 = lookaheadArgs(lookahead=True)

    run_model_lh_adam_wd1 = runModel(model_lh_adam_wd1, optimizer_lh_adam_wd1, lookaheadArgs_wd1)
    run_model_lh_adam_wd1.train(train_loader, val_loader)
    run_model_lh_adam_wd1.save_experiement_data("results/experiment_LH_wd1.csv")

    # running the lookahead(Adam) model with beta1 0.9
    model_lh_adam_wd2 = LSTM()
    optimizer_lh_adam_wd2 = optim.Adam(model_lh_adam_wd2.parameters(), lr=0.001, weight_decay=0.002)
    lookaheadArgs_wd2 = lookaheadArgs(lookahead=True)

    run_model_lh_adam_wd2 = runModel(model_lh_adam_wd2, optimizer_lh_adam_wd2, lookaheadArgs_wd2)
    run_model_lh_adam_wd2.train(train_loader, val_loader)
    run_model_lh_adam_wd2.save_experiement_data("results/experiment_LH_wd2.csv")

    # running the lookahead(Adam) model with beta1 0.98
    model_lh_adam_wd3 = LSTM()
    optimizer_lh_adam_wd3 = optim.Adam(model_lh_adam_wd3.parameters(), lr=0.001, weight_decay=0.003)
    lookaheadArgs_wd3 = lookaheadArgs(lookahead=True)

    run_model_lh_adam_wd3 = runModel(model_lh_adam_wd3, optimizer_lh_adam_wd3, lookaheadArgs_wd3)
    run_model_lh_adam_wd3.train(train_loader, val_loader)
    run_model_lh_adam_wd3.save_experiement_data("results/experiment_LH_wd3.csv")

    # running the adam model with beta1 0.5
    model_adam_wd1 = LSTM()
    optimizer_adam_wd1 = optim.Adam(model_adam_wd1.parameters(), lr=0.001, weight_decay=0.001)
    adamArgs_wd1 = lookaheadArgs(lookahead=False)

    run_model_adam_wd1 = runModel(model_adam_wd1, optimizer_adam_wd1, adamArgs_wd1)
    run_model_adam_wd1.train(train_loader, val_loader)
    run_model_adam_wd1.save_experiement_data("results/experiment_adam_wd1.csv")

    # running the adam model with beta1 0.9
    model_adam_wd2 = LSTM()
    optimizer_adam_wd2 = optim.Adam(model_adam_wd2.parameters(), lr=0.001, weight_decay=0.002)
    adamArgs_wd2 = lookaheadArgs(lookahead=False)

    run_model_adam_wd2 = runModel(model_adam_wd2, optimizer_adam_wd2, adamArgs_wd2)
    run_model_adam_wd2.train(train_loader, val_loader)
    run_model_adam_wd2.save_experiement_data("results/experiment_adam_wd2.csv")

    # running the adam model with beta1 0.98
    model_adam_wd3 = LSTM()
    optimizer_adam_wd3 = optim.Adam(model_adam_wd3.parameters(), lr=0.001, weight_decay=0.003)
    adamArgs_wd3 = lookaheadArgs(lookahead=False)

    run_model_adam_wd3 = runModel(model_adam_wd3, optimizer_adam_wd3, adamArgs_wd3)
    run_model_adam_wd3.train(train_loader, val_loader)
    run_model_adam_wd3.save_experiement_data("results/experiment_adam_wd3.csv")

    run_model_lh_adam_wd1.plot_train_loss((run_model_lh_adam_wd1, "Lookahead wd0.001 Training Loss", "red"),
                                            (run_model_lh_adam_wd2, "Lookahead wd0.002 Training Loss", "red"),
                                            (run_model_lh_adam_wd3, "Lookahead wd0.003 Training Loss", "red"),
                                            (run_model_adam_wd1, "Adam wd0.001 Training Loss", "green"),
                                            (run_model_adam_wd2, "Adam wd0.002 Training Loss", "green"),
                                            (run_model_adam_wd3, "Adam wd0.003 Training Loss", "green")
                                            )
    run_model_lh_adam_wd1.plot_val_loss((run_model_lh_adam_wd1, "Lookahead wd0.001 Validation Loss", "red"),
                                          (run_model_lh_adam_wd2, "Lookahead wd0.002 Validation Loss", "red"),
                                          (run_model_lh_adam_wd3, "Lookahead wd0.003 Validation Loss", "red"),
                                          (run_model_adam_wd1, "Adam wd0.001 Validation Loss", "green"),
                                          (run_model_adam_wd2, "Adam wd0.002 Validation Loss", "green"),
                                          (run_model_adam_wd3, "Adam wd0.003 Validation Loss", "green")
                                          )

    run_model_lh_adam_wd1.plot_mem_cost((run_model_lh_adam_wd1, "Lookahead wd0.001 Memory Cost", "red"),
                                          (run_model_lh_adam_wd2, "Lookahead wd0.002 Memory Cost", "red"),
                                          (run_model_lh_adam_wd3, "Lookahead wd0.003 Memory Cost", "red"),
                                          (run_model_adam_wd1, "Adam wd0.001 Memory Cost", "green"),
                                          (run_model_adam_wd2, "Adam wd0.002 Memory Cost", "green"),
                                          (run_model_adam_wd3, "Adam wd0.003 Memory Cost", "green")
                                          )

    # -------------------------------------------beta1 experiments----------------------------------
    # # running the lookahead(Adam) model with beta1 0.5
    # model_lh_adam_beta1 = LSTM()
    # optimizer_lh_adam_beta1 = optim.Adam(model_lh_adam_beta1.parameters(), lr=0.001, betas=(0.5, 0.999))
    # lookaheadArgs_beta1 = lookaheadArgs(lookahead=True)
    #
    # run_model_lh_adam_beta1 = runModel(model_lh_adam_beta1, optimizer_lh_adam_beta1, lookaheadArgs_beta1)
    # run_model_lh_adam_beta1.train(train_loader, val_loader)
    # run_model_lh_adam_beta1.save_experiement_data("results/experiment_LH_beta1.csv")
    #
    # # running the lookahead(Adam) model with beta1 0.9
    # model_lh_adam_beta2 = LSTM()
    # optimizer_lh_adam_beta2 = optim.Adam(model_lh_adam_beta2.parameters(), lr=0.001, betas=(0.9, 0.999))
    # lookaheadArgs_beta2 = lookaheadArgs(lookahead=True)
    #
    # run_model_lh_adam_beta2 = runModel(model_lh_adam_beta2, optimizer_lh_adam_beta2, lookaheadArgs_beta2)
    # run_model_lh_adam_beta2.train(train_loader, val_loader)
    # run_model_lh_adam_beta2.save_experiement_data("results/experiment_LH_beta2.csv")
    #
    # # running the lookahead(Adam) model with beta1 0.98
    # model_lh_adam_beta3 = LSTM()
    # optimizer_lh_adam_beta3 = optim.Adam(model_lh_adam_beta3.parameters(), lr=0.001, betas=(0.98, 0.999))
    # lookaheadArgs_beta3 = lookaheadArgs(lookahead=True)
    #
    # run_model_lh_adam_beta3 = runModel(model_lh_adam_beta3, optimizer_lh_adam_beta3, lookaheadArgs_beta3)
    # run_model_lh_adam_beta3.train(train_loader, val_loader)
    # run_model_lh_adam_beta3.save_experiement_data("results/experiment_LH_beta3.csv")
    #
    # # running the adam model with beta1 0.5
    # model_adam_beta1 = LSTM()
    # optimizer_adam_beta1 = optim.Adam(model_adam_beta1.parameters(), lr=0.001, betas=(0.5, 0.999))
    # adamArgs_beta1 = lookaheadArgs(lookahead=False)
    #
    # run_model_adam_beta1 = runModel(model_adam_beta1, optimizer_adam_beta1, adamArgs_beta1)
    # run_model_adam_beta1.train(train_loader, val_loader)
    # run_model_adam_beta1.save_experiement_data("results/experiment_adam_beta1.csv")
    #
    # # running the adam model with beta1 0.9
    # model_adam_beta2 = LSTM()
    # optimizer_adam_beta2 = optim.Adam(model_adam_beta2.parameters(), lr=0.001, betas=(0.9, 0.999))
    # adamArgs_beta2 = lookaheadArgs(lookahead=False)
    #
    # run_model_adam_beta2 = runModel(model_adam_beta2, optimizer_adam_beta2, adamArgs_beta2)
    # run_model_adam_beta2.train(train_loader, val_loader)
    # run_model_adam_beta2.save_experiement_data("results/experiment_adam_beta2.csv")
    #
    # # running the adam model with beta1 0.98
    # model_adam_beta3 = LSTM()
    # optimizer_adam_beta3 = optim.Adam(model_adam_beta3.parameters(), lr=0.001, betas=(0.98, 0.999))
    # adamArgs_wd3 = lookaheadArgs(lookahead=False)
    #
    # run_model_adam_beta3 = runModel(model_adam_beta3, optimizer_adam_beta3, adamArgs_wd3)
    # run_model_adam_beta3.train(train_loader, val_loader)
    # run_model_adam_beta3.save_experiement_data("results/experiment_adam_beta3.csv")
    #
    # run_model_lh_adam_beta1.plot_train_loss((run_model_lh_adam_beta1, "Lookahead beta1 0.5 Training Loss", "red"),
    #                                       (run_model_lh_adam_beta2, "Lookahead beta1 0.9 Training Loss", "red"),
    #                                       (run_model_lh_adam_beta3, "Lookahead beta1 0.98 Training Loss", "red"),
    #                                       (run_model_adam_beta1, "Adam beta1 0.5 Training Loss", "green"),
    #                                       (run_model_adam_beta2, "Adam beta1 0.9 Training Loss", "green"),
    #                                       (run_model_adam_beta3, "Adam beta1 0.98 Training Loss", "green")
    #                                       )
    # run_model_lh_adam_beta1.plot_val_loss((run_model_lh_adam_beta1, "Lookahead beta1 0.5 Validation Loss", "red"),
    #                                     (run_model_lh_adam_beta2, "Lookahead beta1 0.9 Validation Loss", "red"),
    #                                     (run_model_lh_adam_beta3, "Lookahead beta1 0.98 Validation Loss", "red"),
    #                                     (run_model_adam_beta1, "Adam beta1 0.5 Validation Loss", "green"),
    #                                     (run_model_adam_beta2, "Adam beta1 0.9 Validation Loss", "green"),
    #                                     (run_model_adam_beta3, "Adam beta1 0.98 Validation Loss", "green")
    #                                     )
    #
    # run_model_lh_adam_beta1.plot_mem_cost((run_model_lh_adam_beta1, "Lookahead beta1 0.5 Memory Cost", "red"),
    #                                     (run_model_lh_adam_beta2, "Lookahead beta1 0.9 Memory Cost", "red"),
    #                                     (run_model_lh_adam_beta3, "Lookahead beta1 0.98 Memory Cost", "red"),
    #                                     (run_model_adam_beta1, "Adam beta1 0.5 Memory Cost", "green"),
    #                                     (run_model_adam_beta2, "Adam beta1 0.9 Memory Cost", "green"),
    #                                     (run_model_adam_beta3, "Adam beta1 0.98 Memory Cost", "green")
    #                                     )

    # -------------------------------------------learning rate experiments----------------------------------
    # # running the lookahead(Adam) model with lr 0.001
    # model_lh_adam_lr1 = LSTM()
    # optimizer_lh_adam_lr1 = optim.Adam(model_lh_adam_lr1.parameters(), lr=0.001)
    # lookaheadArgs_lr1 = lookaheadArgs(lookahead=True)
    #
    # run_model_lh_adam_lr1 = runModel(model_lh_adam_lr1, optimizer_lh_adam_lr1, lookaheadArgs_lr1)
    # run_model_lh_adam_lr1.train(train_loader, val_loader)
    # run_model_lh_adam_lr1.save_experiement_data("results/experiment_LH_lr1.csv")
    #
    # # running the lookahead(Adam) model with lr 0.002
    # model_lh_adam_lr2 = LSTM()
    # optimizer_lh_adam_lr2 = optim.Adam(model_lh_adam_lr2.parameters(), lr=0.002)
    # lookaheadArgs_lr2 = lookaheadArgs(lookahead=True)
    #
    # run_model_lh_adam_lr2 = runModel(model_lh_adam_lr2, optimizer_lh_adam_lr2, lookaheadArgs_lr2)
    # run_model_lh_adam_lr2.train(train_loader, val_loader)
    # run_model_lh_adam_lr2.save_experiement_data("results/experiment_LH_lr2.csv")
    #
    # # running the lookahead(Adam) model with lr 0.004
    # model_lh_adam_lr3 = LSTM()
    # optimizer_lh_adam_lr3 = optim.Adam(model_lh_adam_lr3.parameters(), lr=0.004)
    # lookaheadArgs_lr3 = lookaheadArgs(lookahead=True)
    #
    # run_model_lh_adam_lr3 = runModel(model_lh_adam_lr3, optimizer_lh_adam_lr3, lookaheadArgs_lr3)
    # run_model_lh_adam_lr3.train(train_loader, val_loader)
    # run_model_lh_adam_lr3.save_experiement_data("results/experiment_LH_lr3.csv")
    #
    # # running the adam model with lr 0.001
    # model_adam_lr1 = LSTM()
    # optimizer_adam_lr1 = optim.Adam(model_adam_lr1.parameters(), lr=0.001)
    # adamArgs_lr1 = lookaheadArgs(lookahead=False)
    #
    # run_model_adam_lr1 = runModel(model_adam_lr1, optimizer_adam_lr1, adamArgs_lr1)
    # run_model_adam_lr1.train(train_loader, val_loader)
    # run_model_adam_lr1.save_experiement_data("results/experiment_adam_lr1.csv")
    #
    # # running the adam model with lr 0.002
    # model_adam_lr2 = LSTM()
    # optimizer_adam_lr2 = optim.Adam(model_adam_lr2.parameters(), lr=0.002)
    # adamArgs_lr2 = lookaheadArgs(lookahead=False)
    #
    # run_model_adam_lr2 = runModel(model_adam_lr2, optimizer_adam_lr2, adamArgs_lr2)
    # run_model_adam_lr2.train(train_loader, val_loader)
    # run_model_adam_lr2.save_experiement_data("results/experiment_adam_lr2.csv")
    #
    # # running the adam model with lr 0.004
    # model_adam_lr3 = LSTM()
    # optimizer_adam_lr3 = optim.Adam(model_adam_lr3.parameters(), lr=0.004)
    # adamArgs_lr3 = lookaheadArgs(lookahead=False)
    #
    # run_model_adam_lr3 = runModel(model_adam_lr3, optimizer_adam_lr3, adamArgs_lr3)
    # run_model_adam_lr3.train(train_loader, val_loader)
    # run_model_adam_lr3.save_experiement_data("results/experiment_adam_lr3.csv")
    #
    #
    #
    # run_model_lh_adam_lr1.plot_train_loss((run_model_lh_adam_lr1, "Lookahead lr0.001 Training Loss", "red"),
    #                                       (run_model_lh_adam_lr2, "Lookahead lr0.002 Training Loss", "red"),
    #                                       (run_model_lh_adam_lr3, "Lookahead lr0.003 Training Loss", "red"),
    #                                       (run_model_adam_lr1, "Adam lr0.001 Training Loss", "green"),
    #                                       (run_model_adam_lr2, "Adam lr0.002 Training Loss", "green"),
    #                                       (run_model_adam_lr3, "Adam lr0.004 Training Loss", "green")
    #                                       )
    # run_model_lh_adam_lr1.plot_val_loss((run_model_lh_adam_lr1, "Lookahead lr0.001 Validation Loss", "red"),
    #                                     (run_model_lh_adam_lr2, "Lookahead lr0.002 Validation Loss", "red"),
    #                                     (run_model_lh_adam_lr3, "Lookahead lr0.003 Validation Loss", "red"),
    #                                     (run_model_adam_lr1, "Adam lr0.001 Validation Loss", "green"),
    #                                     (run_model_adam_lr2, "Adam lr0.002 Validation Loss", "green"),
    #                                     (run_model_adam_lr3, "Adam lr0.004 Validation Loss", "green")
    #                                     )
    #
    # run_model_lh_adam_lr1.plot_mem_cost((run_model_lh_adam_lr1, "Lookahead lr0.001 Memory Cost", "red"),
    #                                     (run_model_lh_adam_lr2, "Lookahead lr0.002 Memory Cost", "red"),
    #                                     (run_model_lh_adam_lr3, "Lookahead lr0.003 Memory Cost", "red"),
    #                                     (run_model_adam_lr1, "Adam lr0.001 Memory Cost", "green"),
    #                                     (run_model_adam_lr2, "Adam lr0.002 Memory Cost", "green"),
    #                                     (run_model_adam_lr3, "Adam lr0.003 Memory Cost", "green")
    #                                     )



    # ----------------------------------- Optimizer experiment -------------------------------------------------------

    #
    # learning_rate = 0.001
    #
    # # running the lookahead(Adam) model
    # model_lh_adam = LSTM()
    # optimizer_lh_adam = optim.Adam(model_lh_adam.parameters(), lr=learning_rate)
    # lookaheadArgs_2 = lookaheadArgs(lookahead=True)
    #
    # run_model_lh_adam = runModel(model_lh_adam, optimizer_lh_adam, lookaheadArgs_2)
    # run_model_lh_adam.train(train_loader, val_loader)
    # run_model_lh_adam.save_experiement_data("results/experiment_LH.csv")
    #
    # # running the adam model
    # model_adam = LSTM()
    # optimizer_adam = optim.Adam(model_adam.parameters(), lr=learning_rate)
    # lookaheadArgs_1 = lookaheadArgs(lookahead=False)
    #
    # run_model_adam = runModel(model_adam, optimizer_adam, lookaheadArgs_1)
    # run_model_adam.train(train_loader, val_loader)
    # run_model_adam.save_experiement_data("results/experiment_adam.csv")
    #
    # # running the SGD model
    # model_SGD = LSTM()
    # optimizer_SGD = optim.SGD(model_SGD.parameters(), lr=learning_rate)
    # lookaheadArgs_3 = lookaheadArgs(lookahead=False)
    #
    # run_model_SGD = runModel(model_SGD, optimizer_SGD, lookaheadArgs_3)
    # run_model_SGD.train(train_loader, val_loader)
    # run_model_SGD.save_experiement_data("results/experiment_SGD.csv")
    #
    # # running the Adadelta model
    # model_adadelta = LSTM()
    # optimizer_adadelta = optim.Adadelta(model_adadelta.parameters(), lr=learning_rate)
    # lookaheadArgs_4 = lookaheadArgs(lookahead=False)
    #
    # run_model_adadelta = runModel(model_adadelta, optimizer_adadelta, lookaheadArgs_4)
    # run_model_adadelta.train(train_loader, val_loader)
    # run_model_adadelta.save_experiement_data("results/experiment_ADADelta.csv")
    #
    # # running the Adagrad model
    # if torch.cuda.is_available():
    #     model_adagrad = LSTM().cuda()
    # else:
    #     model_adagrad = LSTM()
    # optimizer_adagrad = optim.Adagrad(model_adagrad.parameters(), lr=learning_rate)
    # lookaheadArgs_5 = lookaheadArgs(lookahead=False)
    #
    # run_model_adagrad = runModel(model_adagrad, optimizer_adagrad, lookaheadArgs_5)
    # run_model_adagrad.train(train_loader, val_loader)
    # run_model_adagrad.save_experiement_data("results/experiment_ADAGrad.csv")
    #
    # # running the RMSprop model
    # model_RMSprop = LSTM()
    # optimizer_RMSprop = optim.RMSprop(model_RMSprop.parameters(), lr=learning_rate)
    # lookaheadArgs_6 = lookaheadArgs(lookahead=False)
    #
    # run_model_RMSprop = runModel(model_RMSprop, optimizer_RMSprop, lookaheadArgs_6)
    # run_model_RMSprop.train(train_loader, val_loader)
    # run_model_RMSprop.save_experiement_data("results/experiment_RMSprop.csv")
    #
    # # plotting results
    # run_model_adam.plot_train_loss((run_model_adam, "Adam Training Loss", "blue"),
    #                                (run_model_lh_adam, "Lookahead(Adam) Training Loss", "red"),
    #                                (run_model_SGD, "SGD Loss", "green"),
    #                                (run_model_adadelta, "ADAdelta Training Loss", "purple"),
    #                                (run_model_adagrad, "ADAgrad Training Loss", "orange"),
    #                                (run_model_RMSprop, "RMSprop Training Loss", "pink"))
    #
    # run_model_adam.plot_val_loss((run_model_adam, "Adam Validation Loss", "blue"),
    #                              (run_model_lh_adam, "Lookahead(Adam) Validation Loss", "red"),
    #                              (run_model_SGD, "SGD Loss", "green"),
    #                              (run_model_adadelta, "ADAdelta Validation Loss", "purple"),
    #                              (run_model_adagrad, "ADAgrad Validation Loss", "orange"),
    #                              (run_model_RMSprop, "RMSprop Validation Loss", "pink"))
    #
    # run_model_adam.plot_mem_cost((run_model_adam, "Adam Memory Cost", "blue"),
    #                              (run_model_lh_adam, "Lookahead(Adam) Memory Cost", "red"),
    #                              (run_model_SGD, "SGD Memory Cost", "green"),
    #                              (run_model_adadelta, "ADAdelta Memory Cost", "purple"),
    #                              (run_model_adagrad, "ADAgrad Memory Cost", "orange"),
    #                              (run_model_RMSprop, "RMSprop Memory Cost", "pink"))
    
if __name__ == "__main__":
    main()