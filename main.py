from utils import runModel

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
    pass

if __name__ == "__main__":
    main()
