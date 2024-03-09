from utils import runModel

def main():
    """ 
    Creates instance of model class, trains with different optimizers (Adam, Lookahead(Adam)), then displays results. 
        - Example usage:
            model = Model()
            opt_lookahead = Lookahead(adam)
            run_model_lookahead = runModel(model, opt_lookahead)
            run_model_lookahead.train()
            run_model_lookahead.plot_loss()
    """
    pass

if __name__ == "__main__":
    main()