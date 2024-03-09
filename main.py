from utils import runModel

def main():
    """ 
    Creates instance of model class, trains with different optimizers (Adam, Lookahead(Adam)), then displays results. 
        - Example use case:
            model = Model()
            opt = Lookahead(adam)
            run_model_lookahead = runModel(model, opt)
            run_model_lookahead.train()
            run_model_lookahead.display_loss()
    """
    pass

if __name__ == "__main__":
    main()