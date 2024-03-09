class lookaheadArgs():
    def __init__(self, lookahead=False, la_steps=5, la_alpha=0.8, pullback_momentum="none") -> None:
        """  
        Init method to intialized instance of lookaheadArgs.
        Args:
            - lookahead (Boolean): True = use Lookahead; False = do not use Lookahead
            - la_steps (int): number of lookahead steps
            - la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer.
            - pullback_momentum (str): one of ["reset", "pullback", "none"]
        """
        self.lookahead = lookahead
        self.la_steps = la_steps
        self.la_alpha = la_alpha

        pullback_momentum = pullback_momentum.lower() # enforce lowercase
        assert pullback_momentum in ["reset", "pullback", "none"] # catch errors
        self.pullback_momentum = pullback_momentum  