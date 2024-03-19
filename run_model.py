from utils import *
from lookahead_pytorch import Lookahead

class runModel():
    """
    Class that scafolds the training and evaluation methods and attributes
    for each test case (Test case: Adam, Test case: Lookahead(Adam)).
    """
    def __init__(self, model, optimizer, args) -> None:
        """
        Init method to intialized instance of runModel.
        Args:
            - optimizer (pytorch optimizer): should be Adam
            - args (lookaheadArgs class instance): arguments for Lookahead
        Returns:
            - None
        """
        self.model = model
        self.lookahead = args.lookahead # whether to use lookahead or not

        # setting lookahead optimizer if True
        if self.lookahead:
            self.optimizer = Lookahead(optimizer, la_steps=args.la_steps,
                                       la_alpha=args.la_alpha,
                                       pullback_momentum=args.pullback_momentum)
        else:
            self.optimizer = optimizer

        # model training attributes
        self.time_elasped = 0.0
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.epoch_arr = []

    def _get_accuracy(self, data):
        """
        Method to compute accuracy per epoch for the training and validation data
        NOTE: Change as needed according to dataset

        Args:
            - data (DataLoader): training or validation data wrapped by DataLoader
        Returns:
            - curr_acc (float): training or validation accuracy for the current epoch
        """
        correct = 0
        total = 0

        for features, labels in data: # iterate over dataloader object

            # CUDA
            if torch.cuda.is_available():
                self.model.cuda()
                features = features.cuda()
                labels = labels.cuda()

            output = self.model(features) # get predictinos

            # select index with maximum prediction score
            # pred = output.max(1, keepdim=True)[1]
            correct += ((output.argmax(dim=1)) == labels.argmax(dim=1)).sum()
            total += features.shape[0]

        curr_acc = correct / total

        return curr_acc # return accuracy

    def _get_val_loss(self, val, criterion):
        """
        Method to compute validation loss per epoch
        Args:
            - val (DataLoader): validation data wrapped by DataLoader
            - criterion (pytorch method): loss function where default is MSE - should be changed as needed according to data
        Returns:
            - val_loss (float): validation loss for the current epoch
        """
        total_loss = 0.0

        for i, val_dataset in enumerate(val, 0):
            features, labels = val_dataset # Note: depends on how data is loaded in DataLoader

            # CUDA
            if torch.cuda.is_available():
                self.model.cuda()
                features = features.cuda()
                labels = labels.cuda()

            out = self.model(features) # compute predictions
            loss = criterion(out, labels.float()) # get loss

            # compute loss
            total_loss += loss.item()

        # compute average loss
        val_loss = float(total_loss) / (i + 1)

        return val_loss

    def train(self, train, val, criterion=nn.MSELoss(), epochs=10) -> None:
        """
        Train method that trains the model.
        Args:
            - train (DataLoader): train data wrapped by DataLoader
            - val (DataLoader): validation data wrapped by DataLoader
            - criterion (pytorch method): loss function where default is MSE - should be changed as needed according to data
            - epochs (integer): number of training epochs to run
        Returns:
            - None
        """
        torch.manual_seed(1000)

        start_time = time.time() # time elapsed

        print('Started Training')
        for epoch in range(epochs):  # loop over whole dataset
            total_train_loss = 0.0

            for i, dataset in enumerate(train, 0): # train in batches
                features, labels = dataset # Note: depends on how data is loaded in DataLoader

                # CUDA
                if torch.cuda.is_available():
                    self.model.cuda()
                    features = features.cuda()
                    labels = labels.cuda()

                self.optimizer.zero_grad() # zero parameter gradients

                # forward pass, backward pass, and optimize
                out = self.model(features)
                loss = criterion(out, labels.float())
                loss.backward()
                self.optimizer.step()

                # calculate training loss
                total_train_loss += loss.item()

            # get training and validation loss per epoch and store in array
            self.train_loss.append(float(total_train_loss) / (i + 1)) # average training loss per epoch
            self.val_loss.append(self._get_val_loss(val, criterion)) # average val loss per epoch

            # get training and validation accuracy per epoch and store in array
            self.train_acc.append(self._get_accuracy(train))
            self.val_acc.append(self._get_accuracy(val))

            self.epoch_arr.append(epoch + 1) # store current epoch + 1 for plotting

            # print training progress
            print(f'Epoch {self.epoch_arr[epoch]} | Train loss: {self.train_loss[epoch]} | \
                    Validation loss: {self.val_loss[epoch]} | Train accuracy: {self.train_acc[epoch]} | \
                    Validation accuracy: {self.val_acc[epoch]}')

        print('Finished Training')

        # getting elasped training time
        end_time = time.time()
        self.elapsed_time = end_time - start_time
        print(f'Total time elapsed: {self.elapsed_time:.3f} seconds')

        return

    def plot_loss() -> None:
        """
        Placeholder code: to be implemented by Edward
        """
        pass