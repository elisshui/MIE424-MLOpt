from utils import np, nn, torch, time

class runModel():
    """  
    Class that scafolds the training and evaluation methods and attributes
    for each test case (Test case: Adam, Test case: Lookahead(Adam)).
    """
    def __init__(self, model, optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

        # model training attributes
        self.time_elasped = 0.0
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.epoch_arr = []

    def _get_accuracy(self):
        pass

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
            labels, features = val_dataset

            # CUDA
            if torch.cuda.is_available():
                features = features.cuda()
                labels = labels.cuda()
            
            out = self.model(features) # compute predictions
            loss = criterion(out, labels) # get loss

            # compute loss
            total_loss += loss.item()
        
        # compute average loss
        val_loss = float(total_loss) / (i + 1)

        return val_loss

    def train(self, optimizer, train, val, criterion=nn.MSELoss(),
              epochs=10, batch_size=20, learning_rate=0.001) -> None:
        """ 
        Train method that trains the model.
        Args:
            - optimizer (pytorch optimizer): either Adam or Lookahead(Adam)
            - train (DataLoader): train data wrapped by DataLoader
            - val (DataLoader): validation data wrapped by DataLoader
            - criterion (pytorch method): loss function where default is MSE - should be changed as needed according to data
            - epochs (integer): number of training epochs to run
            - batch_size (integer): training batch size
            - learning_rate (float): training learning rate - defaults to 0.001
            - display_results (Boolean): whether to call result display methods
        Returns:
            - None
        """
        torch.manual_seed(1000)

        start_time = time.time() # time elapsed

        print('Started Training')
        for epoch in range(epochs):  # loop over whole dataset
            total_train_loss = 0.0

            for i, dataset in enumerate(train, 0): # train in batches
                labels, features = dataset

                # CUDA
                if torch.cuda.is_available():
                    features = features.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad() # zero parameter gradients

                # forward pass, backward pass, and optimize
                out = self.model(features)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

                # calculate training loss
                total_train_loss += loss.item()

            # get training and validation loss per epoch and store in array
            self.train_loss.append(float(total_train_loss) / (i + 1)) # average training loss per epoch
            self.val_loss.append(self._get_val_loss(val, criterion)) # average val loss per epoch

            # get training and validation accuracy per epoch and store in array
            self.train_acc.append(self._get_accuracy(train, batch_size=batch_size))
            self.val_acc.append(self._get_accuracy(val, batch_size=batch_size))

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