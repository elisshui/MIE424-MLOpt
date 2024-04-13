from utils import *
from lookahead_pytorch import Lookahead

class runModel():
    """
    Class that holds the training and evaluation methods as well as
    attributes for each experiment.
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

        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
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
        self.memory = []  # memory used per epoch

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
            #print(output)
            correct += ((output.argmax(dim=1)) == labels.argmax(dim=1)).sum().item()
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

    def train(self, train, val, criterion=nn.MSELoss(), epochs=200) -> None:
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

        start_time = time.time()  # time elapsed

        print('Started Training')
        for epoch in range(epochs):  # loop over whole dataset
            tracemalloc.start()  # start memory tracking
            
            total_train_loss = 0.0

            for i, dataset in enumerate(train, 0):  # train in batches
                features, labels = dataset  # Note: depends on how data is loaded in DataLoader

                # CUDA
                if torch.cuda.is_available():
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
            self.train_loss.append(float(total_train_loss) / (i + 1))  # average training loss per epoch
            self.val_loss.append(self._get_val_loss(val, criterion))  # average val loss per epoch

            # get training and validation accuracy per epoch and store in array
            self.train_acc.append(self._get_accuracy(train))
            self.val_acc.append(self._get_accuracy(val))

            self.epoch_arr.append(epoch + 1)  # store current epoch + 1 for plotting

            # print training progress
            print(f'Epoch {self.epoch_arr[epoch]} | Train loss: {self.train_loss[epoch]} | val loss: {self.val_loss[epoch]} | Train acc: {self.train_acc[epoch]} | val acc: {self.val_acc[epoch]}')
            
            self.memory.append(tracemalloc.get_traced_memory()[1])  # store memory used per epoch
            tracemalloc.stop()  # stop memory tracking
            
        print('Finished Training')

        # getting elasped training time
        end_time = time.time()
        self.elapsed_time = end_time - start_time
        print(f'Total time elapsed: {self.elapsed_time:.3f} seconds')

        return
    
    def save_experiement_data(self, filename):
        """ 
        Method to save the data stored from each experiment as a csv file.
        Args:
            - filename (String): the path in which the save the csv.
        Returns: None
        """
        experiment_data = pd.DataFrame({'epoch': self.epoch_arr,
                                        'train_loss': self.train_loss,
                                        'val_loss': self.val_loss,
                                        'train_acc': self.train_acc,
                                        'val_acc': self.val_acc,
                                        'memory': self.memory})

        experiment_data.to_csv(filename, index=False)

        print("Experiment data saved.")

    def plot_train_loss(self, *args):
        """
        Method to plot the training loss for all optimizers.
        Args:
            - Tuple with experiemnt object and label for the graph (model, "Model Training Loss")
        Returns: None
        """
        plt.figure(figsize=(10, 5))
        
        for arg in args:
            plt.plot(arg[0].epoch_arr, arg[0].train_loss, label=arg[1], color=arg[2], alpha=0.5)
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

        plt.savefig("results/training_loss.png")

        plt.show()
        
    def plot_val_loss(self, *args):
        """
        Method to plot the validation loss for all optimizers.
        Args:
            - Tuple with experiemnt object and label for the graph (model, "Model Training Loss")
        Returns: None
        """
        plt.figure(figsize=(10, 5))
        
        for arg in args:
            plt.plot(arg[0].epoch_arr, arg[0].val_loss, label=arg[1], color=arg[2], alpha=0.5)
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()

        plt.savefig("results/validation_loss.png")

        plt.show()
    
    def plot_mem_cost(self, *args):
        """
        Method to plot the memory usage for all optimizers.
        Args:
            - Tuple with experiemnt object and label for the graph (model, "Model Training Loss")
        Returns: None
        """
        plt.figure(figsize=(10, 5))
        
        for arg in args:
            plt.plot(arg[0].epoch_arr, arg[0].memory, label=arg[1], color=arg[2], alpha=0.5)
        
        plt.xlabel('Epochs')
        plt.ylabel('Memory usage')
        plt.title('Memory per Epoch')
        plt.legend()

        plt.savefig("results/memory_cost.png")

        plt.show()

    def plot_loss(self) -> None:
        """
        Plot the training loss and validation loss against the epoch in two different graphs.
        """
        # Plot the training loss
        plt.figure(figsize=(10, 5))
        plt.plot(self.epoch_arr, self.train_loss, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.show()

        # Plot the validation loss
        plt.figure(figsize=(10, 5))
        plt.plot(self.epoch_arr,self.val_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()
        plt.show()