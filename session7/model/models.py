import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Models:
    """
    In this class, we organize our neural network architectures as nested/inner classes.
    This approach groups related functionalities and creates an organized and encapsulated
    code structure. Each neural network architecture is defined as an inner class within
    this Models class. This allows for easy instantiation and clear hierarchy of neural
    network models, each with its distinct architecture and characteristics.
    """
    @staticmethod
    def evaluate_model(model_class, input_size=(1, 28, 28)):
        """
        Static method to evaluate the model architecture.
        This method will print a summary of the model showing the layers and parameters.
        
        Parameters:
        model_class (class): The inner class representing the neural network architecture to evaluate.
        input_size (tuple): The size of the input to the model. Default is (1, 28, 28) for MNIST dataset.
        """
        # Check for CUDA availability and set the device accordingly
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        # Initialize the model from the inner class and move to the appropriate device
        model = model_class().to(device)

        # Print the summary of the model
        summary(model, input_size=input_size)

    

    class NetA(nn.Module):
        """
        Inner class representing a specific neural network architecture.
        """
        def __init__(self):
            super(Models.NetA, self).__init__()
            # Convolutional layers
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # Reduced filters to 8
            self.bn1 = nn.BatchNorm2d(8)
            self.conv2 = nn.Conv2d(8, 16, 3, padding=1) # Reduced filters to 16
            self.bn2 = nn.BatchNorm2d(16)
            self.pool = nn.MaxPool2d(2, 2)
            self.drop = nn.Dropout(0.25)

            self.conv3 = nn.Conv2d(16, 32, 3, padding=1) # Reduced filters to 32
            self.bn3 = nn.BatchNorm2d(32)
            # Fully connected layer
            # Adjust the number of input features to the FC layer based on the output size of the last conv layer
            self.fc1 = nn.Linear(32 * 7 * 7, 120)  # This needs adjustment
            self.fc2 = nn.Linear(120, 10)  # Final output layer
            

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.drop(x)
            x = F.relu(self.bn3(self.conv3(x)))

            # Flatten the output for the FC layer
            x = x.view(-1, 32 * 7 * 7)  # Adjust based on the actual output size
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
