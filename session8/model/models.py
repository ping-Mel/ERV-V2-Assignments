import torch
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
        Inner class representing an initial neural network architecture.
        """
        def __init__(self):
            super(Models.NetA, self).__init__()
            # Convolutional layers
            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 1    28      1   1    3      28     1    1  3
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 3    28      1   1    5      28     1    1  3
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 5    28      1   2    6      14     2    0  2
            self.pool1 = nn.MaxPool2d(2, 2)
            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 6    14      2   1    10     14     2    1  3
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 10    14     2   1    14     14     2    1  3
            self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 14    14     2   2    16     7     4     0  2
            self.pool2 = nn.MaxPool2d(2, 2)
            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 16    7      4   1    24     5     4     0  3
            self.conv5 = nn.Conv2d(256, 512, 3)
            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 24    5     4    1    32     3     4     0  3
            self.conv6 = nn.Conv2d(512, 1024, 3)
            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 32    3      4   1    40     1     4     0  3
            self.conv7 = nn.Conv2d(1024, 10, 3)

        def forward(self, x):
            x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
            x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
            x = F.relu(self.conv6(F.relu(self.conv5(x))))
            x = self.conv7(x)
            x = x.view(-1, 10) #1x1x10> 10
            return F.log_softmax(x, dim=-1)

    class NetB(nn.Module):
        """
        Inner class representing a simple neural network architecture with reduced the channel size.
        """
        def __init__(self):
            super(Models.NetB, self).__init__()
            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 1    28      1   1    3      26     1    0  3
            self.convblock1 = nn.Sequential(
              nn.Conv2d(1, 16, 3),
              nn.ReLU()
            )
            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 3    26      1   1    5      24     1    0  3
            self.convblock2 = nn.Sequential(
              nn.Conv2d(16, 32 , 3),
              nn.ReLU()
            )
            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 5    24      1   2    6      12     2    0  2
            self.pool1 = nn.MaxPool2d(2, 2)
            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 6    12      2   1    10       10     2    0  3
            self.convblock3 = nn.Sequential(
              nn.Conv2d(32, 64 , 3),
              nn.ReLU()
            )
            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 10    10      2   2    14    5     4     0  2
            self.pool2 = nn.MaxPool2d(2, 2)
            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 14    5      4   1    22    3     4     0  3
            self.convblock4 = nn.Sequential(
              nn.Conv2d(64, 32 , 3),
              nn.ReLU()
            )
            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 22    3      4   1    30    1     4     0  3
            self.convblock5 = nn.Sequential(
              nn.Conv2d(32, 10 , 3),
            )


        def forward(self, x):
          x = self.convblock1(x)
          x = self.convblock2(x)
          x = self.pool1(x)
          x = self.convblock3(x)
          x = self.pool2(x)
          x = self.convblock4(x)
          x = self.convblock5(x)
          x = x.view(-1, 10)
          return F.log_softmax(x, dim=-1)

    class NetC(nn.Module):
        """
        Inner class representing a simple neural network architecture with reduced the channel size.
        """
        def __init__(self):
            super(Models.NetC, self).__init__()
            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 1    28      1   1    3     28     1     1  3
            self.convblock1 = nn.Sequential(
              nn.Conv2d(1, 10, 3, padding=1),
              nn.BatchNorm2d(10),
              nn.ReLU(),
              nn.Dropout(0.25)
            )
            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 3    28      1   1    5     28     1     1  3
            self.convblock2 = nn.Sequential(
              nn.Conv2d(10, 20 , 3, padding=1),
              nn.BatchNorm2d(20),
              nn.ReLU(),
              nn.Dropout(0.25)
            )
            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 5    28      1   2    6     14     2     0  2
            self.pool1 = nn.MaxPool2d(2, 2)
            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 6    14      2   1    10     14     2    1  3
            self.convblock3 = nn.Sequential(
              nn.Conv2d(20, 10 , 3, padding=1),
              nn.BatchNorm2d(10),
              nn.ReLU(),
              nn.Dropout(0.25)
            )
            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 10    14      2   2    12     7     4    0  2
            self.pool2 = nn.MaxPool2d(2, 2)
            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 12    7      4   1    20     5     4    0  3
            self.convblock4 = nn.Sequential(
              nn.Conv2d(10, 10 , 3),
              nn.BatchNorm2d(10),
              nn.ReLU()
            )
            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 20    5      4   1    28     3     4    0  3
            self.convblock5 = nn.Sequential(
              nn.Conv2d(10, 10 , 3),
              nn.AdaptiveAvgPool2d(1)
            )


        def forward(self, x):
          x = self.convblock1(x)
          x = self.convblock2(x)
          x = self.pool1(x)
          x = self.convblock3(x)
          x = self.pool2(x)
          x = self.convblock4(x)
          x = self.convblock5(x)
          x = x.view(-1, 10)
          return F.log_softmax(x, dim=-1)
    class NetD(nn.Module):
        """
        Inner class representing a simple neural network architecture with reduced the channel size.
        """
        def __init__(self):
            super(Models.NetD, self).__init__()
            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 1    28      1   1    3     26     1     0  3
            self.convblock1 = nn.Sequential(
              nn.Conv2d(1, 16, 3),
              nn.BatchNorm2d(16),
              nn.ReLU(),
              nn.Dropout(0.1)
            )
            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 3    26      1   1    5     24     1     0  3
            self.convblock2 = nn.Sequential(
              nn.Conv2d(16, 16 , 3),
              nn.BatchNorm2d(16),
              nn.ReLU(),
              nn.Dropout(0.1)
            )

            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 5     24     1  1     5      23    1   0  1
            self.convblock3 = nn.Sequential(
              nn.Conv2d(16, 10 , 1),
              nn.BatchNorm2d(10),
              nn.ReLU(),
              nn.Dropout(0.1)
            )
            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 5    23      1   2    6      11     2     0  2
            self.pool1 = nn.MaxPool2d(2, 2)

            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 6    11      2   1     6     11     2    0  1
            self.convblock4 = nn.Sequential(
              nn.Conv2d(10, 10 , 1),
              nn.BatchNorm2d(10),
              nn.ReLU(),
              nn.Dropout(0.1)
            )

            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 6    11     2   2     8      5    4    0  2
            self.pool2 = nn.MaxPool2d(2, 2)

            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 8    5      4   1    16     5       4    1  3
            self.convblock5 = nn.Sequential(
              nn.Conv2d(10, 10 , 3, padding=1),
              nn.ReLU(),
              nn.BatchNorm2d(10),
              nn.Dropout(0.1)
            )

            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 16    5      4   1    24     5       4    1  3
            self.convblock6 = nn.Sequential(
              nn.Conv2d(10, 10 , 3, padding=1),
              nn.ReLU(),
              nn.BatchNorm2d(10),
              nn.Dropout(0.1)
            )

            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 24    5      4   1    32     3       4   0  3
            self.convblock7 = nn.Sequential(
              nn.Conv2d(10, 10 , 3),
              nn.ReLU(),
              nn.BatchNorm2d(10),
              nn.Dropout(0.1)
            )


            #R_in, N_in, j_in, S, R_out, N_out, J_out, P, K
            # 32    3      4   1    40     1       4   0  3
            self.convblock8 = nn.Sequential(
              nn.Conv2d(10, 10 , 3)
            )

            self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global pooling to reduce parameters

            self.classifier = nn.Linear(10, 10)  # Classifier


        def forward(self, x):
          x = self.convblock1(x)
          x = self.convblock2(x)
          x = self.convblock3(x)
          x = self.pool1(x)
          x = self.convblock4(x)
          x = self.pool2(x)
          x = self.convblock5(x)
          x = self.convblock6(x)
          x = self.convblock7(x)
          x = self.convblock8(x)
          x = self.global_avg_pool(x)
          x = x.view(-1, 10)
          x = self.classifier(x)
          return F.log_softmax(x, dim=-1)