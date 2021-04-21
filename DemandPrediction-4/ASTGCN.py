import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device=torch.device("cuda:0")
class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class Temporal_Attention_layer(nn.Module):
    """
    compute temporal attention scores
    """

    def __init__(self, num_of_vertices, num_of_features, num_of_timesteps):
        """
        Temporal Attention Layer
        :param num_of_vertices: int
        :param num_of_features: int
        :param num_of_timesteps: int
        """
        super(Temporal_Attention_layer, self).__init__()

        self.U_1 = torch.randn(num_of_vertices, requires_grad=True).cuda()
        self.U_2 = torch.randn(5, num_of_timesteps, requires_grad=True).cuda()      #改过
        self.U_3 = torch.randn(60, requires_grad=True).cuda()                       #改过
        self.b_e = torch.randn(1, num_of_timesteps, num_of_timesteps, requires_grad=True).cuda()
        self.V_e = torch.randn(num_of_timesteps, num_of_timesteps, requires_grad=True).cuda()

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.tensor, x^{(r - 1)}_h
                       shape is (batch_size, V, C_{r-1}, T_{r-1})

        Returns
        ----------
        E_normalized: torch.tensor, S', spatial attention scores
                      shape is (batch_size, T_{r-1}, T_{r-1})

        """
        # _, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # N == batch_size
        # V == num_of_vertices
        # C == num_of_features
        # T == num_of_timesteps

        # compute temporal attention scores
        # shape of lhs is (N, T, V)

        # lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U_1),self.U_2)
        s=torch.matmul(x, self.U_1)#.permute(0, 3, 1, 2)
        lhs = torch.matmul(torch.matmul(x, self.U_1), self.U_2)    #.permute(0, 3, 1, 2)     #改过

        # shape is (batch_size, V, T)
        # rhs = torch.matmul(self.U_3, x.transpose((2, 0, 1, 3)))
        rhs = torch.matmul(x, self.U_3)  # Is it ok to switch the position?.permute((0, 1, 3, 2))

        # lhs = lhs.permute(0, 2, 1)      #改过
        # rhs = rhs.permute(2, 1, 0)

        product = torch.matmul(lhs.permute(1,0), rhs)  # wd: (batch_size, T, T)

        # (batch_size, T, T)
        E = torch.matmul(self.V_e, torch.sigmoid(product + self.b_e))

        # normailzation
        E = E - torch.max(E, 1, keepdim=True)[0]
        exp = torch.exp(E)
        E_normalized = exp / torch.sum(exp, 1, keepdim=True)
        return E_normalized


class Spatial_Attention_layer(nn.Module):
    """
    compute spatial attention scores
    """
    def __init__(self, num_of_vertices, num_of_features, num_of_timesteps):
        """
        Compute spatial attention scores
        :param num_of_vertices: int
        :param num_of_features: int
        :param num_of_timesteps: int
        """
        super(Spatial_Attention_layer, self).__init__()

        self.W_1 = torch.randn(num_of_timesteps, requires_grad=True).cuda()
        self.W_2 = torch.randn(60, num_of_vertices, requires_grad=True).cuda()     #改过
        self.W_3 = torch.randn(5, requires_grad=True).cuda()                       #改过
        self.b_s = torch.randn(1, num_of_vertices, num_of_vertices, requires_grad=True).cuda()
        self.V_s = torch.randn(num_of_vertices, num_of_vertices, requires_grad=True).cuda()

    def forward(self, x):
        """
        Parameters
        ----------
        x: tensor, x^{(r - 1)}_h,
           shape is (batch_size, N, C_{r-1}, T_{r-1})

           initially, N == num_of_vertices (V)

        Returns
        ----------
        S_normalized: tensor, S', spatial attention scores
                      shape is (batch_size, N, N)

        """
        # get shape of input matrix x
        # batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # The shape of x could be different for different layer, especially the last two dimensions

        # compute spatial attention scores
        # shape of lhs is (batch_size, V, T)
        lhs = torch.matmul(torch.matmul(x.permute(0,2,1), self.W_1), self.W_2)

        # shape of rhs is (batch_size, T, V)
        # rhs = torch.matmul(self.W_3, x.transpose((2, 0, 3, 1)))
        rhs = torch.matmul(x.permute(0,2,1), self.W_3)  # do we need to do transpose??.permute((0, 3, 1, 2))

        # shape of product is (batch_size, V, V)
        product = torch.matmul(lhs.permute(1,0), rhs)

        S = torch.matmul(self.V_s, torch.sigmoid(product + self.b_s))

        # normalization
        S = S - torch.max(S, 1, keepdim=True)[0]
        exp = torch.exp(S)
        S_normalized = exp / torch.sum(exp, 1, keepdim=True)
        return S_normalized

class cheb_conv_with_SAt(nn.Module):
    """
    K-order chebyshev graph convolution with Spatial Attention scores
    """

    def __init__(self, num_of_filters, K, cheb_polynomials, num_of_features):
        """
        Parameters
        ----------
        num_of_filters: int

        num_of_features: int, num of input features

        K: int, up K - 1 order chebyshev polynomials
                will be used in this convolution

        """
        super(cheb_conv_with_SAt, self).__init__()
        self.K = K
        self.num_of_filters = num_of_filters
        self.cheb_polynomials = cheb_polynomials

        global device
        self.Theta = torch.randn(self.K, num_of_features, num_of_filters, requires_grad=True).to(device)

    def forward(self, x, spatial_attention):
        """
        Chebyshev graph convolution operation

        Parameters
        ----------
        x: mx.ndarray, graph signal matrix
           shape is (batch_size, N, F, T_{r-1}), F is the num of features

        spatial_attention: mx.ndarray, shape is (batch_size, N, N)
                           spatial attention scores

        Returns
        ----------
        mx.ndarray, shape is (batch_size, N, self.num_of_filters, T_{r-1})

        """
        (batch_size, num_of_vertices,
         num_of_features, num_of_timesteps) = x.shape

        global device

        outputs = []
        for time_step in range(num_of_timesteps):
            # shape is (batch_size, V, F)
            graph_signal = x[:, :, :, time_step]
            output = torch.zeros(batch_size, num_of_vertices,
                                 self.num_of_filters).to(device)  # do we need to set require_grad=True?
            for k in range(self.K):
                # shape of T_k is (V, V)
                T_k = self.cheb_polynomials[k]

                # shape of T_k_with_at is (batch_size, V, V)
                T_k_with_at = T_k * spatial_attention

                # shape of theta_k is (F, num_of_filters)
                # theta_k = self.Theta.data()[k]
                theta_k = self.Theta[k]

                # shape is (batch_size, V, F)
                # rhs = nd.batch_dot(T_k_with_at.transpose((0, 2, 1)),  # why do we need to transpose?
                #                    graph_signal)
                rhs = torch.matmul(T_k_with_at.permute((0, 2, 1)),
                                   graph_signal)

                output = output + torch.matmul(rhs, theta_k)
            # outputs.append(output.expand_dims(-1))
            outputs.append(torch.unsqueeze(output, -1))
        return F.relu(torch.cat(outputs, dim=-1))


class STGCNBlock1(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    # def __init__(self, in_channels, spatial_channels, out_channels_1, out_channels_2, num_nodes):
    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock1, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        # self.Theta1 = nn.Parameter(torch.FloatTensor(24, 24*8))
        self.Theta = torch.randn(60, 29*5, requires_grad=True).to(device)
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)

        # self.temporal1 = TimeBlock(in_channels=in_channels,
        #                            out_channels=out_channels_1)
        # self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels_2,
        #                                              spatial_channels))
        # self.temporal2 = TimeBlock(in_channels=spatial_channels,
        #                            out_channels=out_channels_2)

        self.batch_norm = nn.BatchNorm1d(5)
        # self.reset_parameters()

        # self.TAt = Temporal_Attention_layer(num_of_vertices=24, num_of_features=1, num_of_timesteps=7)
        # self.SAt = Spatial_Attention_layer(num_of_vertices=24, num_of_features=1, num_of_timesteps=7)

        self.TAt = Temporal_Attention_layer(num_of_vertices=60, num_of_features=60, num_of_timesteps=5)         #改过num_of_features  1->24
        self.SAt = Spatial_Attention_layer(num_of_vertices=60, num_of_features=60, num_of_timesteps=5)
        self.conv = nn.Conv1d(in_channels=60,
                            out_channels=60,
                            kernel_size= 3,
                            padding=1)

    # def reset_parameters(self):
    #     stdv = 1. / math.sqrt(self.Theta1.shape[1])
    #     self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        # t = self.temporal1(X)
        # lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.matmul(lfs, self.Theta1))
        # t3 = self.temporal2(t2)
        # return self.batch_norm(t3)
        # X = X.permute(0, 1, 3, 2)
        temporal_At = self.TAt(X)
        x_TAt = torch.matmul(X.permute(0,2,1), temporal_At).reshape(-1, 5, 60)#.reshape(X.shape[0], -1, 60)
        spatial_At = self.SAt(x_TAt)
        lfs = torch.einsum("ij,jkl->kil", [A_hat, spatial_At.permute(1, 0, 2)])
        lfs=torch.unsqueeze(lfs,dim=0)
        #extenddimlfs = torch.unsqueeze(lfs, -1)
        #resdimlfs = lfs#extenddimlfs.permute(0, 3, 1, 2)
        t2 = F.elu(torch.matmul(x_TAt,lfs))
        # t2 = F.relu(torch.matmul(lfs, self.Theta)).reshape(-1, 24, 8, 24)
        #t2 = F.elu(self.conv(resdimlfs))
        t3 = t2.reshape(-1, 5,60)
        return self.batch_norm(t3)

'''



class STGCNBlock2(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    # def __init__(self, in_channels, spatial_channels, out_channels_1, out_channels_2, num_nodes):
    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock2, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        # self.Theta1 = nn.Parameter(torch.FloatTensor(24, 8*24))
        # self.Theta=torch.randn(24, 8*24, requires_grad=True).to(device)
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=7,
                              kernel_size=(3, 3),
                              padding=1)

        # self.temporal1 = TimeBlock(in_channels=in_channels,
        #                            out_channels=out_channels_1)
        # self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels_2,
        #                                              spatial_channels))
        # self.temporal2 = TimeBlock(in_channels=spatial_channels,
        #                            out_channels=out_channels_2)

        #self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.batch_norm = nn.LayerNorm(num_nodes)
        # self.reset_parameters()

        self.TAt = Temporal_Attention_layer(num_of_vertices=60, num_of_features=60, num_of_timesteps=29)
        self.SAt = Spatial_Attention_layer(num_of_vertices=60, num_of_features=60, num_of_timesteps=29)

    #
    # def reset_parameters(self):
    #     stdv = 1. / math.sqrt(self.Theta1.shape[1])
    #     self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        # X = X.permute(0, 1, 3, 2)
        temporal_At = self.TAt(X)
        x_TAt = torch.matmul(X.reshape(X.shape[0], -1, 7), temporal_At).reshape(-1, 24, 24, 7)
        spatial_At = self.SAt(x_TAt)
        lfs = torch.einsum("ij,jkl->kil", [A_hat, spatial_At.permute(1, 0, 2)])
        extenddimlfs = torch.unsqueeze(lfs, -1)
        resdimlfs = extenddimlfs.permute(0, 3, 1, 2)
        # t2 = F.relu(torch.matmul(lfs, self.Theta)).reshape(-1, 24, 8, 24)
        t2 = F.elu(self.conv(resdimlfs))
        t3 = t2.reshape(-1, 24, 7, 24)
        return self.batch_norm(t3)
'''
class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        self.block1 = STGCNBlock1(in_channels=num_features, out_channels=64,
                                 spatial_channels=24, num_nodes=num_nodes)
        # self.block2 = STGCNBlock2(in_channels=num_features, out_channels=64,
        #                          spatial_channels=24, num_nodes=num_nodes)
        #self.conv=nn.Conv2d()

        # self.block1 = STGCNBlock(in_channels=num_features, out_channels_1=16, out_channels_2=64,
        #                          spatial_channels=16, num_nodes=num_nodes)
        # self.block2 = STGCNBlock(in_channels=64, out_channels_1=16, out_channels_2=64,
        #                          spatial_channels=16, num_nodes=num_nodes)

        # self.last_temporal = TimeBlock(in_channels=64, out_channels=64)

        # self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 64,num_timesteps_output)
        self.conv = nn.Conv1d(in_channels=5,
                            out_channels=1,
                            kernel_size=3,
                            padding= 1
                            )
        # self.conv1=nn.Conv2d(in_channels=24,
        #                     out_channels=24,
        #                     kernel_size=(4,1)
        #                     )
        self.ln=nn.LayerNorm(600)
        self.fully = nn.Linear(60, 60)

    def forward(self, A_hat, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        out1 = self.block1(X, A_hat)
        out3 = self.block1(out1, A_hat)
        out3 = F.relu(self.conv(out3))
        # out3 = F.relu(self.conv1(out3))
        #out3=self.ln(out3)


        # out3 = self.last_temporal(out2)
        out4 = self.fully(out3.reshape((-1, 60)))#.reshape(-1, 24, 1, 24)
        out4=torch.unsqueeze(out4,dim=1)
        return out4


