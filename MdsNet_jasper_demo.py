import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

seed = 1234
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Load DATA
data = sio.loadmat("jasper_hs_dataset.mat")

abundance_GT = torch.from_numpy(data["A"])  # true abundance
original_HSI = torch.from_numpy(data["Y"])  # mixed abundance
hyper_segmentation = torch.from_numpy(data["Yc"])

# VCA_endmember and GT
VCA_endmember = data["M1"]
GT_endmember = data["M"]

endmember_init = torch.from_numpy(VCA_endmember).unsqueeze(2).unsqueeze(3).float()
GT_init = torch.from_numpy(GT_endmember).unsqueeze(2).unsqueeze(3).float()

band_Number = original_HSI.shape[0]
endmember_number, pixel_number = abundance_GT.shape

col = 100

original_HSI = torch.reshape(original_HSI, (band_Number, col, col))
abundance_GT = torch.reshape(abundance_GT, (endmember_number, col, col))
hyper_segmentation = torch.reshape(hyper_segmentation, (band_Number, col, col))

batch_size = 256####
EPOCH = 800

alpha = 1
beta = 1
gamma = 0.01
zeta = 0.4
eta = 0
drop_out = 0.1
learning_rate = 5e-2
reduction = 2
minimum = 1e-3


class L1_norm(nn.Module):

    def __int__(self):
        super(L1_norm, self).__init__()

    def forward(self, x):
        loss = torch.sum(torch.sqrt((x * x) + minimum))
        return loss


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(4, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()
        self.downsampling22 = nn.AvgPool2d(2, 2, ceil_mode=True)

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

cross_loss = nn.CrossEntropyLoss()



# abundance normalization
def norm_abundance_GT(abundance_input, abundance_GT_input):
    abundance_input = abundance_input / (torch.sum(abundance_input, dim=1))
    abundance_input = torch.reshape(
        abundance_input.squeeze(0), (endmember_number, col, col)
    )
    abundance_input = abundance_input.cpu().detach().numpy()
    abundance_GT_input = abundance_GT_input / (torch.sum(abundance_GT_input, dim=0))
    abundance_GT_input = abundance_GT_input.cpu().detach().numpy()
    return abundance_input, abundance_GT_input


# endmember normalization
def norm_endmember(endmember_input, endmember_GT):
    for i in range(0, endmember_number):
        endmember_input[:, i] = endmember_input[:, i] / np.max(endmember_input[:, i])
        endmember_GT[:, i] = endmember_GT[:, i] / np.max(endmember_GT[:, i])
    return endmember_input, endmember_GT


# plot abundance
def plot_abundance(abundance_input, abundance_GT_input):
    for i in range(0, endmember_number):

        plt.subplot(2, endmember_number, i + 1)
        plt.imshow(abundance_input[i, :, :], cmap="jet")

        plt.subplot(2, endmember_number, endmember_number + i + 1)
        plt.imshow(abundance_GT_input[i, :, :], cmap="jet")
    plt.show()


# plot endmember
def plot_endmember(endmember_input, endmember_GT):
    for i in range(0, endmember_number):
        plt.subplot(1, endmember_number, i + 1)
        plt.plot(endmember_input[:, i], color="b")
        plt.plot(endmember_GT[:, i], color="r")

    plt.show()


# change the index of abundance and endmember
def arange_A_E(abundance_input, abundance_GT_input, endmember_input, endmember_GT):
    RMSE_matrix = np.zeros((endmember_number, endmember_number))
    SAD_matrix = np.zeros((endmember_number, endmember_number))
    RMSE_index = np.zeros(endmember_number).astype(int)
    SAD_index = np.zeros(endmember_number).astype(int)
    RMSE_abundance = np.zeros(endmember_number)
    SAD_endmember = np.zeros(endmember_number)

    for i in range(0, endmember_number):
        for j in range(0, endmember_number):
            RMSE_matrix[i, j] = AbundanceRmse(
                abundance_input[i, :, :], abundance_GT_input[j, :, :]
            )
            SAD_matrix[i, j] = SAD_distance(endmember_input[:, i], endmember_GT[:, j])

        RMSE_index[i] = np.argmin(RMSE_matrix[i, :])
        SAD_index[i] = np.argmin(SAD_matrix[i, :])
        RMSE_abundance[i] = np.min(RMSE_matrix[i, :])
        SAD_endmember[i] = np.min(SAD_matrix[i, :])

    abundance_input[np.arange(endmember_number), :, :] = abundance_input[
        RMSE_index, :, :
    ]
    endmember_input[:, np.arange(endmember_number)] = endmember_input[:, SAD_index]

    return abundance_input, endmember_input, RMSE_abundance, SAD_endmember


class load_data(torch.utils.data.Dataset):
    def __init__(self, img, gt, hs, transform=None):
        self.img = img.float()
        self.gt = gt.float()
        self.hs = hs.float()
        self.transform = transform

    def __getitem__(self, idx):
        return self.img, self.gt, self.hs

    def __len__(self):
        return 1


# calculate RMSE of abundance
def AbundanceRmse(inputsrc, inputref):
    rmse = np.sqrt(((inputsrc - inputref) ** 2).mean())
    return rmse


def conv33(inchannel, outchannel):
    return nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1)


def conv11(inchannel, outchannel):
    return nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1)


def transconv11(inchannel,outchannel):
    return nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1)


# calculate SAD of endmember
def SAD_distance(src, ref):
    cos_sim = np.dot(src, ref) / (np.linalg.norm(src) * np.linalg.norm(ref))
    SAD_sim = np.arccos(cos_sim)
    return SAD_sim


# my net
class mds_net(nn.Module):
    def __init__(self):
        super(mds_net, self).__init__()
        self.encoder_un = nn.Sequential(
            conv33(band_Number, 96),
            #nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Dropout(drop_out),
            conv33(96, 48),
            # nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(drop_out),
            conv33(48, endmember_number),
        )

        self.se_attention = nn.Sequential(
            nn.Conv2d(endmember_number, endmember_number // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PReLU(endmember_number // reduction),
            nn.Conv2d(endmember_number // reduction, endmember_number, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

        self.encoder_seg = nn.Sequential(
            conv33(band_Number, 96),
            # nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Dropout(drop_out),
            conv33(96, 48),
            # nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(drop_out),
            conv33(48, endmember_number),
        )

        self.softmax = nn.Softmax(dim=1)

        self.decoder_origin = nn.Sequential(
            nn.Conv2d(
                in_channels=endmember_number,
                out_channels=band_Number,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.LeakyReLU(0.2)
        )

        self.decoder_seg = self.decoder_origin

    def forward(self, x, y):
        unmixing = self.encoder_un(x)
        seg = self.encoder_seg(y)
        attention = self.se_attention(seg)
        abundance = self.softmax(torch.mul(unmixing,attention))
        seg_abundance = self.softmax(seg)
        hsi_recon = self.decoder_origin(abundance)
        seg_recon = self.decoder_seg(seg)

        return abundance, hsi_recon, seg_abundance, seg_recon


# SAD loss of reconstruction
def reconstruction_SADloss(output, target):

    _, band, h, w = output.shape
    output = torch.reshape(output, (band, h * w))
    target = torch.reshape(target, (band, h * w))
    abundance_loss = torch.acos(torch.cosine_similarity(output, target, dim=0))
    abundance_loss = torch.mean(abundance_loss)

    return abundance_loss

MSE = torch.nn.MSELoss(size_average=True)

# weights_init
def weights_init(m):
    nn.init.kaiming_normal_(net.encoder_un[0].weight.data)
    nn.init.kaiming_normal_(net.encoder_un[4].weight.data)
    nn.init.kaiming_normal_(net.encoder_un[8].weight.data)

    nn.init.kaiming_normal_(net.encoder_seg[0].weight.data)
    nn.init.kaiming_normal_(net.encoder_seg[4].weight.data)
    nn.init.kaiming_normal_(net.encoder_seg[8].weight.data)


# load data
train_dataset = load_data(
    img=original_HSI, gt=abundance_GT, hs=hyper_segmentation, transform=transforms.ToTensor()
)
# Data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=False
)

net = mds_net().cuda()
edgeLoss = EdgeLoss()
L1loss = CharbonnierLoss()
L1norm = L1_norm()

# weight init
net.apply(weights_init)

# decoder weight init by VCA
model_dict = net.state_dict()
model_dict["decoder_origin.0.weight"] = endmember_init

net.load_state_dict(model_dict)

# optimizer
#optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-3)
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.6)


# train
for epoch in range(EPOCH):
    for i, (x, y, z) in enumerate(train_loader):
        x = x.cuda()
        z = z.cuda()
        net.train().cuda()

        abundance1, recon_result1, abundance2, recon_result2 = net(x, z)

        reconLoss = reconstruction_SADloss(x, recon_result1)

        MSELoss = MSE(x, recon_result1)

        edge = edgeLoss(abundance1, abundance2)
        abundanceLoss = L1loss(abundance1, abundance2)
        sparseloss = L1norm(x)

        ALoss = reconLoss
        BLoss = MSELoss
        CLoss = edge
        DLoss = abundanceLoss
        ELoss = sparseloss

        total_loss = (alpha * ALoss) + (beta * BLoss) + (gamma * CLoss) + (zeta * DLoss) + (eta * sparseloss)
        optimizer.zero_grad()

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0:
            print(
                "Epoch:",
                epoch,
                "| loss: %.4f" % total_loss.cpu().data.numpy(),
            )


net.eval()


abundance1, recon_result1, abundance2, recon_result2 = net(x, z)

decoder_para = net.state_dict()["decoder_origin.0.weight"].cpu().numpy()
decoder_para = np.mean(np.mean(decoder_para, -1), -1)

en_abundance, abundance_GT = norm_abundance_GT(abundance1, abundance_GT)
decoder_para, GT_endmember = norm_endmember(decoder_para, GT_endmember)

en_abundance, decoder_para, RMSE_abundance, SAD_endmember = arange_A_E(
    en_abundance, abundance_GT, decoder_para, GT_endmember
)
print("RMSE", RMSE_abundance)
print("mean_RMSE", RMSE_abundance.mean())
print("endmember_SAD", SAD_endmember)
print("mean_SAD", SAD_endmember.mean())
sio.savemat('jasper/jasper_result.mat', {'A':en_abundance, 'E':decoder_para})

plot_abundance(en_abundance, abundance_GT)
plot_endmember(decoder_para, GT_endmember)