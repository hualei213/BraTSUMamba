

import h5py
import torch

from unet.network.lib.pointops2.functions import pointops

# from .hier import *
from unet.network.hier import *
# from .inter import *
from unet.network.inter import *

seed=0
# pl.seed_everything(seed) # , workers=True
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if use multi-GPU
# torch.backends.cudnn.deterministic=True
# torch.backends.cudnn.benchmark=False 
# from model import get as get_model
class RandomCrop(object):
    """
    Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired outupt size. If int,
        square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # we assume the order is: height, width, depth
        h, w, d = image.shape[:3]
        new_h, new_w, new_d = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        back = np.random.randint(0, d - new_d)

        image = image[top : top + new_h,
                      left: left + new_w,
                      back: back + new_d]
        label = label[top : top + new_h,
                      left: left + new_w,
                      back: back + new_d]

        return {'image': image, 'label': label}

class BilinearFeedForward(nn.Module):

    def __init__(self, in_planes1, in_planes2, out_planes):
        super().__init__()
        self.bilinear = nn.Bilinear(in_planes1, in_planes2, out_planes)

    def forward(self, x):
        x = x.contiguous()
        x = self.bilinear(x, x)
        return x

####################################################################################

class NoIntraSetLayer(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.out_planes = out_planes
        self.nsample = nsample

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)

        x_knn, knn_idx = pointops.queryandgroup(
            self.nsample, p, p, x, None, o, o, use_xyz=True, return_idx=True)  # (n, k, 3+c)
        p_r = x_knn[:, :, 0:3]
        x_knn = x_knn[:, :, 3:]

        return (x, x_knn, knn_idx, p_r)

# PointMixerIntraSetLayer_ECCV22
class PointMixerIntraSetLayer(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample

        self.channelMixMLPs01 = nn.Sequential( # input.shape = [N, K, C]
            nn.Linear(3+in_planes, nsample),
            nn.ReLU(inplace=True),
            BilinearFeedForward(nsample, nsample, nsample))
        
        self.linear_p = nn.Sequential( # input.shape = [N, K, C]
            nn.Linear(3, 3, bias=False),
            nn.Sequential(
                Rearrange('n k c -> n c k'),
                nn.BatchNorm1d(3),
                Rearrange('n c k -> n k c')),
            nn.ReLU(inplace=True), 
            nn.Linear(3, out_planes))
        self.shrink_p = nn.Sequential(
            Rearrange('n k (a b) -> n k a b', b=nsample),
            Reduce('n k a b -> n k b', 'sum', b=nsample))

        self.channelMixMLPs02 = nn.Sequential( # input.shape = [N, K, C]
            Rearrange('n k c -> n c k'),
            nn.Conv1d(nsample+nsample, mid_planes, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_planes), 
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_planes, mid_planes//share_planes, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_planes//share_planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_planes//share_planes, out_planes//share_planes, kernel_size=1),
            Rearrange('n c k -> n k c'))

        self.channelMixMLPs03 = nn.Linear(in_planes, out_planes)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)

        x_knn, knn_idx = pointops.queryandgroup(
            self.nsample, p, p, x, None, o, o, use_xyz=True, return_idx=True)  # (n, k, 3+c)
        p_r = x_knn[:, :, 0:3]

        energy = self.channelMixMLPs01(x_knn) # (n, k, k)
        
        p_embed = self.linear_p(p_r) # (n, k, out_planes)
        p_embed_shrink = self.shrink_p(p_embed) # (n, k, k)

        energy = torch.cat([energy, p_embed_shrink], dim=-1)
        energy = self.channelMixMLPs02(energy) # (n, k, out_planes/share_planes)
        w = self.softmax(energy)

        x_v = self.channelMixMLPs03(x)  # (n, in_planes) -> (n, k)
        n = knn_idx.shape[0]; knn_idx_flatten = knn_idx.flatten()
        x_v  = x_v[knn_idx_flatten, :].view(n, self.nsample, -1)

        n, nsample, out_planes = x_v.shape
        x_knn = (x_v + p_embed).view(
            n, nsample, self.share_planes, out_planes//self.share_planes)
        x_knn = (x_knn * w.unsqueeze(2))
        x_knn = x_knn.reshape(n, nsample, out_planes)

        x = x_knn.sum(1)
        return (x, x_knn, knn_idx, p_r)

####################################################################################

class PointMixerBlock(nn.Module):
    expansion = 1

    def __init__(self, 
                 in_planes, planes, share_planes=8, 
                 nsample=16, 
                 use_xyz=False,
                 intraLayer='PointMixerIntraSetLayer',
                 interLayer='PointMixerInterSetLayer'):
        super().__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = nn.Sequential(
            globals()[intraLayer](planes, planes, share_planes, nsample),
            globals()[interLayer](in_planes, share_planes, nsample))
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes*self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x = x + identity
        x = self.relu(x)
        return [p, x, o]

####################################################################################

class PointMixerSegNet(nn.Module):
    mixerblock = PointMixerBlock

    def __init__(
        self, block, blocks, 
        c=4, k=13, nsample=[8,16,16,16,16], stride=[1,4,4,4,4],
        intraLayer='PointMixerIntraSetLayer',
        interLayer='PointMixerInterSetLayer',
        transup='SymmetricTransitionUpBlock', 
        transdown='TransitionDownBlock'):
        super().__init__()
        
        self.c = c
        self.intraLayer = intraLayer
        self.interLayer = interLayer
        self.transup = transup
        self.transdown = transdown
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        
        assert stride[0] == 1, 'or you will meet errors.'

        self.enc1 = self._make_enc(planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])  # N/256
        self.dec5 = self._make_dec(planes[4], 2, share_planes, nsample=nsample[4], is_head=True)  # transform p5
        self.dec4 = self._make_dec(planes[3], 2, share_planes, nsample=nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1
        self.cls = nn.Sequential(
            nn.Linear(planes[0], planes[0]), 
            nn.BatchNorm1d(planes[0]), 
            nn.ReLU(inplace=True), 
            nn.Linear(planes[0], k))

    def _make_enc(self, planes, blocks, share_planes, stride, nsample):
        layers = []
        layers.append(globals()[self.transdown]( 
            in_planes=self.in_planes, 
            out_planes=planes, 
            stride=stride, 
            nsample=nsample))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(self.mixerblock(
                in_planes=self.in_planes, 
                planes=self.in_planes, 
                share_planes=share_planes,
                nsample=nsample,
                intraLayer=self.intraLayer,
                interLayer=self.interLayer))
        return nn.Sequential(*layers)

    def _make_dec(self, planes, blocks, share_planes, nsample, is_head=False):
        layers = []
        layers.append(globals()[self.transup](
            in_planes=self.in_planes, 
            out_planes=None if is_head else planes, 
            nsample=nsample))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(self.mixerblock(
                in_planes=self.in_planes, 
                planes=self.in_planes, 
                share_planes=share_planes,
                nsample=nsample,
                intraLayer=self.intraLayer,
                interLayer=self.interLayer))
        return nn.Sequential(*layers)

    def forward(self, pxo):
        B,C,H,W,D = pxo.shape
        m, n, l = H, W, D
        x = np.linspace(0, m - 1, m)
        y = np.linspace(0, n - 1, n)
        z = np.linspace(0, l - 1, l)
        x, y, z = np.meshgrid(x, y, z)
        x.shape = (m * n * l, 1)
        y.shape = (m * n * l, 1)
        z.shape = (m * n * l, 1)
        arr_3D = np.concatenate((x, y, z), axis=1)
        p0 = torch.Tensor(arr_3D).to(torch.device("cuda"))
        p0 = p0.repeat_interleave(B,dim=0)
        # p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        x0 = pxo.reshape(-1).unsqueeze(1)  # (n, 3), (n, c), (b)
        start,_ = arr_3D.shape
        o0 = torch.range(start,start*B,start,dtype=torch.int32).to(torch.device('cuda'))

        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)

        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        x = self.cls(x1)

        x0 = x.reshape(B,2,H,W,D)
        return x0

####################################################################################

def getPointMixerSegNet(**kwargs):
    '''
    kwargs['transup'] = 'SymmetricTransitionUpBlock'
    '''
    model = PointMixerSegNet(PointMixerBlock, [2, 3, 4, 6, 3], **kwargs)
    return model

def get_pointmix():
    kwargs = \
        {
            'intraLayer': "PointMixerIntraSetLayer",
            'interLayer': "PointMixerInterSetLayer",
            'transup': "SymmetricTransitionUpBlock",
            'transdown': "SymmetricTransitionDownBlock",
            'stride': [1, 4, 4, 4, 4],
        }
    # device = torch.device("cuda")
    model = getPointMixerSegNet(c=4, k=2, nsample=[8, 16, 16, 16, 16], **kwargs)
    return model

if __name__=="__main__":

    kwargs = \
        {
            'intraLayer': "PointMixerIntraSetLayer",
            'interLayer': "PointMixerInterSetLayer",
            'transup': "SymmetricTransitionUpBlock",
            'transdown': "SymmetricTransitionDownBlock",
            'stride': [1, 4, 4, 4, 4],
        }
    device = torch.device("cuda")
    model = getPointMixerSegNet(c=4, k=2, nsample=[8,16,16,16,16], **kwargs).to(device)
    print(model)

    m, n, l = 128, 128, 128
    x = np.linspace(0, m - 1, m)
    y = np.linspace(0, n - 1, n)
    z = np.linspace(0, l - 1, l)

    x, y, z = np.meshgrid(x, y, z)
    x.shape = (m * n * l, 1)
    y.shape = (m * n * l, 1)
    z.shape = (m * n * l, 1)
    arr_3D = np.concatenate((x, y, z), axis=1)
    data_path = "/media/shenhualei/SSD05/sggq/datasets/S3DIS/s3dis/brain_2class_hdf5_t1_MNI/0/train/trainval_fullarea/instance-IBSR_08.hdf5"
    f = h5py.File(data_path, 'r')
    rcop=RandomCrop(128)
    f = rcop(f)
    image = f['image']
    label = f['label']
    image = image[:]
    label = label[:]
    # image = np.reshape(image,-1)
    p0 = torch.Tensor(arr_3D).to(device)
    x0 = torch.Tensor((image)).to(device)
    x0 = torch.unsqueeze(x0,dim=1)
    o0 = torch.tensor([4000,8000], dtype=torch.int32).to(device)

    result = x0
    image = torch.Tensor(image.transpose(3,0,1,2)).unsqueeze(0).to(device)
    out_put = model(image)

    # model = get_model('net_pointmixer')
