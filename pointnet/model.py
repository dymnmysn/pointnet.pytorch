import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import GraphConv
from pytorch3d.io import load_objs_as_meshes


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = GraphConv(3, 64)
        self.conv2 = GraphConv(64, 128)
        self.conv3 = GraphConv(128, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        es = torch.tensor([[0, 1]])
        x = F.relu(self.conv1(x, es))
        x = F.relu(self.conv2(x, es))
        x = F.relu(self.conv3(x, es))
        x = torch.max(x, 0, keepdim=True)[0]

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        ident = torch.eye(3, 3)
        if x.is_cuda:
            ident = ident.cuda()
        x = x.view(3, 3)
        x = x + ident
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()

        self.conv1 = GraphConv(k, 64)
        self.conv2 = GraphConv(64, 128)
        self.conv3 = GraphConv(128, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k ** 2)
        self.relu = nn.ReLU()
        self.k = k

    def forward(self, x):
        es = torch.tensor([[0, 1]])
        x = F.relu(self.conv1(x, es))
        x = F.relu(self.conv2(x, es))
        x = F.relu(self.conv3(x, es))
        x = torch.max(x, 0, keepdim=True)[0]

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        ident = torch.eye(self.k, self.k)
        if x.is_cuda:
            ident = ident.cuda()
        x = x.view(self.k, self.k)
        x = x + ident
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = GraphConv(3, 64)
        self.conv2 = GraphConv(64, 128)
        self.conv3 = GraphConv(128, 1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        edges = torch.tensor([[0, 1]])
        n_pts = x.size()[0]
        trans = self.stn(x)
        x = F.relu(self.conv1(x, edges))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = torch.matmul(x, trans_feat)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.conv2(x, edges))
        x = self.conv3(x, edges)
        x = torch.max(x, 0, keepdim=True)[0]
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.repeat(n_pts,1)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = GraphConv(1088, 512)
        self.conv2 = GraphConv(512, 256)
        self.conv3 = GraphConv(256, 128)
        self.conv4 = GraphConv(128, self.k)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        edges = torch.tensor([[0,1]])
        x = F.relu(self.conv1(x, edges))
        x = F.relu(self.conv2(x, edges))
        x = F.relu(self.conv3(x, edges))
        x = self.conv4(x, edges)
        x = F.log_softmax(x, dim=-1)
        return x, trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d)
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.norm(torch.matmul(trans, trans.transpose(1, 0)) - I)
    return loss


if __name__ == '__main__':
    my_mesh = load_objs_as_meshes(["..\pointnet\dolphin.obj"])
    sim_data = my_mesh.verts_packed()
    trans = STN3d()
    out = trans(sim_data)
    # print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = torch.rand(2500, 64)
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k=5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k=3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
