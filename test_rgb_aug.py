import argparse

import torch
from torch.utils.data import DataLoader

from mini_imagenet_rgb import MiniImageNet
from samplers import CategoriesSampler
from convnet import Convnet
from utils import pprint, set_gpu, count_acc, Averager, euclidean_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--load', default='/home/shenyq/zsl/prototypical-network-pytorch/save/proto-1/max-acc.pth')
    parser.add_argument('--batch', type=int, default=2000)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=30)
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)

    dataset = MiniImageNet('test')
    sampler = CategoriesSampler(dataset.label,
                                args.batch, args.way, args.shot + args.query)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=8, pin_memory=True)

    model = Convnet().cuda()
    model.load_state_dict(torch.load(args.load))
    model.eval()

    
    ave_acc = Averager()
    for i, batch in enumerate(loader, 1):
        # data, _ = [_.cuda() for _ in batch]
        data, mask,bg,_ = [_.cuda() for _ in batch]
        p = args.shot * args.test_way
        data_spt, data_query = data[:p], data[p:]
        mask_spt = mask[:p]
        mask_spt[mask_spt!=0]=1

        N,c,w,h = data.size()
        data_spt_repeat = data_spt.unsqueeze(0).repeat(N,1,1,1,1)
        mask_spt_repeat = mask_spt.unsqueeze(0).repeat(N,1,1,1,1)
        bg_repeat = bg.unsqueeze(0).repeat(p,1,1,1,1).permute(1,0,2,3,4)
        merge_spt = data_spt_repeat*mask_spt_repeat.float() + bg_repeat*(1-mask_spt_repeat.float())
        m,n,c,w,h = merge_spt.size()
        merge_spt = merge_spt.reshape(m*n,c,w,h)

        proto = model(merge_spt)
        proto = proto.reshape(m, args.test_way, -1).mean(dim=0)

        label = torch.arange(args.test_way).repeat(args.query)
        label = label.type(torch.cuda.LongTensor)

        logits = euclidean_metric(model(data_query), proto)
        
        acc = count_acc(logits, label)

        ave_acc.add(acc)
        print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))
        
        proto = None; logits = None; loss = None
