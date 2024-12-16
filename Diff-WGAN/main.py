"""
Train a diffusion model for recommendation
"""

import argparse
from ast import parse
import os
import time
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import models.gaussian_diffusion as gd
from models.DNN import DNN
from models.Dis import Discriminator, compute_gradient_penalty
import evaluate_utils
import data_utils
from copy import deepcopy

import random
random_seed = 1
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) # gpu
np.random.seed(random_seed) # numpy
random.seed(random_seed) # random and transforms
torch.backends.cudnn.deterministic=True # cudnn
def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='yelp_clean', help='choose the dataset')
parser.add_argument('--data_path', type=str, default='./datasets/', help='load data path')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--lr2', type=float, default=0.0005, help='learning rate for Discriminator')
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=400)
parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
parser.add_argument('--topN', type=str, default='[10, 20, 50, 100]')
parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')
parser.add_argument('--log_name', type=str, default='log', help='the log name')
parser.add_argument('--round', type=int, default=1, help='record the experiment')

# params for the model
parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
parser.add_argument('--dims', type=str, default='[1000]', help='the dims for the DNN')
parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')

# params for diffusion
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=100, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound for noise generating')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')

args = parser.parse_args()
print("args:", args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0" if args.cuda else "cpu")

print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

### DATA LOAD ###
train_path = args.data_path + 'train_list.npy'
valid_path = args.data_path + 'valid_list.npy'
test_path = args.data_path + 'test_list.npy'

train_data, valid_y_data, test_y_data, n_user, n_item, _ = data_utils.data_load(train_path, valid_path, test_path)
train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

if args.tst_w_val:
    tv_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A) + torch.FloatTensor(valid_y_data.A))
    test_twv_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False)
mask_tv = train_data + valid_y_data

print('data ready.')


### Build Gaussian Diffusion ###
if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
    mean_type = gd.ModelMeanType.EPSILON
else:
    raise ValueError("Unimplemented mean type %s" % args.mean_type)

diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, \
        args.noise_scale, args.noise_min, args.noise_max, args.steps, device).to(device)

### Build MLP ###
out_dims = eval(args.dims) + [n_item]
in_dims = out_dims[::-1]

model_gen = DNN(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm).to(device)
model_dis = Discriminator(n_item).to(device)

optimizer_gen = optim.AdamW(model_gen.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# optimizer_dis = optim.AdamW(model_dis.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# optimizer_dis = optim.Adagrad(model_dis.parameters(), 
#                   lr=args.lr, initial_accumulator_value=1e-8, weight_decay=args.weight_decay)
optimizer_dis = optim.Adam(model_dis.parameters(), lr=args.lr2, weight_decay=args.weight_decay)
# optimizer_dis = optim.SGD(model_dis.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# optimizer_dis = optim.SGD(model_dis.parameters(), lr=args.lr, momentum=0.95, weight_decay=args.weight_decay)

print("models ready.")

param_num = 0
mlp_num = sum([param.nelement() for param in model_gen.parameters()])
diff_num = sum([param.nelement() for param in diffusion.parameters()])  # 0
param_num = mlp_num + diff_num
print("Number of gen parameters:", param_num)
print("Number of dis parameters:", sum([param.nelement() for param in model_dis.parameters()]))

regularization = nn.MSELoss()

def evaluate(data_loader, data_te, mask_his, topN):
    model_gen.eval()
    e_idxlist = list(range(mask_his.shape[0])) # users id
    e_N = mask_his.shape[0] # number of users

    predict_items = []
    target_items = [] # [[107, 172, 175, 177, 180, 332, 564, 1052], [89, 103, 189, 193, 207, 211, 261, 305, 396, 432, 433, 583, 587, 823],...] target items of each user
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            his_data = mask_his[e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]] # user-item matrix; e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]: get users id
            batch = batch.to(device)
            prediction = diffusion.p_sample(model_gen, batch, args.sampling_steps, args.sampling_noise)
            prediction[his_data.nonzero()] = -np.inf

            _, indices = torch.topk(prediction, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)

    test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)

    return test_results

best_recall, best_epoch = -100, 0
best_test_result = None
step_gen, step_dis = 2, 4
lambda_gp = 100

print("Start training...")
for epoch in range(1, args.epochs + 1):
    if epoch - best_epoch >= 100:
        print('-'*18)
        print('Exiting from training early')
        break

    start_time = time.time()

    total_loss_dis, total_loss_gen = 0.0, 0.0

    model_dis.eval()
    model_gen.train()
    # Generator
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        optimizer_gen.zero_grad()
        
        terms = diffusion.training_losses(model_gen, batch, args.reweight)
        elbo = terms["loss"].mean()
        fake_batch = terms["pred_xstart"]

        fake_result = model_dis(fake_batch)
        gen_loss = -torch.mean(torch.log(fake_result + 10e-9)) + 0.1*regularization(fake_batch, batch)
        loss = elbo + gen_loss
        
        total_loss_gen += loss
        loss.backward()
        optimizer_gen.step()

    end_time_dis = time.time()
    
    model_dis.train()
    model_gen.eval()
    # Discriminator
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        optimizer_dis.zero_grad()

        fake_batch = diffusion.p_sample(model_gen, batch, args.sampling_steps, args.sampling_noise)
        fake_batch = 0.5*fake_batch + 0.5*batch

        fake_result = model_dis(fake_batch)
        real_result = model_dis(batch)
        
        dis_loss = -torch.mean(real_result*torch.log(fake_result + 10e-9) + (1-real_result)*torch.log(1.0 - fake_result + 10e-9))

        # # Gradient penalty
        # gradient_penalty = compute_gradient_penalty(model_dis, batch, fake_batch)
        # # Adversarial loss
        # dis_loss = -torch.mean(real_result) + torch.mean(fake_result) + lambda_gp * gradient_penalty

        # dis_loss.backward()
        # total_loss_dis += dis_loss
        total_loss_dis += dis_loss
        dis_loss.backward()
        optimizer_dis.step()
    
    if epoch % 5 == 0:
        valid_results = evaluate(test_loader, valid_y_data, train_data, eval(args.topN))
        if args.tst_w_val:
            test_results = evaluate(test_twv_loader, test_y_data, mask_tv, eval(args.topN))
        else:
            test_results = evaluate(test_loader, test_y_data, mask_tv, eval(args.topN))
        evaluate_utils.print_results(None, valid_results, test_results)

        if valid_results[1][1] > best_recall: # recall@20 as selection
            best_recall, best_epoch = valid_results[1][1], epoch
            best_results = valid_results
            best_test_results = test_results

            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save(model_gen, '{}{}_lr{}_lr2{}_wd{}_bs{}_dims{}_emb{}_{}_steps{}_scale{}_min{}_max{}_sample{}_reweight{}_{}_gan.pth' \
                .format(args.save_path, args.dataset, args.lr, args.lr2, args.weight_decay, args.batch_size, args.dims, args.emb_size, args.mean_type, \
                args.steps, args.noise_scale, args.noise_min, args.noise_max, args.sampling_steps, args.reweight, args.log_name))
    
    
    print("Runing Epoch {:03d} ".format(epoch) + 'train dis loss {:.4f}'.format(total_loss_dis) + ' train gen loss {:.4f}'.format(total_loss_gen) + " costs gen " + time.strftime(
                        "%H: %M: %S", time.gmtime(time.time()-end_time_dis)) + " costs dis " + time.strftime("%H: %M: %S", time.gmtime(end_time_dis-start_time)))
    print('---'*18)

print('==='*18)
print("End. Best Epoch {:03d} ".format(best_epoch))
evaluate_utils.print_results(None, best_results, best_test_results)   
print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
