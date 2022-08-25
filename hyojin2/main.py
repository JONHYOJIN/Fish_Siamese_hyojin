import constant as const
from model import SiameseNet
from data_loader import get_train_validation_loader, get_test_loader
from one_cycle_policy import OneCyclePolicy

import os
import torch
import torch.optim as optim
import torch.nn as nn
from glob import glob
from tqdm import tqdm
from utils.train_utils import AverageMeter


def train():
    # Train set & Validation set 로드
    trainloader, valloader = get_train_validation_loader()
    
    # Model 생성
    model = SiameseNet()

    # Optimizer & Loss 선언
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()  # -> Binary Cross Entropy

    # Best epoch, val_acc 등 선언
    best_epoch = 0
    start_epoch = 0
    best_valid_acc = 0
    # Optimizer Momentum, Learning_rate 조절 (L2 regularization;weight decay는 적용 못 함)
    one_cycle = OneCyclePolicy(optimizer, num_steps=const.EPOCHS, momentum_range=(0.5, 1), lr_range=(1e-4, 1e-3))

    # Validation acc가 줄지 않은 횟수
    counter = 0

    # Batch 수
    num_train = len(trainloader)
    num_valid = len(valloader)

    # 학습
    main_pbar = tqdm(range(start_epoch, const.EPOCHS), # epoch : 0 ~ 50
                    initial=start_epoch, 
                    # position=0, 
                    total=const.EPOCHS, 
                    desc="Process")

    for epoch in main_pbar:
        # Losses
        train_losses = AverageMeter()
        valid_losses = AverageMeter()

        # Train
        model.train()
        train_pbar = tqdm(enumerate(trainloader), total=num_train, desc="Train", position=1, leave=False)
        for i, (x1, x2, y) in train_pbar:
            out = model(x1, x2)
            loss = criterion(out, y.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.update(loss.item(), x1.shape[0])

            train_pbar.set_postfix_str(f"loss: {train_losses.val:0.3f}")
        one_cycle.step()

        # Validate
        model.eval()
        valid_acc = 0
        correct_sum = 0
        valid_pbar = tqdm(enumerate(valloader), total=num_valid, desc="Validation", position=1, leave=False)
        with torch.no_grad():
            for i, (x1, x2, y) in valid_pbar:
                out = model(x1, x2)
                loss = criterion(out, y.unsqueeze(1))

                diffs = torch.abs(y-out.reshape(-1))
                for diff in diffs:
                    if diff < 0.5:
                        correct_sum+=1
                
                valid_losses.update(loss.item(), x1.shape[0])

                valid_acc = correct_sum / len(valloader.dataset)
                valid_pbar.set_postfix_str(f"accuracy: {valid_acc:0.3f}")
        
        # Best Model 체크 후 저장
        if valid_acc > best_valid_acc:
            is_best = True
            best_valid_acc = valid_acc
            best_epoch = epoch
            counter=0
        else:
            is_best=False
            counter+=1
        if is_best or epoch==const.EPOCHS:
            save_checkpoint({
                'epoch' : epoch,
                'model_state' : model.state_dict(),
                'optim_state' : optimizer.state_dict(),
                'best_valid_acc' : best_valid_acc,
                'best_epoch' : best_epoch
            }, is_best)

        main_pbar.set_postfix_str(f"best acc: {best_valid_acc:.3f} best epoch: {best_epoch} ")

        tqdm.write(
                f"[{epoch}] train loss: {train_losses.avg:.3f} - valid loss: {valid_losses.avg:.3f} - valid acc: {valid_acc:.3f} {'[BEST]' if is_best else ''}")

def test():
    # Best Model 로드
    model = SiameseNet()
    model.load_state_dict(torch.load('./best_model.pt')['model_state'])

    # Test set 로드
    test_loader = get_test_loader()

    correct_sum = 0
    num_test = len(test_loader.dataset)

    # Test
    pbar = tqdm(enumerate(test_loader), total=num_test, desc="Test")
    with torch.no_grad():
        for i, (x1, x2, y) in pbar:
            out = model(x1, x2)
            
            diffs = torch.abs(y-out.reshape(-1))
            for diff in diffs:
                if diff < 0.5:
                    correct_sum+=1

            pbar.set_postfix_str(f"accuracy: {correct_sum / num_test}")
    test_acc = correct_sum / num_test
    print(f"Test Acc: {correct_sum}/{num_test} ({test_acc:.2f}%)")

    return out


def save_checkpoint(state, is_best):
    if is_best:
        filename='./best_model.pt'
    else:
        filename=f'./model_ckpt{state["epochs"]}.pt'
    
    torch.save(state, filename)

def load_checkpoint(best):
    if best:
        model_path = './best_model.pt'
    else:
        model_path = sorted(glob('./models/model_ckpt_*.pt'), key=len)[-1]

    ckpt = torch.load(model_path)

    if best:
        print(
            f"Loaded {os.path.basename(model_path)} checkpoint @ epoch {ckpt['epoch']} with best valid acc of {ckpt['best_valid_acc']:.3f}")
    else:
        print(f"Loaded {os.path.basename(model_path)} checkpoint @ epoch {ckpt['epoch']}")

    return ckpt['epoch'], ckpt['best_epoch'], ckpt['best_valid_acc'], ckpt['model_state'], ckpt['optim_state']

if __name__=='__main__':
    train()
    result = test()
    print(result)