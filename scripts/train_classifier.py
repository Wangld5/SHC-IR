from audioop import cross
from pandas import test
import torch
import torch.nn.functional as F
from model.Net import AlexNetClass, VGGNetClass, weightConstrain
from scripts.head import *

class myLoss(nn.Module):
    def __init__(self):
        super(myLoss, self).__init__()
        self.CELoss = torch.nn.CrossEntropyLoss().cuda()

    def forward(self, probs, labels, w):
        celoss = self.CELoss(probs, labels)
        Q_loss = (w.abs()-1).pow(2).mean()
        return celoss + Q_loss

def top_k_accuracy(output, target, k=1):
    with torch.no_grad():
        _, predicted = torch.max(output.data, 1)
        total_correct = (predicted == target).sum().item()
        total = target.shape[0]
    return total_correct, total

def test_val(config, model, test_loader):
    model.eval()
    acc = 0
    total = 0
    with torch.no_grad():
        for img, label in tqdm(test_loader):
            img = img.cuda()
            label = label.cuda()
            preds = model(img)
            temp_acc,  temp_batch= top_k_accuracy(preds, label, k=1)
            acc += temp_acc
            total += temp_batch
    return 100 * acc / total

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    print(f"lr is {lr}")
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_val(config):
    device = config['device']
    if config['dataset'] == 'imagenet':
        train_loader, test_loader, database_loader, num_train, num_test, num_database = get_imagenet_data(config)
    else:
        train_loader, test_loader, database_loader, num_train, num_test, num_database = get_data(config)
    net = ResNetClass(config['n_class']).cuda()
    if config['n_gpu'] > 1:
        net = torch.nn.DataParallel(net)

    optimizer = optim.SGD(net.parameters(), lr=config['optim_parms']['lr'], weight_decay=config['optim_parms']['weight_decay'])
    config['num_train'] = num_train

    Best_acc = 0
    print('finish load config')

    count = 0
    logger.info(f"config: {str(config)}")
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    cross_entropy_loss = myLoss()

    for epoch in range(config['epoch']):
        current_time = time.strftime("%H:%M:%S", time.localtime(time.time()))
        logger.info(
            f"{epoch + 1}/{config['epoch']} {current_time} dataset: {config['dataset']} training...")
        # adjust_learning_rate(optimizer, epoch, config['optim_parms']['lr'])
        train_loss = 0
        train_acc = 0
        total = 0
        net.train()
        for img, label in tqdm(train_loader):
            img = img.cuda()
            label = label.cuda()
            optimizer.zero_grad()

            probs = net(img)
            loss = cross_entropy_loss(probs, label, net.model_resnet.fc.weight)
            temp_acc, temp_batch= top_k_accuracy(probs, label, k=1)
            train_acc += temp_acc
            total += temp_batch
            if config['n_gpu'] > 1:
                loss = loss.mean()
            train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), config['max_norm'])
            optimizer.step()

        train_loss /= len(train_loader)
        train_acc /= total
        logger.info(f"train loss: {train_loss}, train accuracy: {100 * train_acc}")

        if (epoch + 1) % config['test_map'] == 0:
            acc = test_val(config, net, test_loader)
            if acc > Best_acc:
                Best_acc = acc
                count = 0
                logger.info(f'save in ./results/class_model')
                torch.save(net.state_dict(), f'./results/class_model/{config["cls_model"]}_{config["dataset"]}_model_w_{config["optim_parms"]["lr"]}.pt')
                net.eval()
                with torch.no_grad():
                    W = net.fc.weight.cpu().numpy()
                np.save(f'./weight/{config["cls_model"]}_{config["dataset"]}_class_head_{config["optim_parms"]["lr"]}.npy', W)
            else:
                if count == config['stop_iter']:
                    logger.info(f"valid acc: {Best_acc}")
                    end_time = time.strftime("%H:%M:%S", time.localtime(time.time()))
                    with open(f'./results/class_model/map_result.txt', 'a') as f:
                        f.write('valid: ' + str(Best_acc) + '\t' + 'start time: ' + str(start_time) +
                                '\t' + 'end_time:' + str(end_time) + str(config) +'\n')

            
                    break
                count += 1
            logger.info(
                f"{epoch + 1}/{config['epoch']} {current_time} dataset: {config['dataset']} Best acc: {Best_acc}, current acc: {acc}")
        if (epoch + 1) == config['epoch']:
            logger.info(f"valid acc: {Best_acc}")
            end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            with open(f'./results/class_model/map_result.txt', 'a') as f:
                f.write('valid: ' + str(Best_acc) + '\t' + 'start time: ' + str(start_time) +
                        '\t' + 'end_time:' + str(end_time) + str(config) +'\n')
            logger.info(f'save in ./results/class_model')
            torch.save(net.state_dict(), f'./results/class_model/{config["cls_model"]}_{config["dataset"]}_model_w_{config["optim_parms"]["lr"]}.pt')
            net.eval()
            with torch.no_grad():
                W = net.fc.weight.cpu().numpy()
            np.save(f'./weight/{config["cls_model"]}_{config["dataset"]}_class_head_{config["optim_parms"]["lr"]}.npy', W)
