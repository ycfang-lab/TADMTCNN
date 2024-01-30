import os,json,sys
sys.path.append('../../..')
import torch
import networks
import cifar10
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import pandas as pd
from pandas import DataFrame



tasks = ['airplane',
 'automobile',
 'bird',
 'cat',
 'deer',
 'dog',
 'frog',
 'horse',
 'ship',
 'truck']

group_list = ['airplane_ship', 'automobile_truck', 'bird_frog', 'cat_dog', 'deer_horse']
task_group = {
    'airplane_ship': ['airplane', 'ship'],
    'automobile_truck': ['automobile', 'truck'],
    'bird_frog': ['bird', 'frog'],
    'cat_dog': ['cat', 'dog'],
    'deer_horse': ['deer', 'horse']
}

#=================================================================================================================
#辅助函数
def cal_acc(preds, labels):
    cor = preds[preds == labels]
    elem = set(cor.cpu().numpy())
    acc = []
    for e in elem:
        pre = len(cor[cor == e])
        tru = len(labels[labels == e])
        if tru > 0:
            acc.append(pre / tru)
    if len(acc) == 0:
        return 0
    return sum(acc) / len(acc)
#=================================================================================================================
#训练函数
def AdvFocusDMTL_training(hyper_para):

    num_epoches = hyper_para['num_epoches']
    batch_size = hyper_para['batch_size']
    device = hyper_para['device']

    # image_aug = torchvision.transforms.Compose(
    #     [
    #         torchvision.transforms.RandomHorizontalFlip(),
    #         torchvision.transforms.RandomVerticalFlip(),
    #         torchvision.transforms.RandomRotation((0,360)),
    #         torchvision.transforms.ToTensor()
    #     ]
    # )
    label_names = list(map(bytes.decode, cifar10.unpickle(os.path.join(hyper_para['cifar10_dir'], 'batches.meta'))[b'label_names']))
    train_set = cifar10.cifar10(hyper_para['cifar10_dir'], mode='train')
    test_set = cifar10.cifar10(hyper_para['cifar10_dir'], mode='test')

    train_dataloader =  DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=6)
    test_dataloader =  DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=6)

    log_txt = hyper_para['log']
    task_labels = {group:[i] for i,group in enumerate(group_list)}

    model = networks.PSMCNN(group_list, hyper_para['backbone'], hyper_para['alpha_grain'])
    if hyper_para['pre_trained']:
        model.load_state_dict(torch.load(hyper_para['pre_trained_model']))
        print('loaded!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    classifiers = cifar10.classifier(512, task_group)
    model = model.to(device)
    classifiers = classifiers.to(device)

    
    if hyper_para["optim"] == "Adam":
        optimizer_all = torch.optim.Adam([{'params':model.parameters()}, {'params':classifiers.parameters()}], lr=hyper_para['learning_rate'], betas=hyper_para['betas'])
    elif hyper_para["optim"] == "AdamW":
        optimizer_all = torch.optim.AdamW([{'params':model.parameters()}, {'params':classifiers.parameters()}], lr=hyper_para['learning_rate'], betas=hyper_para['betas'], weight_decay=hyper_para['weight_decay'])
    elif hyper_para['optim'] == "RMSProp":
        optimizer_all = torch.optim.RMSprop([{'params':model.parameters()}, {'params':classifiers.parameters()}], lr=hyper_para['learning_rate'], momentum=hyper_para['betas'][0], alpha=hyper_para['betas'][1], weight_decay=hyper_para['weight_decay'])
    elif hyper_para['optim'] == 'SGD':
        optimizer_all = torch.optim.SGD([{'params':model.parameters()}, {'params':classifiers.parameters()}], lr=hyper_para['learning_rate'],momentum=hyper_para['betas'][0], weight_decay=hyper_para['weight_decay'], nesterov=hyper_para['nesterov'])
        if hyper_para['lr_scheduler'] == "step":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_all, hyper_para['step'], hyper_para['gamma'])
        elif hyper_para['lr_scheduler'] == "multi_step":
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_all, hyper_para['milestones'], hyper_para['gamma'])
        elif hyper_para['lr_scheduler'] == "expon":
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_all, hyper_para['gamma'])

    bce_criterion = torch.nn.BCELoss()  
    highest_acc = []
    highest_batch = []
    highest_avg_acc = 0
    result = pd.DataFrame()
    tot_batch = 0

    if not os.path.exists('./TTC'):
        os.mkdir('./TTC')
    if not os.path.exists('./model'):
        os.mkdir('./model')
    with open(log_txt, 'w') as log:
        for epoch in range(num_epoches):
            model.train()
            classifiers.train()
            for i_batch, (samples, labels) in enumerate(train_dataloader):
                log.write('=========================================================\n')
                print('=========================================================')
                log.write('epoch {} batch {}:\n'.format(epoch, i_batch))
                print('epoch {} batch {}:'.format(epoch, i_batch))
                log.write('batch_loss/batch_acc/highest_test_acc/highest_epoch\n')
                print('batch_loss/batch_acc/highest_test_acc/highest_epoch')
                samples = samples.to(device)
                main_output = model(samples)
                main_output = classifiers(main_output)
                main_losses = {}
                acc = []
                #计算主要loss
                for task in main_output.keys():
                    output = torch.sigmoid(main_output[task]).squeeze()
                    label = (labels == label_names.index(task)).type(torch.FloatTensor).to(device)
                    main_losses[task] = bce_criterion(output, label)
                    with torch.no_grad():
                        preds = (output.view(output.size()[0]) >= 0.5).float()
                        acc.append(cal_acc(preds, label))
                log.write('{}/{:.4f}/{:.4f}/{:.4f}\n'.format('avg', sum(main_losses.values())/len(main_losses.values()), sum(acc)/len(acc), highest_avg_acc))
                print('{}/{:.4f}/{:.4f}/{:.4f}'.format('avg', sum(main_losses.values())/len(main_losses.values()), sum(acc)/len(acc), highest_avg_acc))
                
                total_loss = sum(main_losses.values())
                
                optimizer_all.zero_grad()
                total_loss.backward()
                optimizer_all.step()
            
                tot_batch += 1

                if tot_batch % 10 > 0:
                    continue

                log.write('=======================================================================================\n')
                print('=======================================================================================')
                log.write('Testing batch:{}\n'.format(tot_batch))
                print('Testing batch:{}'.format(tot_batch))

                curresult = {}

                with torch.no_grad():
                    model.eval()
                    classifiers.eval()
                    preds_list = {task:[] for task in tasks}
                    labels_list = {task:[] for task in tasks}           
                    for samples, labels in test_dataloader:
                        samples = samples.to(device)
                        main_output = model(samples)
                        main_output = classifiers(main_output)
                        for task in main_output.keys():
                            label = (labels == label_names.index(task)).type(torch.FloatTensor).to(device)
                            output = torch.sigmoid(main_output[task])
                            preds = (output.view(output.size()[0]) >= 0.5).float()
                            preds_list[task].append(preds)
                            labels_list[task].append(label)
                    for task in tasks:
                        acc = cal_acc(torch.cat(preds_list[task]), torch.cat(labels_list[task]))
                        curresult[task] = acc

                avg_acc = sum(curresult.values()) / len(curresult)
                if avg_acc > highest_avg_acc:
                    highest_avg_acc = avg_acc
                    highest_acc.append(avg_acc)
                    highest_batch.append(tot_batch)
                    torch.save(model.state_dict(), os.path.join('./model', 'highest'))
                curresult['average'] = avg_acc
                result = result.append(curresult, ignore_index = True)
                result.to_csv('result.csv', index = None)
                torch.save(model.layer_alpha, os.path.join('./TTC', '{}'.format(tot_batch)))
                dfhigh = pd.DataFrame({'highest_acc': highest_acc, 'highest_batch': highest_batch})
                dfhigh.to_csv('./highest.csv', columns = ['highest_batch', 'highest_acc'], index = False)
                if hyper_para['optim'] == 'SGD':
                    lr_scheduler.step()
            
    torch.save(model.state_dict(), os.path.join('./model', hyper_para['save_path']))
    return model

if __name__ == "__main__":
    with open(sys.argv[1],'r') as f:
        content = f.read()
        hyper_para = json.loads(content)
    AdvFocusDMTL_training(hyper_para)




 