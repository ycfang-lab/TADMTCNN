import os,json,sys
sys.path.append('../../..')
import torch
import networks
import CelebA
import numpy as np
import pandas as pd
import torchvision
from torch.utils.data import DataLoader

tasks = ['5_o_Clock_Shadow','Attractive','Blurry','Chubby','Heavy_Makeup','Male','Oval_Face','Pale_Skin','Straight_Hair','Smiling','Wavy_Hair','Young', 'Arched_Eyebrows','Bags_Under_Eyes','Bald','Bangs','Black_Hair','Blond_Hair','Brown_Hair','Bushy_Eyebrows','Eyeglasses','Gray_Hair','Narrow_Eyes','Receding_Hairline','Wearing_Hat','Big_Nose','High_Cheekbones','Pointy_Nose','Rosy_Cheeks','Sideburns','Wearing_Earrings','Big_Lips','Double_Chin','Goatee','Mustache','Mouth_Slightly_Open','No_Beard','Wearing_Lipstick','Wearing_Necklace','Wearing_Necktie']

group_list = ['upper_group', 'middle_group', 'lower_group', 'whole_image_group']
task_group = {
    'upper_group': ['Arched_Eyebrows','Bags_Under_Eyes','Bald','Bangs','Black_Hair','Blond_Hair','Brown_Hair','Bushy_Eyebrows','Eyeglasses','Gray_Hair','Narrow_Eyes','Receding_Hairline','Wearing_Hat'],
    'middle_group': ['Big_Nose','High_Cheekbones','Pointy_Nose','Rosy_Cheeks','Sideburns','Wearing_Earrings'],
    'lower_group': ['Big_Lips','Double_Chin','Goatee','Mustache','Mouth_Slightly_Open','No_Beard','Wearing_Lipstick','Wearing_Necklace','Wearing_Necktie'],
    'whole_image_group': ['5_o_Clock_Shadow','Attractive','Blurry','Chubby','Heavy_Makeup','Male','Oval_Face','Pale_Skin','Straight_Hair','Smiling','Wavy_Hair','Young']
}

#=================================================================================================================
#辅助函数
def cal_acc(preds, labels):
    return torch.sum((preds==labels)).item()*1.0 / preds.size()[0]

def AdvFocusDMTL_pre_training(hyper_para):

    num_epoches = hyper_para['num_epoches']
    batch_size = hyper_para['batch_size']
    device = hyper_para['device']

    train_set = CelebA.CelebA_Dataset(hyper_para['img_dir'], hyper_para['eval_dir'], hyper_para['label_dir'], mode='train')
    test_set = CelebA.CelebA_Dataset(hyper_para['img_dir'], hyper_para['eval_dir'], hyper_para['label_dir'], mode='test')

    train_dataloader =  DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=6)
    test_dataloader =  DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=6)

    log_txt = hyper_para['log']

    model = networks.AdvFocusDMTL(group_list, hyper_para['backbone'], hyper_para['share_dis'], True, hyper_para['alpha_grain'], hyper_para['discrimator_level'])
    subnet_classifiers = CelebA.classifier(512, task_group)
    model = model.to(device)
    subnet_classifiers = subnet_classifiers.to(device)
    share_classifiers = CelebA.classifier(512, {'whole':tasks})
    share_classifiers = share_classifiers.to(device)

    if hyper_para["optim"] == "Adam":
        optimizer_all = torch.optim.Adam([{'params':model.parameters()}, {'params':subnet_classifiers.parameters()}, {'params':share_classifiers.parameters()}], lr=hyper_para['learning_rate'], betas=hyper_para['betas'])
    elif hyper_para["optim"] == "AdamW":
        optimizer_all = torch.optim.AdamW([{'params':model.parameters()}, {'params':subnet_classifiers.parameters()}, {'params':share_classifiers.parameters()}], lr=hyper_para['learning_rate'], betas=hyper_para['betas'], weight_decay=hyper_para['weight_decay'])
    elif hyper_para['optim'] == "RMSProp":
        optimizer_all = torch.optim.RMSprop([{'params':model.parameters()}, {'params':subnet_classifiers.parameters()}, {'params':share_classifiers.parameters()}], lr=hyper_para['learning_rate'], momentum=hyper_para['betas'][0], alpha=hyper_para['betas'][1], weight_decay=hyper_para['weight_decay'])
    elif hyper_para['optim'] == 'SGD':
        optimizer_all = torch.optim.SGD([{'params':model.parameters()}, {'params':subnet_classifiers.parameters()}, {'params':share_classifiers.parameters()}], lr=hyper_para['learning_rate'],momentum=hyper_para['betas'][0], weight_decay=hyper_para['weight_decay'], nesterov=hyper_para['nesterov'])
        if hyper_para['lr_scheduler'] == "step":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_all, hyper_para['step'], hyper_para['gamma'])
        elif hyper_para['lr_scheduler'] == "multi_step":
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_all, hyper_para['milestones'], hyper_para['gamma'])
        elif hyper_para['lr_scheduler'] == "expon":
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_all, hyper_para['gamma'])

    if not os.path.exists('./model'):
        os.mkdir('./model')
    bce_criterion = torch.nn.BCELoss()
    highest_acc = []
    highest_batch = []
    highest_avg_acc = 0
    result = pd.DataFrame()
    tot_batch = 0
    with open(log_txt, 'w') as log:
        for epoch in range(num_epoches):
            model.train()
            subnet_classifiers.train()
            for i_batch, sample_batched in enumerate(train_dataloader):
                log.write('=========================================================\n')
                print('=========================================================')
                log.write('epoch {} batch {}:\n'.format(epoch, i_batch))
                print('epoch {} batch {}:'.format(epoch, i_batch))
                log.write('batch_loss/batch_acc/highest_test_acc/highest_epoch\n')
                print('batch_loss/batch_acc/highest_test_acc/highest_epoch')
                sample = sample_batched['sample'].to(device)
                main_output, share_output = model(sample, pre_training=True)
                main_output = subnet_classifiers(main_output)
                share_output = share_classifiers({'whole':share_output})
                main_losses = {}
                acc = []
                #计算主要loss
                for task in main_output.keys():
                    output = torch.sigmoid(main_output[task])
                    label = sample_batched['label'][:,CelebA.CelebAcast2num_40(task)].to(device)
                    main_losses[task] = bce_criterion(output, label)
                    with torch.no_grad():
                        preds = (output.view(output.size()[0]) >= 0.5).float()
                        acc.append(cal_acc(preds, label))
                log.write('{}/{:.4f}/{:.4f}/{:.4f}\n'.format('subnets', sum(main_losses.values())/len(main_losses.values()), sum(acc)/len(acc), highest_avg_acc))
                print('{}/{:.4f}/{:.4f}/{:.4f}'.format('subnets', sum(main_losses.values())/len(main_losses.values()), sum(acc)/len(acc), highest_avg_acc))

                share_losses = {}
                acc = []
                #计算主要loss
                for task in share_output.keys():
                    output = torch.sigmoid(share_output[task])
                    label = sample_batched['label'][:,CelebA.CelebAcast2num_40(task)].to(device)
                    share_losses[task] = bce_criterion(output, label)
                    with torch.no_grad():
                        preds = (output.view(output.size()[0]) >= 0.5).float()
                        acc.append(cal_acc(preds, label))
                log.write('{}/{:.4f}/{:.4f}/\n'.format('share', sum(share_losses.values())/len(share_losses.values()), sum(acc)/len(acc)))
                print('{}/{:.4f}/{:.4f}'.format('share', sum(share_losses.values())/len(share_losses.values()), sum(acc)/len(acc)))

                total_main_loss = sum(main_losses.values())
                total_share_loss = sum(share_losses.values())

                optimizer_all.zero_grad()

                total_main_loss.backward()
                total_share_loss.backward()
                
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
                    subnet_classifiers.eval()
                    preds_list = {task:[] for task in tasks}
                    labels_list = {task:[] for task in tasks}            
                    for sample_batched in test_dataloader:
                        sample = sample_batched['sample'].to(device)
                        main_output,share_output = model(sample, pre_training=True)
                        main_output = subnet_classifiers(main_output)
                        for task in main_output.keys():
                            label = sample_batched['label'][:,CelebA.CelebAcast2num_40(task)].to(device)
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
                dfhigh = pd.DataFrame({'highest_acc': highest_acc, 'highest_batch': highest_batch})
                dfhigh.to_csv('./highest.csv', columns = ['highest_batch', 'highest_acc'], index = False)

                if hyper_para['optim'] == 'SGD':
                    lr_scheduler.step()


if __name__ == "__main__":
    with open(sys.argv[1],'r') as f:
        content = f.read()
        hyper_para = json.loads(content)
    AdvFocusDMTL_pre_training(hyper_para)
