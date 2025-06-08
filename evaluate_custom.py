import numpy as np
import torch
import argparse
import os
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import wandb
import datetime
from torch.utils.data import DataLoader, TensorDataset
from huggingface_hub import hf_hub_download
import zipfile

from data import load, load_multiple, load_custom_data, load_custom_data_per_participant
from utils import compute_metrics_np, set_random_seed, select_participants, accumulate_participant_files
from contrastive import ContrastiveModule
import pickle as cp

def main(args):

    if args.case_study == 'cv':
        if args.dataset != "C24":
            _, _, test_users = select_participants(args.users_list, args.special_participant_list,
                                                                0.8, test=True, loocv=args.loocv, round=args.round)
        else: # no need for test participants for C24. 
            # _, _, test_users = select_participants(args.users_list, args.special_participant_list,
            #                                                     0.8, test=False, loocv=args.loocv, round=args.round)
            test_users = args.users[args.dataset][1]
    else:
        if args.dataset != "C24":
            test_users = args.users[args.dataset][0]

        else:
            test_users = args.users[args.dataset][1]
    print("test_users are", test_users)

    if args.case_study == 'cv':
        checkpoint = os.path.join("./checkpoint", args.dataset, f'{args.dataset}_{args.round}_best_loss.pth')
    else:
        checkpoint = os.path.join("./checkpoint", args.train_dataset, f'{args.train_dataset}_0_best_loss.pth')
    print("Loading pretrained model from", checkpoint)
    model = ContrastiveModule(args).cuda()
    model.model.load_state_dict(torch.load('./checkpoint/UniMTS.pth'))
    model.load_state_dict(torch.load(f'{checkpoint}'))

    cms = []                                                
    for user in test_users:
        print("Participant", user)
        ## Test data
        data_test, labels_test = accumulate_participant_files(args, [user])

        real_inputs, real_masks, real_labels, label_list, all_text = load_custom_data_per_participant(
            data_test, labels_test, args.config_path, args.joint_list, args.original_sampling_rate, padding_size=args.padding_size, split='test', k=args.k, few_shot_path=None
        )

        real_dataset = TensorDataset(real_inputs, real_masks, real_labels)
        test_real_dataloader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False)

        model.eval()
        with torch.no_grad():
            pred_whole, logits_whole = [], []
            for input, mask, label in test_real_dataloader:
                
                input = input.cuda()
                mask = mask.cuda()
                label = label.cuda()

                if not args.gyro:
                    b, t, c = input.shape
                    indices = np.array([range(i, i+3) for i in range(0, c, 6)]).flatten()
                    input = input[:,:,indices]

                b, t, c = input.shape
                if args.stft:
                    input_stft = input.permute(0,2,1).reshape(b * c,t)
                    input_stft = torch.abs(torch.stft(input_stft, n_fft = 25, hop_length = 28, onesided = False, center = True, return_complex = True))
                    input_stft = input_stft.reshape(b, c, input_stft.shape[-2], input_stft.shape[-1]).reshape(b, c, t).permute(0,2,1)
                    input = torch.cat((input, input_stft), dim=-1)

                input = input.reshape(b, t, 22, -1).permute(0, 3, 1, 2).unsqueeze(-1)
                
                logits_per_imu = model.classifier(input)
                logits_whole.append(logits_per_imu)
                
                pred = torch.argmax(logits_per_imu, dim=-1).detach().cpu().numpy()
                pred_whole.append(pred)

            pred = np.concatenate(pred_whole)
            acc = accuracy_score(real_labels, pred)
            prec = precision_score(real_labels, pred, average='macro')
            rec = recall_score(real_labels, pred, average='macro')
            f1 = f1_score(real_labels, pred, average='macro')

            all_classes = np.arange(args.num_class)  
            conf_matrix = confusion_matrix(real_labels, pred, labels=all_classes)
            print(f"acc: {acc}, prec: {prec}, rec: {rec}, f1: {f1}")
            logits_whole = torch.cat(logits_whole)
            r_at_1, r_at_2, r_at_3, r_at_4, r_at_5, mrr_score = compute_metrics_np(logits_whole.detach().cpu().numpy(), real_labels.numpy())
            print(f"R@1: {r_at_1}, R@2: {r_at_2}, R@3: {r_at_3}, R@4: {r_at_4}, R@5: {r_at_5}, MRR: {mrr_score}")        
            cms.append(conf_matrix)
    
    return cms

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Unified Pre-trained Motion Time Series Model')

    # data
    parser.add_argument('--padding_size', type=int, default='200', help='padding size (default: 200)')
    parser.add_argument('--config_path', type=str, required=True, help='/path/to/config/')
    parser.add_argument('--joint_list', nargs='+', type=int, required=True, help='List of joint indices')
    parser.add_argument('--original_sampling_rate', type=int, required=True, help='original sampling rate')
    parser.add_argument('--num_class', type=int, required=True, help='number of classes')
    parser.add_argument('--k', type=int, help='few shot samples per class (default: None)')

    parser.add_argument('--case_study', type=str, default='cv', choices=['cv','d2d'], help='the case I am running')
    # parser.add_argument('--dataset', type=str, help='Dataset name', choices=['HHAR', 'DSA', 'MHEALTH', 'selfBACK', 'PAMAP2', 'GOTOV', 'C24'])
    parser.add_argument('--stage', type=str, default='finetune', help='training stage')

    # training
    parser.add_argument('--gyro', type=int, default=0, help='using gyro or not')
    parser.add_argument('--stft', type=int, default=0, help='using stft or not')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')

    
    args = parser.parse_args()

    args.users = {'C24': [[25, 27, 30,
                         47, 48, 49, 50, 51, 52, 53,
                         54, 56, 57, 59, 60, 61, 62, 64, 67, 68, 69, 71, 72, 73, 76, 77, 79, 80,
                         91, 92, 96, 97, 98, 99, 100, 102, 103, 104,
                         105, 106, 107, 109, 110, 111, 112, 113, 115, 117, 118, 119, 120, 122,
                         125, 128, 130, 131, 132, 133, 134, 135, 139, 141, 142, 143, 144, 145,
                         146, 148, 150], 
                         [1, 9, 10, 12, 13, 14, 16, 17, 19, 23, 24, 26, 28,
                         29, 31, 32, 34, 37, 41, 55, 58, 63, 65, 66, 70, 74, 75, 78, 87, 89, 90,
                         93, 94, 95, 101, 108, 114, 116, 121, 123, 124, 126, 127, 129, 136, 137,
                         138, 140, 147, 149, 151]],
                 'MHEALTH':[[i for i in range(1,11)]],
                 'DSA':[[i for i in range(1,9)]],
                 'GOTOV':[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                              23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]],
                 'HHAR':[[i for i in range(1,10)]],
                 'PAMAP2':[[i for i in range(1,9)]],
                 'selfBACK':[[26, 27, 28, 29, 
                              30, 31, 33, 34, 36, 39,
                              40, 41, 42, 43, 44, 46, 47, 48, 49,
                              50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                              60, 61, 62, 63]]}
    args.special_participants = { 'C24': [[12,47,98,113,131,132],[25,48,127,128,143]],}
    
    
    all_datasets = ['C24','DSA','HHAR', 'MHEALTH', 'selfBACK', 'PAMAP2', 'GOTOV']

    if args.case_study == "cv":
        for args.dataset in all_datasets:
            set_random_seed(10)
            args.loocv = False
            args.round = 0
            if args.dataset in args.special_participants.keys():
                args.special_participant_list = args.special_participants[args.dataset]
            else:
                args.special_participant_list = []
            print("Working on", args.dataset)
            cm_round_filename = os.path.join("results","evaluation_results", args.case_study, args.dataset)
            os.makedirs(cm_round_filename, exist_ok=True)
            cms_self = []
            
            users = args.users[args.dataset]
            num_participants = len(users[0])
            if len(users) == 1: # in lab datasets
                args.total_rounds = min(10, num_participants)
                if num_participants <= 10:                
                    print("Applying L.O.O.CV.")
                    args.loocv = True
            else: # This is C24
                args.total_rounds = 1 # just one round with the presplit C24, 100 for training, 51 for testing        
                print("C24, only doing one round...")  

            args.users_list = users[0]

            for round in range(args.total_rounds):
                print("ROUND ", round)   
                args.round = round
                cms = main(args)
                for cm in cms:
                    cms_self.append(cm)
            
            
            f = open(os.path.join(cm_round_filename, "self.cms"), 'wb')
            cp.dump(cms_self, f, protocol=cp.HIGHEST_PROTOCOL)
            f.close()


    elif args.case_study == "d2d":
        for args.train_dataset in all_datasets:
            print("Trained on", args.train_dataset)
            if args.train_dataset != 'C24':
                # in 2 in
                test_datasets = [ds for ds in all_datasets if ds != args.train_dataset and ds != 'C24']
                cm_round_filename = os.path.join("results","evaluation_results", args.case_study, args.train_dataset)
                os.makedirs(cm_round_filename, exist_ok=True)
                cms_in2in = []
                for args.dataset in test_datasets:
                    print("Evaluating on", args.dataset)
                    cms = main(args)
                    for cm in cms:
                        cms_in2in.append(cm)
                
                f = open(os.path.join(cm_round_filename, "in2in.cms"), 'wb')
                cp.dump(cms_in2in, f, protocol=cp.HIGHEST_PROTOCOL)
                f.close()

                #### in 2 out 

                args.dataset = 'C24'
                cms_in2out = []
                print("Evaluating on", args.dataset)
                cms = main(args)
                for cm in cms:
                    cms_in2out.append(cm)

                f = open(os.path.join(cm_round_filename, "in2out.cms"), 'wb')
                cp.dump(cms_in2out, f, protocol=cp.HIGHEST_PROTOCOL)
                f.close()
            else:
                # out 2 in
                test_datasets = [ds for ds in all_datasets if ds != args.train_dataset]
                cm_round_filename = os.path.join("results","evaluation_results", args.case_study, args.train_dataset)
                os.makedirs(cm_round_filename, exist_ok=True)
                cms_out2in = []
                for args.dataset in test_datasets:
                    print("Evaluating on", args.dataset)
                    cms = main(args)
                    for cm in cms:
                        cms_out2in.append(cm)
                f = open(os.path.join(cm_round_filename, "out2in.cms"), 'wb')
                cp.dump(cms_out2in, f, protocol=cp.HIGHEST_PROTOCOL)
                f.close()
