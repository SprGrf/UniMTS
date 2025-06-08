import numpy as np
import torch
import torch.nn.functional as F

import argparse
import os
import numpy as np
import wandb
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from data import load_custom_data_per_participant
from utils import set_random_seed, select_participants, accumulate_participant_files
from contrastive import ContrastiveModule


def main(args):

    if args.dataset != "C24":
        train_users, valid_users, _ = select_participants(args.users_list, args.special_participant_list,
                                                            0.8, test=True, loocv=args.loocv, round=args.round)
    else: # no need for test participants for C24. 
        train_users, valid_users, _ = select_participants(args.users_list, args.special_participant_list,
                                                                0.8, test=False, loocv=args.loocv, round=args.round)

    ## Train data
    data_train, labels_train = accumulate_participant_files(args, train_users)

    train_inputs, train_masks, train_labels, _, _ = load_custom_data_per_participant(
        data_train, labels_train, args.config_path, args.joint_list, args.original_sampling_rate, padding_size=args.padding_size, split='train', k=args.k, few_shot_path=None
    )
    train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    ## Validation data
    data_val, labels_val = accumulate_participant_files(args, valid_users)

    val_inputs, val_masks, val_labels, _, _ = load_custom_data_per_participant(
        data_val, labels_val, args.config_path, args.joint_list, args.original_sampling_rate, padding_size=args.padding_size, split='val', k=args.k, few_shot_path=None
    )

    val_dataset = TensorDataset(val_inputs, val_masks, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    save_path = './checkpoint/%s/' % args.dataset
    os.makedirs(save_path, exist_ok=True)

    model = ContrastiveModule(args).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    if args.mode == 'full' or args.mode == 'probe':
        model.model.load_state_dict(torch.load(f'{args.checkpoint}'))
    if args.mode == 'probe':
        for name, param in model.model.named_parameters():
            param.requires_grad = False
    
    best_loss = None
    for epoch in range(args.num_epochs):

        tol_loss = 0
        model.train()
        for i, (input, mask, label) in enumerate(train_dataloader):

            input = input.cuda()
            labels = label.cuda()

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
            output = model.classifier(input)
            
            loss = F.cross_entropy(output.float(), labels.long(), reduction="mean")
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tol_loss += len(input) * loss.item()
                
        # Validation
        tol_val_loss = 0
        model.eval()
        with torch.no_grad():
            for input, mask, label in val_dataloader:
                input = input.cuda()
                labels = label.cuda()

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

                val_output = model.classifier(input)
                val_loss = F.cross_entropy(val_output.float(), labels.long(), reduction="mean")
                tol_val_loss += len(input) * val_loss.item()
        
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Training Loss: {tol_loss / len(train_dataset):.4f}, Validation Loss: {tol_val_loss / len(val_dataset):.4f}')

        if best_loss is None or tol_val_loss < best_loss:
            print("Saving model...")
            best_loss = tol_val_loss
            torch.save(model.state_dict(), os.path.join(save_path, f'{args.dataset}_{args.round}_best_loss.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Pre-trained Motion Time Series Model')

    # model 
    parser.add_argument('--mode', type=str, default='full', choices=['random','probe','full'], help='full fine-tuning, linear probe, random init')
    parser.add_argument('--case_study', type=str, default='cv', choices=['cv'], help='the case I am running')

    # data
    parser.add_argument('--padding_size', type=int, default='200', help='padding size (default: 200)')
    parser.add_argument('--k', type=int, help='few shot samples per class (default: None)')
    parser.add_argument('--config_path', type=str, required=True, help='/path/to/config/')
    parser.add_argument('--few_shot_path', type=str, help='/path/to/few/shot/indices/')
    parser.add_argument('--joint_list', nargs='+', type=int, required=True, help='List of joint indices')
    parser.add_argument('--original_sampling_rate', type=int, required=True, help='original sampling rate')
    parser.add_argument('--num_class', type=int, required=True, help='number of classes')

    # training
    parser.add_argument('--stage', type=str, default='finetune', help='training stage')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of fine-tuning epochs (default: 200)')
    parser.add_argument('--run_tag', type=str, default='exp0', help='logging tag')
    parser.add_argument('--gyro', type=int, default=0, help='using gyro or not')
    parser.add_argument('--stft', type=int, default=0, help='using stft or not')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')

    parser.add_argument('--checkpoint', type=str, default='./checkpoint/UniMTS.pth', help='/path/to/checkpoint/')
    
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

    all_datasets = ['HHAR', 'DSA', 'MHEALTH', 'selfBACK', 'PAMAP2', 'GOTOV', 'C24']

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
                main(args)
    