import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from util import load_data, separate_data
from models.graphcnn import GraphCNN
import copy

criterion = nn.CrossEntropyLoss()

def train(args, model, device, train_graphs, optimizer, epoch, spec_iter=0):
    model.train()

    total_iters = args.iters_per_epoch
    if spec_iter > 0:
        total_iters = spec_iter
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        #compute loss
        loss = criterion(output, labels)

        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()
        

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        #report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum/total_iters
    print("loss training: %f" % (average_loss))
    
    return average_loss

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def test(args, model, device, train_graphs, test_graphs, epoch):
    model.eval()

    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    print("accuracy train: %f test: %f" % (acc_train, acc_test))

    return acc_train, acc_test

def min_min_attack(train_graphs, model, noise, tags, rounds):
    def flip(bnoise, idx):
        if bnoise[idx]==1:
            bnoise[idx] = 0
        elif bnoise[idx]==0:
            bnoise[idx] = 1
        else:
            print('Error: non binary noise.')

    model.eval()
    
    selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]
    batch_graph = [train_graphs[idx] for idx in selected_idx]
    batch_noise = [noise[idx] for idx in selected_idx]
    batch_tags = [tags[idx] for idx in selected_idx]
    for _ in range(rounds):
        for i in range(len(batch_graph)):
            # For each graph, do numeric gradient
            prev_loss = float('inf')
            temp_graph = copy.deepcopy(batch_graph[i])
            cur_tag = batch_tag[i]
            cur_noise = batch_noise[i]
            for dim in len(temp_graph.g):
                flip(cur_noise, dim)
                new_graph = [temp_graph.add_noise(cur_noise, cur_tag)]
                output = model(new_graph)
                labels = torch.LongTensor([graph.label for graph in new_graph]).to(device)
                # compute loss
                loss = criterion(output, labels)
                # if worse, flip back
                if loss > prev_loss:
                    flip(cur_noise, dim)
    return 0

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true",
    					help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type = str, default = "",
                                        help='output file')
    args = parser.parse_args()

    #set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    graphs, num_classes, tagset = load_data(args.dataset, args.degree_as_tag)

    ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)

    model = GraphCNN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    #########################################
    #
    #
    # Genereate Noise
    #
    #
    #########################################

    condition = True
    noise = [([1]*(len(g.g))) for g in train_graphs] # values exclude self

    df_tags = [graph.label for graph in train_graphs]
    tag_score_dict = [tag:[1000.0]*len(train_graphs) for tag in tagset]
    best_tag = [-1]*len(train_graphs)
    while condition:
        nsd_train_graphs = copy.deepcopy(train_graphs)
        nsd_train_graphs = [nsd_train_graphs[idx].add_noise()]
        scheduler.step()
        train(args, model, device, nsd_train_graphs, optimizer, epoch, 30)


        pbar = tqdm(range(args.iters_per_epoch), unit='batch')
        for pos in pbar:
            min_min_attack(train_graphs, model, noise, df_tags, 20)
            pbar.set_description('Noise Training...')

        acc_train, acc_test = test(args, model, device, train_graphs, test_graphs, epoch)
        print("Train Accuracy {:2.2%}, Test Accuracy {:2.2%}".format(acc_train, acc_test))
        if acc_train > 0.99:
            condition = False

    poisoned_train_graphs = [train_graphs[i].add_noise(noise[i], df_tags[i]) for i in range(len(train_graphs))]
    train_graphs = poisoned_train_graphs
    #########################################
    #
    #
    # Finish Noise Generation
    #
    #
    #########################################




    for epoch in range(1, args.epochs + 1):
        scheduler.step()

        avg_loss = train(args, model, device, train_graphs, optimizer, epoch)
        acc_train, acc_test = test(args, model, device, train_graphs, test_graphs, epoch)

        if not args.filename == "":
            with open(args.filename, 'w') as f:
                f.write("%f %f %f" % (avg_loss, acc_train, acc_test))
                f.write("\n")
        print("")

        print(model.eps)
    

if __name__ == '__main__':
    main()
