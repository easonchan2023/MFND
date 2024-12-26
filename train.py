import PIL
import numpy as np
import argparse
import time, os

from PIL import Image

# import random
import process_data_weibo2 as process_data
import copy
import pickle as pickle
from random import sample
import torchvision
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import scipy.io as scio
from torch.nn.utils.rnn import pack_padded_sequence

import torchvision.datasets as dsets
import torchvision.transforms as transforms

from sklearn import metrics
from transformers import AutoTokenizer, AutoModelWithLMHead, XLNetConfig
from loss import SupConLoss

from sklearn import metrics
from transformers import *
import warnings
import clip
#import torchkeras
import timm
import coattention
import math
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel


warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def tsne(input, labels):
    plt.figure(figsize=(10, 5), dpi=500)
    # input = input.numpy()
    # labels = labels.numpy()
    tsne = TSNE(n_components=2, init='pca', random_state=0, n_iter=5000, perplexity=30)
    result = tsne.fit_transform(input)
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    label = labels
    for i in range(result.shape[0]):
        if label[i] == 0:
            type1_x.append(result[i][0])
            type1_y.append(result[i][1])
        if label[i] == 1:
            type2_x.append(result[i][0])
            type2_y.append(result[i][1])

    #plt.title("TSNE X")
    #plt.xlim(xmin=-100, xmax=100)
    #plt.ylim(ymin=-100, ymax=150)
    type1 = plt.scatter(type1_x, type1_y, s=10, c='r',marker='<')
    type2 = plt.scatter(type2_x, type2_y, s=10, c='g',marker='<')
    plt.legend((type1, type2), ('Fake', 'Real'))
    plt.show()



class CAM(nn.Module):
    def __init__(self):
        super(CAM, self).__init__()

        #self.encoder1 = nn.Linear(32, 32)
        #self.encoder2 = nn.Linear(32, 32)
        '''
        self.affine_a = nn.Linear(1, 1, bias=False)
        self.affine_v = nn.Linear(1, 1, bias=False)

        self.W_a = nn.Linear(1, 1, bias=False)
        self.W_v = nn.Linear(1, 1, bias=False)
        '''
        self.W_ca = nn.Linear(64, 1, bias=False)
        self.W_cv = nn.Linear(64, 1, bias=False)

        #self.W_ha = nn.Linear(32, 1, bias=False)
        #self.W_hv = nn.Linear(32, 1, bias=False)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        #self.regressor = nn.Sequential(nn.Linear(640, 128), nn.Dropout(0.6), nn.Linear(128, 1))

    def forward(self, f1_norm, f2_norm):

        fin_audio_features = []
        fin_visual_features = []
        sequence_outs = []

        for i in range(f1_norm.shape[0]):
            audfts = f1_norm[i, :]
            visfts = f2_norm[i, :]

            aud_fts = audfts
            vis_fts = visfts
            aud_fts = aud_fts.reshape(1, 32)
            vis_fts = vis_fts.reshape(1, 32)

            aud_vis_fts = torch.cat((aud_fts, vis_fts), 1)
            #a_t = self.affine_a(aud_vis_fts.transpose(0, 1))   #wj
            a_t = aud_vis_fts.transpose(0, 1)
            att_aud = torch.mm(aud_fts.transpose(0, 1), a_t.transpose(0, 1))# xwj
            audio_att = self.tanh(torch.div(att_aud, math.sqrt(aud_vis_fts.shape[1]))) #c

            aud_vis_fts = torch.cat((aud_fts, vis_fts), 1)
           # v_t = self.affine_v(aud_vis_fts.transpose(0, 1))
            v_t = aud_vis_fts.transpose(0, 1)
            att_vis = torch.mm(vis_fts.transpose(0, 1), v_t.transpose(0, 1))
            vis_att = self.tanh(torch.div(att_vis, math.sqrt(aud_vis_fts.shape[1])))

           # H_a = self.relu(self.W_ca(audio_att) + self.W_a(aud_fts.transpose(0, 1)))
            H_a = 0.1 * self.relu(self.W_ca(audio_att) + aud_fts.transpose(0, 1))
            #H_v = self.relu(self.W_cv(vis_att) + self.W_v(vis_fts.transpose(0, 1)))
            H_v = 0.1 * self.relu(self.W_cv(vis_att) + vis_fts.transpose(0, 1))

            H_a = H_a.transpose(0, 1)
            H_v = H_v.transpose(0, 1)
            aud_vis_fts_1 = torch.cat((H_a, H_v), 1)
            sequence_outs.append(aud_vis_fts_1)
        final_outs = torch.stack(sequence_outs)
        return final_outs


#attention
class Attention_ast(nn.Module):
    def __init__(self):
        super(Attention_ast, self).__init__()
        self.weight = torch.rand(100, 1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x2 = x
        x2 = self.fc(x2)
        x2 = torch.sigmoid(x2)
        return x2

def parse_arguments(parser):
    parser.add_argument('training_file', type=str, metavar='<training_file>', help='')
    parser.add_argument('testing_file', type=str, metavar='<testing_file>', help='')
    parser.add_argument('output_file', type=str, metavar='<output_file>', help='')

    parse.add_argument('--static', type=bool, default=True, help='')
    parser.add_argument('--sequence_length', type=int, default=28, help='')
    parser.add_argument('--class_num', type=int, default=2, help='')
    parser.add_argument('--hidden_dim', type=int, default=128, help='')
    parser.add_argument('--embed_dim', type=int, default=32, help='')
    parser.add_argument('--vocab_size', type=int, default=300, help='')
    parser.add_argument('--dropout', type=int, default=0.5, help='')
    parser.add_argument('--filter_num', type=int, default=5, help='')
    parser.add_argument('--lambd', type=int, default=1, help='')
    parser.add_argument('--text_only', type=bool, default=False, help='')

    parser.add_argument('--d_iter', type=int, default=3, help='')
    parser.add_argument('--batch_size', type=int, default=128, help='')

    parser.add_argument('--num_epochs', type=int, default=1, help='')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--event_num', type=int, default=10, help='')

    parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
    parser.add_argument('--alpha', default=1, type=float, help='alpha in OGM-GE')
    parser.add_argument('--modulation', default='OGM', type=str, choices=['Normal', 'OGM', 'OGM_GE'])
    parser.add_argument('--modulation_starts', default=3, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=20, type=int, help='where modulation ends')
    parser.add_argument('--lr_decay_step', default=5, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')
    return parser

#当前最高： modulation_starts：3 epoch   modulation_ends：80 epoch  lr epoch5 0.1  alpha 1 modulation：OGM

class Rumor_Data(Dataset):
    def __init__(self, dataset):
        self.text = torch.from_numpy(np.array(dataset['post_text']))
        self.image = list(dataset['image'])
        # self.social_context = torch.from_numpy(np.array(dataset['social_feature']))
        self.mask = torch.from_numpy(np.array(dataset['mask']))
        self.label = torch.from_numpy(np.array(dataset['label']))
        self.event_label = torch.from_numpy(np.array(dataset['event_label']))
        self.text_clip = dataset['post_texts_clip']
        print('TEXT: %d, Image: %d, label: %d, Event: %d, TEXT_clip: %d'
              % (len(self.text), len(self.image), len(self.label), len(self.event_label), len(self.text_clip)))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (self.text[idx], self.image[idx], self.mask[idx], self.text_clip[idx]), self.label[idx], \
               self.event_label[idx]


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg()
        return grad_output


class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)


# Neural Network Model (1 hidden layer)
class CNN_Fusion(nn.Module):
    def __init__(self, args):
        super(CNN_Fusion, self).__init__()
        self.args = args

        self.event_num = args.event_num

        vocab_size = args.vocab_size
        emb_dim = args.embed_dim

        C = args.class_num
        self.hidden_size = args.hidden_dim
        self.lstm_size = args.embed_dim
        self.social_size = 19

        # bert
        XLNet_model = BertModel.from_pretrained('bert-base-chinese')
        self.bert_hidden_size = args.bert_hidden_dim
        self.fc2 = nn.Linear(self.bert_hidden_size, self.hidden_size)

        for param in XLNet_model.parameters():
            param.requires_grad = False
        self.bertModel = XLNet_model

        self.dropout = nn.Dropout(args.dropout)

        # IMAGE
        vgg_19 = torchvision.models.vgg19(pretrained=True)
        for param in vgg_19.parameters():
            param.requires_grad = False
        # visual model
        num_ftrs = vgg_19.classifier._modules['6'].out_features
        self.vgg = vgg_19
        self.image_fc1 = nn.Linear(num_ftrs, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.lstm_size) #128-32
        self.fc4 = nn.Linear(self.hidden_size, self.lstm_size)

        # Class Classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(2 * self.lstm_size, 2))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))

        # clip降维网络
        self.fc5 = nn.Linear(512, 128)
        self.fc6 = nn.Linear(128, 32)

        self.fc7 = nn.Linear(512, 128)
        self.fc8 = nn.Linear(128, 32)

        # SwinTransformer
        self.model_swintransformer = timm.create_model("swin_base_patch4_window7_224.ms_in22k_ft_in1k", pretrained=True)
        self.model_swintransformer = self.model_swintransformer.eval()
        data_config = timm.data.resolve_model_data_config(self.model_swintransformer)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)
        # clip提取网络
        self.model_clip, self.preprocess_clip = clip.load("ViT-B/32", device='cuda')

        # cross-modal transformer
        self.co_attention_it = coattention.co_attention(d_k=32, d_v=32, n_heads=4, dropout=0, d_model=32,
                                                        visual_len=32, sen_len=32, fea_v=32, fea_s=32, pos=False)

        self.co_attention_it_clip = coattention.co_attention(d_k=32, d_v=32, n_heads=4, dropout=0, d_model=32,
                                                             visual_len=32, sen_len=32, fea_v=32, fea_s=32, pos=False)

        #self.trm = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True)
        #attention 注意力机制
        self.attention1 = Attention_ast()
        self.attention2 = Attention_ast()

        #attention
        self.attention_senior = CAM()

    def init_hidden(self, batch_size):
        return (to_var(torch.zeros(1, batch_size, self.lstm_size)),
                to_var(torch.zeros(1, batch_size, self.lstm_size)))

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (sample number,hidden_dim, length)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        return x

    def forward(self, text, image, mask, text1):
        # IMAGE
        image2 = image
        image3 = image

        # swintransformer
        image_swin_all = []
        image_list = list(image3.chunk(16))
        for i in range(len(image_list)):
            tensor = image_list[i].squeeze(0)
            tensor = np.array(tensor.cpu())
            # to_pil = transforms.ToPILImage()
            tensor = tensor.transpose((1, 2, 0))
            image4 = Image.fromarray(np.uint8(tensor))
            a = self.transforms(image4).unsqueeze(0)
            a = a.cuda()
            output = self.model_swintransformer(a)
            image_swin_all.append(output)

        image_swin_all_final = torch.stack(image_swin_all)
        image_swin_all_final = torch.squeeze(image_swin_all_final)

        image_swin_all_final = image_swin_all_final.cuda()

        image1 = image_swin_all_final

        last_hidden_state = torch.mean(self.bertModel(text.long())[0], dim=1, keepdim=False)

        image = F.tanh(self.image_fc1(image1))
        image = F.tanh(self.fc4(image))

        text = F.tanh(self.fc2(last_hidden_state))
        text = F.tanh(self.fc3(text))

#cross-transformer
        fea_c1, fea_c2 = self.co_attention_it(v=image, s=text, v_len=image.shape[1], s_len=text.shape[1])

        fea_c1 = torch.mean(fea_c1, -2)

        fea_c2 = torch.mean(fea_c2, -2)


        with torch.no_grad():
            text1 = torch.squeeze(text1)
            text_clip = self.model_clip.encode_text(text1)
            image_clip = self.model_clip.encode_image(image2)

        text_clip = F.tanh(self.fc5(text_clip.float()))
        text_clip = F.tanh(self.fc6(text_clip))

        image_clip = F.tanh(self.fc7(image_clip.float()))
        image_clip = F.tanh(self.fc8(image_clip))

        fea_c3, fea_c4 = self.co_attention_it_clip(v=image_clip, s=text_clip, v_len=image_clip.shape[1],
                                                   s_len=text_clip.shape[1])

        fea_c3 = torch.mean(fea_c3, -2)

        fea_c4 = torch.mean(fea_c4, -2)

        att_A1 = self.attention1(fea_c1)
        att_A2 = self.attention1(fea_c3)

        att_A3 = self.attention2(fea_c2)
        att_A4 = self.attention2(fea_c4)

        att_A = torch.cat((att_A1, att_A2), 1)
        att_A = torch.softmax(att_A, dim=1)

        att_A1 = att_A[:, :1]
        att_A2 = att_A[:, 1:2]

        att_B = torch.cat((att_A3, att_A4), 1)
        att_B = torch.softmax(att_B, dim=1)

        att_A3 = att_B[:, :1]
        att_A4 = att_B[:, 1:2]

        F_A = torch.multiply(att_A1, fea_c1) + torch.multiply(att_A2, fea_c3)# + torch.multiply(att_A3, fea_c3) + torch.multiply(att_A4, fea_c4)
        F_B = torch.multiply(att_A3, fea_c2) + torch.multiply(att_A4, fea_c4)

        text_image_clip = self.attention_senior(F_A, F_B)
        text_image_clip = torch.squeeze(text_image_clip)


        class_output = self.class_classifier(text_image_clip)

        return class_output, fea_c1, fea_c2, last_hidden_state, image1, fea_c3, fea_c4, text, image, text_clip, image_clip

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_np(x):
    return x.data.cpu().numpy()


def select(train, selec_indices):
    temp = []
    for i in range(len(train)):
        print("length is " + str(len(train[i])))
        print(i)
        # print(train[i])
        ele = list(train[i])
        temp.append([ele[i] for i in selec_indices])
    return temp


def make_weights_for_balanced_classes(event, nclasses=15):
    count = [0] * nclasses
    for item in event:
        count[item] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(event)
    for idx, val in enumerate(event):
        weight[idx] = weight_per_class[val]
    return weight


def split_train_validation(train, percent):
    whole_len = len(train[0])

    train_indices = (sample(range(whole_len), int(whole_len * percent)))
    train_data = select(train, train_indices)
    print("train data size is " + str(len(train[3])))

    validation = select(train, np.delete(range(len(train[0])), train_indices))
    print("validation size is " + str(len(validation[3])))
    print("train and validation data set has been splited")

    return train_data, validation


def main(args):
    print('loading data')
    train, validation, test = load_data(args)
    test_id = test['post_id']

    train_dataset = Rumor_Data(train)

    validate_dataset = Rumor_Data(validation)

    test_dataset = Rumor_Data(test)

    # Data Loader (Input Pipeline)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)

    train_event = train_loader.dataset.event_label

    validate_loader = DataLoader(dataset=validate_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=True)
    validate_event = validate_loader.dataset.event_label
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)
    test_event = test_loader.dataset.event_label

    train_event_num = torch.max(train_event) + 1
    validate_event_num = torch.max(validate_event) + 1
    test_event_num = torch.max(test_event) + 1

    print(train_event_num, validate_event_num, test_event_num)
    print('building model')
    model = CNN_Fusion(args)

    if torch.cuda.is_available():
        print("CUDA")
        model.cuda()

    # Loss and Optimizer
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()
    criterion = nn.CrossEntropyLoss()

    _loss_1 = 0
    _loss_2 = 0

    _loss_3 = 0
    _loss_4 = 0

    params = []
    for param in model.parameters():
        if param.requires_grad:
            params.append(param)

    optimizer = torch.optim.Adam(params, lr=args.learning_rate, eps=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    print("loader size " + str(len(train_loader)))
    best_validate_acc = 0.000
    best_validate_dir = ''
    print('training model')
    adversarial = True
    # Train the Model
    for epoch in range(args.num_epochs):

        p = float(epoch) / 100
        # lambd = 2. / (1. + np.exp(-10. * p)) - 1
        lr = 0.001 / (1. + 10 * p) ** 0.75

        optimizer.lr = lr
        # rgs.lambd = lambd
        cost_vector = []
        class_cost_vector = []
        acc_vector = []
        valid_acc_vector = []
        vali_cost_vector = []



        for i, (train_data, train_labels, event_labels) in enumerate(train_loader):
            train_text, train_image, train_mask, train_labels, event_labels, clip_text = \
                to_var(train_data[0]), to_var(train_data[1]), to_var(train_data[2]), \
                to_var(train_labels), to_var(event_labels), to_var(train_data[3])

            optimizer.zero_grad()

            class_outputs, output1, output2, last_hidden_state, image1, text_clip_output, image_clip_output, out_1, out_2, out_3, out_4 = model(
                train_text, train_image, train_mask, clip_text)


            weight_size_1 = model.fc3.weight.size(1)
            weight_size_2 = model.fc4.weight.size(1)
            weight_size_3 = model.fc6.weight.size(1)
            weight_size_4 = model.fc8.weight.size(1)  # 128

            out_text_1 = (torch.mm(out_1, model.fc3.weight[:, weight_size_1 // 2:96])
                          + model.fc3.bias / 2)

            out_image_1 = (torch.mm(out_2, model.fc4.weight[:, 32:weight_size_2 // 2])
                           + model.fc4.bias / 2)

            out_text_clip_1 = (torch.mm(out_3, model.fc6.weight[:, weight_size_3 // 2:96])
                               + model.fc6.bias / 2)

            out_image_clip_1 = (torch.mm(out_4, model.fc8.weight[:, 32:weight_size_4 // 2])
                                + model.fc8.bias / 2)

            class_loss = criterion(class_outputs, train_labels.long())


            if (epoch == 0):
                if (i == 0):
                    label_pre = train_labels#[mask]
                else:
                    a = train_labels#[mask]
                    label_pre = torch.cat((label_pre, a), 0)

            if (epoch == 9):
                if (i == 0):
                    label_last = train_labels#[mask]
                else:
                    a = train_labels#[mask]
                    label_last = torch.cat((label_last, a), 0)

            if (epoch == 0):
                if (i == 0):
                    s3 = torch.cat((output1, output2), 1)
                else:
                    c = torch.cat((output1, output2), 1)
                    s3 = torch.cat((s3, c), 0)
            if (epoch == 9):
                if (i == 0):
                    s4 = torch.cat((output1, output2), 1)
                else:
                    d = torch.cat((output1, output2), 1)
                    s4 = torch.cat((s4, d), 0)


            feture1 = torch.unsqueeze(output1, 1)
            feture2 = torch.unsqueeze(output2, 1)

            feture3 = torch.unsqueeze(text_clip_output, 1)
            feture4 = torch.unsqueeze(image_clip_output, 1)

            features = torch.cat((feture1, feture2, feture3, feture4), 1)
            Conloss = SupConLoss(temperature=0.5)
            con_loss = Conloss(features, train_labels)
            loss = class_loss + 0.1 * con_loss

            # 返回损失

            loss_1 = criterion(out_text_1, train_labels.long())
            loss_2 = criterion(out_image_1, train_labels.long())

            loss_3 = criterion(out_text_clip_1, train_labels.long())
            loss_4 = criterion(out_image_clip_1, train_labels.long())

            loss.backward()

            # 计算logit得分

            score_1 = sum([softmax(out_text_1)[i][train_labels[i]] for i in range(out_text_1.size(0))])
            score_2 = sum([softmax(out_image_1)[i][train_labels[i]] for i in range(out_image_1.size(0))])
            score_3 = sum([softmax(out_text_clip_1)[i][train_labels[i]] for i in range(out_text_clip_1.size(0))])
            score_4 = sum([softmax(out_image_clip_1)[i][train_labels[i]] for i in range(out_image_clip_1.size(0))])

            # 计算比率

            ratio_1 = score_1 / (score_1 + score_2 + score_3 + score_4)
            ratio_2 = score_2 / (score_1 + score_2 + score_3 + score_4)
            ratio_3 = score_3 / (score_1 + score_2 + score_3 + score_4)
            ratio_4 = score_4 / (score_1 + score_2 + score_3 + score_4)


            if max(score_1, score_2, score_3, score_4) == score_1:
                coeff_1 = 1 - tanh(args.alpha * relu(ratio_1))
                coeff_2 = 1
                coeff_3 = 1
                coeff_4 = 1

            if max(score_1, score_2, score_3, score_4) == score_2:
                coeff_1 = 1
                coeff_2 = 1 - tanh(args.alpha * relu(ratio_2))
                coeff_3 = 1
                coeff_4 = 1

            if max(score_1, score_2, score_3, score_4) == score_3:
                coeff_1 = 1
                coeff_2 = 1
                coeff_3 = 1 - tanh(args.alpha * relu(ratio_3))
                coeff_4 = 1

            if max(score_1, score_2, score_3, score_4) == score_4:
                coeff_1 = 1
                coeff_2 = 1
                coeff_3 = 1
                coeff_4 = 1 - tanh(args.alpha * relu(ratio_4))

            if args.modulation_starts <= epoch <= args.modulation_ends:
                for name, parms in model.named_parameters():
                    layer = str(name).split('.')[0]
                    if 'fc3' in layer and len(parms.grad.size()) == 4:
                        if args.modulation == 'OGM_GE':  # bug fixed
                            parms.grad = parms.grad * coeff_1 + \
                                         torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                        elif args.modulation == 'OGM':
                            parms.grad *= coeff_1

                    if 'fc4' in layer and len(parms.grad.size()) == 4:
                        if args.modulation == 'OGM_GE':  # bug fixed
                            parms.grad = parms.grad * coeff_2 + \
                                         torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                        elif args.modulation == 'OGM':
                            parms.grad *= coeff_2

                    if 'fc6' in layer and len(parms.grad.size()) == 4:
                        if args.modulation == 'OGM_GE':  # bug fixed
                            parms.grad = parms.grad * coeff_3 + \
                                         torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                        elif args.modulation == 'OGM':
                            parms.grad *= coeff_3

                    if 'fc8' in layer and len(parms.grad.size()) == 4:
                        if args.modulation == 'OGM_GE':  # bug fixed
                            parms.grad = parms.grad * coeff_4 + \
                                         torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                        elif args.modulation == 'OGM':
                            parms.grad *= coeff_4

            else:
                pass

            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()
            _, argmax = torch.max(class_outputs, 1)

            cross_entropy = True

            if True:
                accuracy = (train_labels == argmax.squeeze()).float().mean()
            else:
                _, labels = torch.max(train_labels, 1)
                accuracy = (labels.squeeze() == argmax.squeeze()).float().mean()

            cost_vector.append(loss.item())
            class_cost_vector.append(class_loss.item())
            acc_vector.append(accuracy.item())
            _loss_1 += loss_1.item()
            _loss_2 += loss_2.item()
            _loss_3 += loss_3.item()
            _loss_4 += loss_4.item()

        scheduler.step()

        model.eval()

        validate_acc_vector_temp = []
        for i, (validate_data, validate_labels, event_labels) in enumerate(test_loader):
            validate_text, validate_image, validate_mask, validate_labels, event_labels, clip_text = \
                to_var(validate_data[0]), to_var(validate_data[1]), to_var(validate_data[2]), \
                to_var(validate_labels), to_var(event_labels), to_var(validate_data[3])
            validate_text = validate_text.long()
            validate_outputs, output1, output2, last_hidden_state, image1, text_clip_output, image_clip_output, out_1, out_2, out_3, out_4 = model(
               validate_text, validate_image, validate_mask, clip_text)
            _, validate_argmax = torch.max(validate_outputs, 1)

            vali_loss = criterion(validate_outputs, validate_labels.long())
            validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
            vali_cost_vector.append(vali_loss.item())
            validate_acc_vector_temp.append(validate_accuracy.item())
        validate_acc = np.mean(validate_acc_vector_temp)
        valid_acc_vector.append(validate_acc)

        #model.train()

        print('Epoch [%d/%d],  Loss: %.4f, Class Loss: %.4f, Train_Acc: %.4f,Validate_Acc: %.4f.'
              % (
                  epoch + 1, args.num_epochs, np.mean(cost_vector), np.mean(class_cost_vector),
                  np.mean(acc_vector), validate_acc))

        if validate_acc > best_validate_acc:
            best_validate_acc = validate_acc
            if not os.path.exists(args.output_file):
                os.mkdir(args.output_file)

            best_validate_dir = args.output_file + str(epoch + 1) + '.pkl'
            torch.save(model.state_dict(), best_validate_dir)
        if epoch == 99 or epoch == 98 or epoch == 97:
            torch.save(model.state_dict(), args.output_file + str(epoch + 1) + '.pkl')

    # Test the Model
    print('testing model')
    model = CNN_Fusion(args)
    model.load_state_dict(torch.load(best_validate_dir))
    #    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    test_score = []
    test_pred = []
    test_true = []
    for i, (test_data, test_labels, event_labels) in enumerate(test_loader):
        test_text, test_image, test_mask, test_labels, clip_text = to_var(
            test_data[0]), to_var(test_data[1]), to_var(test_data[2]), to_var(test_labels), to_var(test_data[3])
        test_text = test_text.long()
        test_outputs, output1, output2, last_hidden_state, image1, text_clip_output, image_clip_output, out_1, out_2, out_3, out_4 = model(
            test_text, test_image, test_mask, clip_text)

        _, test_argmax = torch.max(test_outputs, 1)

        if i == 0:
            test_score = to_np(test_outputs.squeeze())
            test_pred = to_np(test_argmax.squeeze())
            test_true = to_np(test_labels.squeeze())
        else:
            test_score = np.concatenate((test_score, to_np(test_outputs)), axis=0)
            test_pred = np.concatenate((test_pred, to_np(test_argmax)), axis=0)
            test_true = np.concatenate((test_true, to_np(test_labels)), axis=0)

    test_accuracy = metrics.accuracy_score(test_true, test_pred)
    test_f1 = metrics.f1_score(test_true, test_pred, average='macro')
    test_precision = metrics.precision_score(test_true, test_pred, average='macro')
    test_recall = metrics.recall_score(test_true, test_pred, average='macro')
    test_score_convert = [x[1] for x in test_score]
    test_aucroc = metrics.roc_auc_score(test_true, test_score_convert, average='macro')

    test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)

    print("Classification Acc: %.4f, AUC-ROC: %.4f"
          % (test_accuracy, test_aucroc))
    print("Classification report:\n%s\n"
          % (metrics.classification_report(test_true, test_pred)))
    print("Classification confusion matrix:\n%s\n"
          % (test_confusion_matrix))

    tsne(s3.cpu().data, label_pre.cpu().data)  # 训练前文本和图像
    tsne(s4.cpu().data, label_last.cpu().data)  # 训练后文本和图像


def get_top_post(output, label, test_id, top_n=500):
    filter_output = []
    filter_id = []
    for i, l in enumerate(label):
        # print(np.argmax(output[i]))
        if np.argmax(output[i]) == l and int(l) == 1:
            filter_output.append(output[i][1])
            filter_id.append(test_id[i])

    filter_output = np.array(filter_output)

    top_n_indice = filter_output.argsort()[-top_n:][::-1]

    top_n_id = np.array(filter_id)[top_n_indice]
    top_n_id_dict = {}
    for i in top_n_id:
        top_n_id_dict[i] = True

    pickle.dump(top_n_id_dict, open("../Data/weibo/top_n_id.pickle", "wb"))

    return top_n_id


def word2vec(post, word_id_map, W):
    word_embedding = []
    mask = []

    for sentence in post:
        sen_embedding = []
        seq_len = len(sentence) - 1
        mask_seq = np.zeros(args.sequence_len, dtype=np.float32)
        mask_seq[:len(sentence)] = 1.0
        for i, word in enumerate(sentence):
            sen_embedding.append(word_id_map[word])

        while len(sen_embedding) < args.sequence_len:
            sen_embedding.append(0)

        word_embedding.append(copy.deepcopy(sen_embedding))
        mask.append(copy.deepcopy(mask_seq))
        # length.append(seq_len)
    return word_embedding, mask


def re_tokenize_sentence(flag):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    tokenized_texts = []
    tokenized_texts_clip = []
    original_texts = flag['original_post']
    for sentence in original_texts:
        tokenized_text = tokenizer.encode(sentence)
        tokenized_texts.append(tokenized_text)

        tokenized_text = clip.tokenize(sentence, truncate=True)
        tokenized_texts_clip.append(tokenized_text)
    flag['post_text'] = tokenized_texts
    flag['post_texts_clip'] = tokenized_texts_clip


def get_all_text(train, validate, test):
    all_text = list(train['post_text']) + list(validate['post_text']) + list(test['post_text'])
    return all_text


def align_data(flag, args):
    text = []
    mask = []
    for sentence in flag['post_text']:
        sen_embedding = []
        mask_seq = np.zeros(args.sequence_len, dtype=np.float32)
        mask_seq[:len(sentence)] = 1.0
        for i, word in enumerate(sentence):
            sen_embedding.append(word)

        while len(sen_embedding) < args.sequence_len:
            sen_embedding.append(0)

        text.append(copy.deepcopy(sen_embedding))
        mask.append(copy.deepcopy(mask_seq))
    flag['post_text'] = text
    flag['mask'] = mask


def load_data(args):
    train, validate, test = process_data.get_data(args.text_only)
    re_tokenize_sentence(train)
    re_tokenize_sentence(validate)
    re_tokenize_sentence(test)
    all_text = get_all_text(train, validate, test)
    max_len = len(max(all_text, key=len))
    args.sequence_len = max_len
    align_data(train, args)
    align_data(validate, args)
    align_data(test, args)
    return train, validate, test


def transform(event):
    matrix = np.zeros([len(event), max(event) + 1])
    # print("Translate  shape is " + str(matrix))
    for i, l in enumerate(event):
        matrix[i, l] = 1.00
    return matrix


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    train = ''
    test = ''
    output = r'/mnt/sdb/user01/chenshu/MCAN_code_4-main/weibo/output_image_text'
    args = parser.parse_args([train, test, output])

    main(args)
