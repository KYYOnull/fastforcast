


import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from util.data_factory import data_provider
from util.tools import adjust_learning_rate
# params
from layers import FNet
import matplotlib.pyplot as plt

# 搭建模型Transoformer模型
class SCINetinitialization():

    def __init__(self, args):
        super(SCINetinitialization, self).__init__()
        self.args = args
        self.model, self.device = self.build_model(args)


    def build_model(self, args):
        model = FNet.Model(self.args).float() # FNet
        if args.use_gpu:
            device = torch.device('cuda:{}'.format(self.args.device))
            print('Use GPU: cuda:{}'.format(args.device))
        else:
            print('Use CPU')
            device = torch.device('cpu')

        total = sum([param.nelement() for param in model.parameters()])
        print('Number of parameters: %.2fM' % (total / 1e6))  # 模型参数量

        if self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=[device])

        return model, device


    def _get_data(self, flag, pre_data=None):
        data_set, data_loader = data_provider(self.args, flag, pre_data)
        return data_set, data_loader


    def train(self, setting):
        _, train_loader = self._get_data(flag='train')

        path = os.path.join(self.args.checkpoints, setting) # ./models/  predict-FNet-data-ETTh1
        if not os.path.exists(path): os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate) # 0.001
        criterion = nn.MSELoss()

        for epoch in range(self.args.train_epochs): # 10
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(epoch + 1, train_steps, train_loss))
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        path = os.path.join(self.args.checkpoints, setting) # ./models/  predict-FNet-data-ETTh1
        best_model_path = path + '/' + 'model.pth'
        torch.save(self.model.state_dict(), best_model_path) # to file

        return self.model


    def predict(self, setting, load=False):
        results = []
        preds = []
        path = os.path.join(self.args.checkpoints, setting) # ./models/  predict-FNet-data-ETTh1
        best_model_path = path + '/' + 'model.pth'
        self.model.load_state_dict(torch.load(best_model_path)) # from file
        self.model.eval()

        if args.rollingforecast:
            pre_data = pd.read_csv(args.root_path + args.rolling_data_path) # ./data/ ETTh1-Test.csv
        else:
            pre_data = None

        for i in (0 if pre_data is None else range(int(len(pre_data)/args.pred_len) - 1)):
            # pre_data为None 只会执行一次
            # 
            if i == 0:
                data_set, pred_loader = self._get_data(flag='pred')
            else:  
                # 滚动预测
                data_set, pred_loader = self._get_data(flag='pred', pre_data=pre_data.iloc[: i * args.pred_len])
                print(f'滚动预测第{i + 1} 次') # 123*4= 

            with torch.no_grad():
                for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                        batch_y.device)
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    outputs = self.model(batch_x)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    outputs = data_set.inverse_transform(outputs)
                    if self.args.features == 'MS':
                        for i in range(args.pred_len):
                            preds.append(outputs[0][i][outputs.shape[2] - 1])  # 取最后一个预测值即对应target列
                    else:
                        for i in range(args.pred_len):
                            preds.append(outputs[0][i][outputs.shape[2] - 1])
                    print(outputs)

        # 保存结果
        if args.rollingforecast:
            df = pd.DataFrame({'real': pre_data['{}'.format(args.target)][:len(preds)], 'forecast': preds})
            df.to_csv('./results/{}-ForecastResults.csv'.format(args.target), index=False)
        else:
            df = pd.DataFrame({'forecast': results})
            df.to_csv('./results/{}-ForecastResults.csv'.format(args.target), index=False)

        if args.show_results:
            plt.style.use('ggplot')

            # 创建折线图
            plt.plot(df['real'].tolist(), label='real', color='blue')  # 实际值
            plt.plot(df['forecast'].tolist(), label='forecast', color='red', linestyle='--')  # 预测值

            plt.grid(True)
            plt.title('real vs forecast')
            plt.xlabel('time')
            plt.ylabel('value')
            plt.legend()

            plt.savefig('results.png')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FNet Multivariate Time Series Forecasting')
    # basic config
    parser.add_argument('--train', type=bool, default=True, help='Whether to conduct training')

    parser.add_argument('--rollingforecast', type=bool, default=True) # 必须提供一个滚动预测的数据集
    parser.add_argument('--rolling_data_path', type=str, default='ETTh1-Test.csv', 
                        help='rolling data file')
    
    parser.add_argument('--show_results', type=bool, default=True, help='Whether show forecast and real results graph')
    parser.add_argument('--model', type=str, default='FNet',help='Model name')

    # data loader
    parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./models/', help='location of model models')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=126, 
                        help='input sequence length')
    parser.add_argument('--label_len', type=int, default=64, help='start token length')
    parser.add_argument('--pred_len', type=int, default=4, 
                        help='prediction sequence length') # 每次预测四条数据

    # model
    parser.add_argument('--norm', action='store_false', default=True, help='whether to apply LayerNorm')
    parser.add_argument('--rev', action='store_true', default=True, help='whether to apply RevIN')
    parser.add_argument('--d_model', type=int, default=512, 
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=1, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, 
                        help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--enc_in', type=int, default=7, 
                        help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
    parser.add_argument('--device', type=int, default=0, help='gpu')

    args = parser.parse_args()
    Exp = SCINetinitialization
    # setting record of experiments
    setting = 'predict-{}-data-{}'.format(args.model, args.data_path[:-4]) # ETTh1.csv
    # models/predict-FNet-data-ETTh1

    SCI = SCINetinitialization(args)  # 实例化模型
    if args.train:
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.model))
        SCI.train(setting)

    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.model))
    SCI.predict(setting, True)

    plt.show()