from models.race_lap_ngrams import RaceLapNgrams
from torch.utils.data import DataLoader, TensorDataset
import torch
from models.arch import MLP_regression_v1, MLP_MC_v1, RNN_regression_v1, RNN_MC_v1, LSTM_regression_v1, LSTM_MC_v1
from datetime import datetime
import json

def json_write(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f)

def eval_metrics(lap_model: torch.nn.Module, pos_model:torch.nn.Module, dataloader):
    lap_model.eval()
    pos_model.eval()

    lap_mae, lap_mse, pos_correct, total = 0, 0, 0, 0

    with torch.no_grad():

        for X, lap_t, pos_t in dataloader:

            lap_z = lap_model(X)
            pos_z = pos_model(X)

            lap_mae += torch.abs(lap_z.squeeze() - lap_t).sum().item()
            lap_mse += ((lap_z.squeeze() - lap_t) ** 2).sum().item()

            _, pos_pred = torch.max(pos_z, 1)
            _, pos_t = torch.max(pos_t, 1)

            pos_correct += (pos_pred == pos_t).sum().item()
            total += X.size(0)
    
    return lap_mae / total, lap_mse / total, pos_correct / total

def train_inner(lap_model, pos_model, train_loader, val_loader, 
                    num_epochs, initial_lr, gamma_scheduler=0.7, step_size_scheduler=20, 
                    verbose_every=200, save_every=10):

    lap_crit = torch.nn.MSELoss()
    pos_crit = torch.nn.CrossEntropyLoss()

    lap_optim = torch.optim.Adam(lap_model.parameters(), lr=initial_lr)
    pos_optim = torch.optim.Adam(pos_model.parameters(), lr=initial_lr)

    lap_scheduler = torch.optim.lr_scheduler.StepLR(lap_optim, step_size=step_size_scheduler, gamma=gamma_scheduler)
    pos_scheduler = torch.optim.lr_scheduler.StepLR(pos_optim, step_size=step_size_scheduler, gamma=gamma_scheduler)

    lap_train_loss_list, lap_train_mae_list, lap_train_mse_list, lap_val_mae_list, lap_val_mse_list = [], [], [], [], []
    pos_train_loss_list, pos_train_acc_list, pos_val_acc_list = [], [], []
    iter_list, iter_time_list, epoch_time_list = [], [], []

    iter_count = 0
    train_start_time = datetime.now()
    prev_epoch_time = train_start_time
    prev_iter_time = train_start_time

    print(f'\t {"="*100}\n')

    print(f'LAPTIME MODEL: {lap_model} \n POSITION MODEL: {pos_model} \n'
          f'num epochs: {num_epochs} \t initial lr: {initial_lr} \t gamma: {gamma_scheduler} \t step size: {step_size_scheduler} \n'
          f'laptime criterion: {lap_crit} \t position criterion: {pos_crit} laptime optimiser: {lap_optim} \t position optimiser: {pos_optim} \n')

    for epoch in range(num_epochs):

        for X, lap_t, pos_t in train_loader:
            lap_model.train()
            pos_model.train()

            lap_z = lap_model(X)
            pos_z = pos_model(X)

            lap_loss = lap_crit(lap_z.squeeze(), lap_t)
            pos_loss = pos_crit(pos_z.squeeze(), pos_t)

            lap_optim.zero_grad()
            lap_loss.backward()
            lap_optim.step()
            pos_optim.zero_grad()
            pos_loss.backward()
            pos_optim.step()

            if (iter_count%verbose_every == 0) or (iter_count%save_every == 0):

                lap_train_loss_list.append(lap_loss.item())
                pos_train_loss_list.append(pos_loss.item())

                train_lap_mae, train_lap_mse, train_pos_acc = eval_metrics(lap_model, pos_model, train_loader)
                val_lap_mae, val_lap_mse, val_pos_acc = eval_metrics(lap_model, pos_model, val_loader)

                lap_train_mae_list.append(train_lap_mae)
                lap_train_mse_list.append(train_lap_mse)
                pos_train_acc_list.append(train_pos_acc)
                lap_val_mae_list.append(val_lap_mae)
                lap_val_mse_list.append(val_lap_mse)
                pos_val_acc_list.append(val_pos_acc)
                iter_list.append(iter_count)

                curr_iter_time = datetime.now()
                this_iter_time = curr_iter_time - prev_iter_time
                iter_time_list.append({
                    'seconds': this_iter_time.seconds,
                    'microseconds': this_iter_time.microseconds
                })
                prev_iter_time = curr_iter_time

                if iter_count%verbose_every == 0:
                    print(f'Epoch:{epoch} / {num_epochs} \t Iter:{iter_count} \t lr:{lap_scheduler.get_last_lr()} \t time:{this_iter_time} \n'
                        f'LAPTIME TRAIN \t loss:{lap_loss.item()} mae:{train_lap_mae:.4f} mse:{train_lap_mse:.4f} \n'
                        f'LAPTIME VAL \t mae:{val_lap_mae:.4f} mse:{val_lap_mse:.4f} \n'
                        f'POSITION TRAIN \t loss:{pos_loss.item()} acc:{train_pos_acc:.4f} \t POSITION VAL \t acc:{val_pos_acc:.4f}\n')
            
            iter_count += 1
        
        lap_scheduler.step()
        pos_scheduler.step()
        
        curr_epoch_time = datetime.now()
        this_epoch_time = curr_epoch_time - prev_epoch_time
        epoch_time_list.append({
            'seconds': this_epoch_time.seconds,
            'microseconds': this_epoch_time.microseconds
        })
        prev_epoch_time = curr_epoch_time
    
    total_time = datetime.now() - train_start_time

    return {
        'total_time': {
            'seconds': total_time.seconds,
            'microseconds': total_time.microseconds
        },
        'model_info': {
            'laptime_model': str(lap_model),
            'position_model': str(pos_model),
        },
        'hyperparameters': {
            'laptime_criterion': str(lap_crit),
            'position_criterion': str(pos_crit),
            'laptime_optimiser': str(lap_optim),
            'position_optimiser': str(pos_optim),
            'laptime_scheduler': str(lap_scheduler),
            'position_scheduler': str(pos_scheduler),
        },
        'metrics': {
            'laptime_loss': lap_train_loss_list,
            'position_loss': pos_train_loss_list,
            'laptime_train_mae': lap_train_mae_list,
            'laptime_train_mse': lap_train_mse_list,
            'laptime_val_mae': lap_val_mae_list,
            'laptime_val_mse': lap_val_mse_list,
            'position_train_acc': pos_train_acc_list,
            'position_val_acc': pos_val_acc_list,
            'iter_count': iter_list,
            'iter_time': iter_time_list,
            'epoch_time': epoch_time_list,
        }
    }


def train_mlp(n=int, batchsize_list=list[int],
              lap_hidden_list = list[int], lap_act_fn = str, pos_hidden_list=list[int], pos_act_fn=str, 
              lr_list=list[float], num_epochs=int, device=torch.device,
              gamma_scheduler=0.7, step_size_scheduler=20,
              output_filename='metrics_1.json', verbose_every=500):
    
    metrics = []
    
    print('loading data...')
    
    race_lap_ngrams = RaceLapNgrams(n=n)
    race_lap_ngrams.split_by_year()

    train_X, train_lap, train_pos = race_lap_ngrams.get_train_tensors()
    val_X, val_lap, val_pos = race_lap_ngrams.get_val_tensors()
    # test_X, test_lap, test_pos = race_lap_ngrams.get_test_tensors()

    train_X = train_X.to(device)
    train_lap = train_lap.to(device)
    train_pos = train_pos.to(device)
    val_X = val_X.to(device)
    val_lap = val_lap.to(device)
    val_pos = val_pos.to(device)
    # test_X = test_X.to(device)
    # test_lap = test_lap.to(device)
    # test_pos = test_pos.to(device)

    print(f'X shape: {train_X.shape} lap shape: {train_lap.shape} pos shape: {train_pos.shape} \n train size: {len(train_X)} val size: {len(val_X)}')

    for batchsize in batchsize_list:

        train_dataloader = DataLoader(TensorDataset(train_X, train_lap, train_pos), batch_size=batchsize, shuffle=True)
        val_dataloader = DataLoader(TensorDataset(val_X, val_lap, val_pos), batch_size=batchsize)
        # test_dataloader = DataLoader(TensorDataset(test_X, test_lap, test_pos), batch_size=batchsize)

        first_train = next(iter(train_dataloader))
        print(f"data shape: X:{first_train[0].shape} lap:{first_train[1].shape} pos:{first_train[2].shape}")

        for lap_hidden, pos_hidden in zip(lap_hidden_list, pos_hidden_list):

            for lr in lr_list:

                laptime_model = MLP_regression_v1(hidden_size=lap_hidden, 
                                                data_dim=race_lap_ngrams.data_dim, 
                                                input_num=n-1, act_fn=lap_act_fn)
                position_model = MLP_MC_v1(hidden_size=pos_hidden, 
                                        data_dim=race_lap_ngrams.data_dim, input_num=n-1, 
                                        act_fn=pos_act_fn, output_classes=20)
                
                laptime_model = laptime_model.to(device)
                position_model = position_model.to(device)

                this_metrics = train_inner(laptime_model, position_model, train_dataloader, val_dataloader, num_epochs, initial_lr=lr, 
                                           gamma_scheduler=gamma_scheduler, step_size_scheduler=step_size_scheduler,
                                           verbose_every=verbose_every, save_every=400)
                this_metrics['device'] = str(device)
                this_metrics['hyperparameters']['init_lr'] = lr
                this_metrics['hyperparameters']['batchsize'] = batchsize
                this_metrics['hyperparameters']['num_epochs'] = num_epochs
                this_metrics['hyperparameters']['n'] = n
                this_metrics['data_info'] = {
                    'train_size': len(train_dataloader),
                    'val_size': len(val_dataloader),
                    'X shape': first_train[0].shape,
                    'laptime shape': first_train[1].shape,
                    'postion shape': first_train[2].shape,
                }

                metrics.append(this_metrics)
        
    
    print('saving metrics...')

    json_write(metrics, output_filename)

    print('DONE')


def train_rnn(n=int, batchsize_list=list[int],
              lap_emb_list=list[int], lap_hidden_list=list[int], pos_emb_list=list[int], pos_hidden_list=list[int],
              lr_list=list[float], num_epochs=int, device=torch.device, 
              gamma_scheduler_list=list[float], step_size_scheduler_list=list[int],
              output_filename='rnn_metrics_.json', verbose_every=500):
    
    metrics = []

    print('loading data...')

    race_lap_ngrams = RaceLapNgrams(n=n)
    race_lap_ngrams.split_by_year()
    train_X, train_lap, train_pos = race_lap_ngrams.get_train_tensors()
    val_X, val_lap, val_pos = race_lap_ngrams.get_val_tensors()
    train_X = train_X.to(device)
    train_lap = train_lap.to(device)
    train_pos = train_pos.to(device)
    val_X = val_X.to(device)
    val_lap = val_lap.to(device)
    val_pos = val_pos.to(device)

    print(f'X shape: {train_X.shape} lap shape: {train_lap.shape} pos shape: {train_pos.shape} \n train size: {len(train_X)} val size: {len(val_X)}')

    for batchsize in batchsize_list:

        train_dataloader = DataLoader(TensorDataset(train_X, train_lap, train_pos), batch_size=batchsize, shuffle=True)
        val_dataloader = DataLoader(TensorDataset(val_X, val_lap, val_pos), batch_size=batchsize)

        first_train = next(iter(train_dataloader))
        print(f"data shape: X:{first_train[0].shape} lap:{first_train[1].shape} pos:{first_train[2].shape}")

        for lap_emb, lap_hidden, pos_emb, pos_hidden in zip(lap_emb_list, lap_hidden_list, pos_emb_list, pos_hidden_list):
            for lr in lr_list:
                for gamma_scheduler in gamma_scheduler_list:
                    for step_size_scheduler in step_size_scheduler_list:

                        laptime_model = RNN_regression_v1(data_dim=race_lap_ngrams.data_dim,
                                                        emb_size=lap_emb, hidden_size=lap_hidden)
                        
                        position_model = RNN_MC_v1(data_dim=race_lap_ngrams.data_dim,
                                                emb_size=pos_emb, hidden_size=pos_hidden, output_classes=20)
                        
                        this_metrics = train_inner(lap_model=laptime_model, pos_model=position_model,
                                                train_loader=train_dataloader, val_loader=val_dataloader,
                                                num_epochs=num_epochs, initial_lr=lr, gamma_scheduler=gamma_scheduler, step_size_scheduler=step_size_scheduler,
                                                verbose_every=verbose_every, save_every=400)
                        
                        this_metrics['device'] = str(device)
                        this_metrics['hyperparameters']['init_lr'] = lr
                        this_metrics['hyperparameters']['batchsize'] = batchsize
                        this_metrics['hyperparameters']['num_epochs'] = num_epochs
                        this_metrics['hyperparameters']['n'] = n
                        this_metrics['data_info'] = {
                            'train_size': len(train_dataloader),
                            'val_size': len(val_dataloader),
                            'X shape': first_train[0].shape,
                            'laptime shape': first_train[1].shape,
                            'postion shape': first_train[2].shape,
                        }

                        metrics.append(this_metrics)

    print('saving metrics...')
    json_write(metrics, output_filename)
    print('DONE')


def train_lstm(n=int, batchsize_list=list[int],
               lap_emb_list=list[int], lap_hidden_list=list[int], pos_emb_list=list[int], pos_hidden_list=list[int],
               lr_list=list[float], num_epochs=int, device=torch.device,
                gamma_scheduler_list=list[float], step_size_scheduler_list=list[int],
                output_filename='lstm_metrics_.json', verbose_every=500):
    
    metrics = []

    print('loading data...')

    race_lap_ngrams = RaceLapNgrams(n=n)
    race_lap_ngrams.split_by_year()
    train_X, train_lap, train_pos = race_lap_ngrams.get_train_tensors()
    val_X, val_lap, val_pos = race_lap_ngrams.get_val_tensors()
    train_X = train_X.to(device)
    train_lap = train_lap.to(device)
    train_pos = train_pos.to(device)
    val_X = val_X.to(device)
    val_lap = val_lap.to(device)
    val_pos = val_pos.to(device)

    print(f'X shape: {train_X.shape} lap shape: {train_lap.shape} pos shape: {train_pos.shape} \n train size: {len(train_X)} val size: {len(val_X)}')

    for batchsize in batchsize_list:

        train_dataloader = DataLoader(TensorDataset(train_X, train_lap, train_pos), batch_size=batchsize, shuffle=True)
        val_dataloader = DataLoader(TensorDataset(val_X, val_lap, val_pos), batch_size=batchsize)

        first_train = next(iter(train_dataloader))
        print(f"data shape: X:{first_train[0].shape} lap:{first_train[1].shape} pos:{first_train[2].shape}")

        for lap_emb, lap_hidden, pos_emb, pos_hidden in zip(lap_emb_list, lap_hidden_list, pos_emb_list, pos_hidden_list):
            for lr in lr_list:
                for gamma_scheduler in gamma_scheduler_list:
                    for step_size_scheduler in step_size_scheduler_list:

                        laptime_model = LSTM_regression_v1(data_dim=race_lap_ngrams.data_dim, emb_size=lap_emb, hidden_size=lap_hidden)
                        position_model = LSTM_MC_v1(data_dim=race_lap_ngrams.data_dim, emb_size=pos_emb, hidden_size=pos_hidden, output_classes=20)

                        this_metrics = train_inner(lap_model=laptime_model, pos_model=position_model,
                                                   train_loader=train_dataloader, val_loader=val_dataloader,
                                                   num_epochs=num_epochs, initial_lr=lr, gamma_scheduler=gamma_scheduler, step_size_scheduler=step_size_scheduler,
                                                   verbose_every=verbose_every, save_every=400)
                        this_metrics['device'] = str(device)
                        this_metrics['hyperparameters']['init_lr'] = lr
                        this_metrics['hyperparameters']['batchsize'] = batchsize
                        this_metrics['hyperparameters']['num_epochs'] = num_epochs
                        this_metrics['hyperparameters']['n'] = n
                        this_metrics['data_info'] = {
                            'train_size': len(train_dataloader),
                            'val_size': len(val_dataloader),
                            'X shape': first_train[0].shape,
                            'laptime shape': first_train[1].shape,
                            'postion shape': first_train[2].shape,
                        }

                        metrics.append(this_metrics)
    
    print('saving metrics...')
    json_write(metrics, output_filename)
    print('DONE')


if __name__ == "__main__":

    # train_mlp(n=4, batchsize_list=[32, 128, 256], lap_hidden_list=[10], lap_act_fn='relu', 
    #           pos_hidden_list=[25], pos_act_fn='relu', lr_list=[7e-5, 2e-5], 
    #           num_epochs=25, device=torch.device('cpu'),
    #           output_filename='metrics_2.json')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for n in [3, 7, 10, 15]:

        # mlp_filename = f'mlp_metrics_{n}.json'
        # train_mlp(n=n, batchsize_list=[32, 128], lap_hidden_list=[10, 15], lap_act_fn='relu',
        #         pos_hidden_list=[15, 25], pos_act_fn='relu', lr_list=[1e-4, 5e-5, 1e-5],
        #         num_epochs=200, device=device,
        #         output_filename=mlp_filename, verbose_every=1000)
        
        rnn_filename = f'rnn_metrics_{n}.json'
        train_rnn(n=n, batchsize_list=[32, 128], 
                lap_emb_list=[50], lap_hidden_list=[32, 128], pos_emb_list=[50], pos_hidden_list=[32, 128],
                lr_list=[1e-2, 1e-4], num_epochs=200, device=device,
                gamma_scheduler_list=[0.5, 0.3], step_size_scheduler_list=[10, 20],
                output_filename=rnn_filename, verbose_every=1000)
        
        # lstm_filename = f'lstm_metrics_{n}.json'
        # train_lstm(n=n, batchsize_list=[32, 128],
        #         lap_emb_list=[50, 30], lap_hidden_list=[25, 10], pos_emb_list=[50, 30], pos_hidden_list=[25, 10],
        #         lr_list=[1e-2, 1e-4], num_epochs=200, device=device,
        #         gamma_scheduler_list=[0.5, 0.3], step_size_scheduler_list=[10, 20],
        #         output_filename=lstm_filename, verbose_every=1000)
    
