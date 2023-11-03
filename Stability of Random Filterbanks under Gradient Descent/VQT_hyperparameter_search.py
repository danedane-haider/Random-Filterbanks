import numpy as np
import torch
import torch.nn.functional as F
import pickle
import fb_utils as fb
from torch.utils.data import DataLoader
import random
import os
import itertools

from model import TDFilterbank
from dataset import TinySol
from losses import KappaLoss

# set seed
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)
np.random.seed(0)


if __name__ == "__main__":
    # MARK: - Config

    config = {
            "N": 2**12,
            "J": 96,
            "T": 1024,
            "sr": 16000,
            "fmin": 64,
            "fmax": 8000,
            "stride": 512,
            "batch_size": 64,
            "epochs": 100,
        }

    info_csv_path="TinySOL_metadata.csv"
    data_dir="TinySOL2020"

    random_filterbank = fb.random_filterbank(config["N"], config["J"], config["T"], tight=False, support_only=True)
    target = 'VQT'
    # get current working directory of file
    cwd = os.path.dirname(os.path.abspath(__file__))

    with open(cwd+'/targets/'+target+'.pkl', 'rb') as fp:
        target_filterbank = pickle.load(fp)["freqz"]
        target_filterbank = torch.from_numpy(target_filterbank.T)

    # device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = 'cpu'

    train_dataset = TinySol(info_csv_path, data_dir, config, target_filterbank, 'train')
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    val_dataset = TinySol(info_csv_path, data_dir, config, target_filterbank, 'val')
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    # MARK: - Define all possible combinations of hyperparameters

    def get_all_combinations(dict_of_value_list):
        '''
        This helper function creates a list of dictionaries with all possible combinations from
        a dictionary of lists of values.
        '''
        keys, values = zip(*dict_of_value_list.items())
        combination_dicts = [dict(zip(keys, v))
                            for v in itertools.product(*values)]
        return combination_dicts

    config_list = {
        'beta': np.linspace(1e-5,0.1,50),
        'lr': np.linspace(1e-5,0.1,50),
    }

    config_combinations = get_all_combinations(config_list)

    if not os.path.exists(f"{cwd}/VSQ Hyperparameter Tuning"):
        os.mkdir(f"{cwd}/VSQ Hyperparameter Tuning")

    for i, parameter_config in enumerate(config_combinations):
        print("--------------------------------------------------")
        print(f"Training model {i+1}/{len(config_combinations)}")
        print("--------------------------------------------------")

        # MARK: - Model
        model_baseline = TDFilterbank(config, random_filterbank)
        model_baseline.to(device)
        model_kappa = TDFilterbank(config, random_filterbank)
        model_kappa.to(device)

        loss_baseline = KappaLoss(beta=0)
        loss_kappa = KappaLoss(beta=parameter_config['beta'])

        optimizer_baseline = torch.optim.Adam(model_baseline.parameters(), lr=parameter_config['lr'])
        optimizer_kappa = torch.optim.Adam(model_kappa.parameters(), lr=parameter_config['lr'])

        fit_baseline = []
        fit_val_baseline = []
        fit_kappa = []
        fit_val_kappa = []
        kappa_baseline = []
        kappa_val_baseline = []
        kappa_kappa = []
        kappa_val_kappa = []

        w = model_baseline.psi.weight.detach().numpy()[:,0,:]
        w = np.pad(w, ((0,0),(0, config["N"]-config["T"])), constant_values=0)
        A,B = fb.frame_bounds_lp(w)
        kappa_baseline.append(B/A)
        kappa_val_baseline.append(B/A)

        w = model_kappa.psi.weight.detach().numpy()[:,0,:]
        w = np.pad(w, ((0,0),(0, config["N"]-config["T"])), constant_values=0)
        A,B = fb.frame_bounds_lp(w)
        kappa_kappa.append(B/A)
        kappa_val_kappa.append(B/A)

        print(f"Init condition numbers:")
        print(f"\tBaseline condition number {kappa_baseline[-1]:.2f}")
        print(f"\tKappa condition number {kappa_kappa[-1]:.2f}")

        for epoch in range(config["epochs"]):
            running_loss = 0.0
            running_kappa = 0.0

            running_val_loss = 0.0
            running_val_kappa = 0.0

            model_baseline.train()
            model_kappa.train()

            for batch in train_loader:
                x = batch['x'].to(device)
                x_out = batch['x_out'].to(device)
                
                x_out_baseline = model_baseline(x)
                base_loss_i, _ = loss_baseline(x_out_baseline, x_out)
                
                x_out_kappa = model_kappa(x)
                w = model_kappa.psi.weight[:,0,:]
                w = F.pad(w,(0,config["N"]-config["T"]), value=0)
                base_loss_kappa_i, loss_kappa_i = loss_kappa(x_out_kappa, x_out, w)

                optimizer_baseline.zero_grad()
                base_loss_i.backward()
                optimizer_baseline.step()

                optimizer_kappa.zero_grad()
                loss_kappa_i.backward()
                optimizer_kappa.step()

                running_loss += base_loss_i.item()
                running_kappa += base_loss_kappa_i.item()
            
            model_baseline.eval()
            model_kappa.eval()
            with torch.no_grad():
                for batch in val_loader:
                    x = batch['x'].to(device)
                    x_out = batch['x_out'].to(device)
                    
                    x_out_baseline = model_baseline(x)
                    base_loss_i, _ = loss_baseline(x_out_baseline, x_out)
                    
                    x_out_kappa = model_kappa(x)
                    w = model_kappa.psi.weight[:,0,:]
                    w = F.pad(w,(0,config["N"]-config["T"]), value=0)
                    base_loss_kappa_i, loss_kappa_i = loss_kappa(x_out_kappa, x_out, w)

                    running_val_loss += base_loss_i.item()
                    running_val_kappa += base_loss_kappa_i.item()

            w = model_baseline.psi.weight.detach().numpy()[:,0,:]
            w = np.pad(w, ((0,0),(0, config["N"]-config["T"])), constant_values=0)
            A,B = fb.frame_bounds_lp(w)
            kappa_baseline.append(B/A)
            
            w = model_kappa.psi.weight.detach().numpy()[:,0,:]
            w = np.pad(w, ((0,0),(0, config["N"]-config["T"])), constant_values=0)
            A,B = fb.frame_bounds_lp(w)
            kappa_kappa.append(B/A)

            fit_baseline.append(running_loss/len(train_loader))
            fit_kappa.append(running_kappa/len(train_loader))

            w = model_baseline.psi.weight.detach().numpy()[:,0,:]
            w = np.pad(w, ((0,0),(0, config["N"]-config["T"])), constant_values=0)
            A,B = fb.frame_bounds_lp(w)
            kappa_val_baseline.append(B/A)
            
            w = model_kappa.psi.weight.detach().numpy()[:,0,:]
            w = np.pad(w, ((0,0),(0, config["N"]-config["T"])), constant_values=0)
            A,B = fb.frame_bounds_lp(w)
            kappa_val_kappa.append(B/A)

            fit_val_baseline.append(running_val_loss/len(val_loader))
            fit_val_kappa.append(running_val_kappa/len(val_loader))


            print(f"Epoch {epoch+1}/{config['epochs']}:")
            print(f"\tBaseline Loss: {fit_baseline[-1]:.2f} with condition number {kappa_baseline[-1]:.2f}")
            print(f"\tKappa Loss: {fit_kappa[-1]:.2f} with condition number {kappa_kappa[-1]:.2f}")
            print(f"\tBaseline Val Loss: {fit_val_baseline[-1]:.2f} with condition number {kappa_val_baseline[-1]:.2f}")
            print(f"\tKappa Val Loss: {fit_val_kappa[-1]:.2f} with condition number {kappa_val_kappa[-1]:.2f}")

            if kappa_val_kappa[-1] > 50 and epoch == 0:
                print(f"Condition number exploded for beta {parameter_config['beta']} and lr {parameter_config['lr']}.")
                print("Skipping to next model config.")
                break
        
        if epoch > 0:
            saved_model = {
                'model_baseline': model_baseline,
                'model_kappa': model_kappa,
                'fit_baseline': fit_baseline,
                'fit_kappa': fit_kappa,
                'fit_val_baseline': fit_val_baseline,
                'fit_val_kappa': fit_val_kappa,
                'kappa_baseline': kappa_baseline,
                'kappa_kappa': kappa_kappa,
                'kappa_val_baseline': kappa_val_baseline,
                'kappa_val_kappa': kappa_val_kappa,
                'parameter_config': parameter_config,
            }

            pickle.dump(saved_model, open(f"{cwd}/VSQ Hyperparameter Tuning/saved_model_{i}.pkl", "wb"))

    print("~~Fin~~")
