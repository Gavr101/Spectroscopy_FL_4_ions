from pathlib import Path
import os, sys
import numpy as np
import json
from tqdm import tqdm

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

import torch
from kan import KAN
from kan.LBFGS import *


def JSON_Create(diction: dict, FileDirectory: str, FileName: str) -> None:
    """
    Function for creating JSON log-file with dictionary.

    :param diction: Dictionary for writing
    :param FileDirectory: Path to logging file
    :param FileName: Name of logging file. Should be ended with ".txt"
    """
    filename = Path(FileDirectory) / FileName  # Full file-path with file-name
    os.makedirs(FileDirectory, exist_ok=True)  # Creating / checking existing of file-path
    with open(filename, 'w') as f:
        json.dump(diction, f, indent=4)  # Writing file


def JSON_Read(FileDirectory: str, FileName: str) -> dict:
    """
    Function for loading dictionary from log-file.

    :param FileDirectory: Path to logging file
    :param FileName: Name of logging file
    """
    filename = Path(FileDirectory) / FileName  # Full file-path with file-name
    with open(filename) as f:
        return json.load(f)  # Loading dictionary


def plotly_multi_scatter(mult_x_y,
                         names = None,
                         main_title="",
                         x_title="",
                         y_title=""):
    """Draws plotly scatter of (x,y).
    mult_x_y - [(x1, y1), (x2, y2), ...] to plot.
    """

    fig = go.Figure()
    fig.update_layout(title=main_title,
                      xaxis_title=x_title,
                      yaxis_title=y_title)
    
    if names==None:
        names = list(range(1, len(mult_x_y)+1))
    
    # Iterating through (x, y) pairs
    for i, (x, y) in enumerate(mult_x_y):
        #fig = px.scatter(x=x, y=y)
        print(x[:5], y[:5])
        fig.add_trace(go.Scatter(x = x, y = y,
                                 name = names[i]))

    fig.show()


def get_sqz_input(x_axis, y_axis):
    ''' Evaluate squeezed data from curve.
    '''
    i_max = np.argmax(y_axis)
    I = y_axis[i_max]  # Max I
    c_I = x_axis[i_max]  # Coordinate of max I

    diff_I = np.absolute(y_axis-I/2)
    c_I2_left = x_axis[ np.argmin(diff_I[:i_max]) ]  # Left I/2 coordinate
    c_I2_right = x_axis[ np.argmin(diff_I[i_max:])+i_max]  # Right I/2 coordinate

    c_I2 = np.mean([c_I2_left, c_I2_right])  # Mean center coordinate on I/2 
    disp_I2 = np.abs(c_I2_right-c_I2_left)  # Width of curve on I/2 y-level

    integr_ratio = np.sum(y_axis[i_max+1:]) / np.sum(y_axis[:i_max])

    sqz_input = [I, c_I, c_I2, disp_I2, integr_ratio]
    
    return sqz_input


def get_all_sqz_input(matr_x, matr_y):
    matr_sqz_input = []
    for x_axis, y_axis in zip(matr_x, matr_y):
        matr_sqz_input.append(get_sqz_input(x_axis, y_axis))


    return np.array(matr_sqz_input)


class KAN_es(KAN):
    """
    KAN class with early stopping training. Early sropping was made closly to skl.MLPRegressor .
    """
    def train_es(self, dataset, tol=0.001, n_iter_no_change=10,
                  opt="LBFGS", steps=100, log=1, lamb=0., lamb_l1=1., lamb_entropy=2., lamb_coef=0., lamb_coefdiff=0., update_grid=True, grid_update_num=10, loss_fn=None, lr=1., stop_grid_update_step=50, batch=-1,
                  small_mag_threshold=1e-16, small_reg_factor=1., metrics=None, sglr_avoid=False, save_fig=False, in_vars=None, out_vars=None, beta=3, save_fig_freq=1, img_folder='./video', device='cpu'):

        ''' Train with early stopping.

        Args:
        -----
        -- Changed --
            dataset : dic
                contains dataset['train_input'], dataset['train_label'], 
                dataset['val_input'], dataset['val_label'], 
                dataset['test_input'], dataset['test_label']
        -- My par-s --
            tol : float
                Delta of validation fit which doesn`t count as fitness improvement. (Tolerence of training).
            n_iter_no_change : int
                Number of iteration with no fit change to early stopping.
        -----
            opt : str
                "LBFGS" or "Adam"
            steps : int
                training steps
            log : int
                logging frequency
            lamb : float
                overall penalty strength
            lamb_l1 : float
                l1 penalty strength
            lamb_entropy : float
                entropy penalty strength
            lamb_coef : float
                coefficient magnitude penalty strength
            lamb_coefdiff : float
                difference of nearby coefficits (smoothness) penalty strength
            update_grid : bool
                If True, update grid regularly before stop_grid_update_step
            grid_update_num : int
                the number of grid updates before stop_grid_update_step
            stop_grid_update_step : int
                no grid updates after this training step
            batch : int
                batch size, if -1 then full.
            small_mag_threshold : float
                threshold to determine large or small numbers (may want to apply larger penalty to smaller numbers)
            small_reg_factor : float
                penalty strength applied to small factors relative to large factos
            device : str
                device   
            save_fig_freq : int
                save figure every (save_fig_freq) step

        Returns:
        --------
            results : dic
                results['train_loss'], 1D array of training losses (RMSE)
                results['val_loss'], 1D array of validation losses (RMSE)
                results['test_loss'], 1D array of test losses (RMSE)
                results['reg'], 1D array of regularization
        '''

        class HiddenPrints:
            ''' Class to avoid unwanted printing'''
            def __enter__(self):
                self._original_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')

            def __exit__(self, exc_type, exc_val, exc_tb):
                sys.stdout.close()
                sys.stdout = self._original_stdout


        # Early stopping stuff preparation
        no_fit_change_steps = 0
        best_val_rmse = np.inf
        s_ckpt = 'tmp_ckpt'
        #best_model = copy.copy(self)

        def reg(acts_scale):

            def nonlinear(x, th=small_mag_threshold, factor=small_reg_factor):
                return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)

            reg_ = 0.
            for i in range(len(acts_scale)):
                vec = acts_scale[i].reshape(-1, )

                p = vec / torch.sum(vec)
                l1 = torch.sum(nonlinear(vec))
                entropy = - torch.sum(p * torch.log2(p + 1e-4))
                reg_ += lamb_l1 * l1 + lamb_entropy * entropy  # both l1 and entropy

            # regularize coefficient to encourage spline to be zero
            for i in range(len(self.act_fun)):
                coeff_l1 = torch.sum(torch.mean(torch.abs(self.act_fun[i].coef), dim=1))
                coeff_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(self.act_fun[i].coef)), dim=1))
                reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

            return reg_

        pbar = tqdm(range(steps), desc='description', ncols=130)

        if loss_fn == None:
            loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
        else:
            loss_fn = loss_fn_eval = loss_fn

        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        if opt == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif opt == "LBFGS":
            optimizer = LBFGS(self.parameters(), lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

        results = {}
        results['train_loss'] = []
        results['val_loss'] = []
        results['test_loss'] = []
        results['reg'] = []
        if metrics != None:
            for i in range(len(metrics)):
                results[metrics[i].__name__] = []

        if batch == -1 or batch > dataset['train_input'].shape[0]:
            batch_size = dataset['train_input'].shape[0]
            batch_size_val = dataset['val_input'].shape[0]
            batch_size_test = dataset['test_input'].shape[0]
        else:
            batch_size = batch
            batch_size_val = batch
            batch_size_test = batch

        global train_loss, reg_

        def closure():
            global train_loss, reg_
            optimizer.zero_grad()
            pred = self.forward(dataset['train_input'][train_id].to(device))
            if sglr_avoid == True:
                id_ = torch.where(torch.isnan(torch.sum(pred, dim=1)) == False)[0]
                train_loss = loss_fn(pred[id_], dataset['train_label'][train_id][id_].to(device))
            else:
                train_loss = loss_fn(pred, dataset['train_label'][train_id].to(device))
            reg_ = reg(self.acts_scale)
            objective = train_loss + lamb * reg_
            objective.backward()
            return objective

        if save_fig:
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)

        # Main training loop
        for _ in pbar:

            train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
            val_id= np.random.choice(dataset['val_input'].shape[0], batch_size_val, replace=False)
            test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)

            if _ % grid_update_freq == 0 and _ < stop_grid_update_step and update_grid:
                self.update_grid_from_samples(dataset['train_input'][train_id].to(device))

            if opt == "LBFGS":
                optimizer.step(closure)

            if opt == "Adam":
                pred = self.forward(dataset['train_input'][train_id].to(device))
                if sglr_avoid == True:
                    id_ = torch.where(torch.isnan(torch.sum(pred, dim=1)) == False)[0]
                    train_loss = loss_fn(pred[id_], dataset['train_label'][train_id][id_].to(device))
                else:
                    train_loss = loss_fn(pred, dataset['train_label'][train_id].to(device))
                reg_ = reg(self.acts_scale)
                loss = train_loss + lamb * reg_
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            val_loss = loss_fn_eval(self.forward(dataset['val_input'][val_id].to(device)), dataset['val_label'][val_id].to(device))
            test_loss = loss_fn_eval(self.forward(dataset['test_input'][test_id].to(device)), dataset['test_label'][test_id].to(device))

            # Early stopping processing stuff
            val_rmse = torch.sqrt(val_loss).cpu().detach().numpy()
            if (val_rmse > best_val_rmse - tol):
                no_fit_change_steps += 1
            else:
                no_fit_change_steps = 0

            if val_rmse < best_val_rmse:
                # remembering best_val_fit and best_model
                best_val_rmse = val_rmse
                with HiddenPrints():
                    self.save_ckpt(s_ckpt)

            if _ % log == 0:
                pbar.set_description("trn_ls: %.2e | vl_ls: %.2e | e_stop: %d/%d | tst_ls: %.2e | reg: %.2e " % (
                                                        torch.sqrt(train_loss).cpu().detach().numpy(), 
                                                        val_rmse, 
                                                        no_fit_change_steps,
                                                        n_iter_no_change,
                                                        torch.sqrt(test_loss).cpu().detach().numpy(), 
                                                        reg_.cpu().detach().numpy() ))

            if metrics != None:
                for i in range(len(metrics)):
                    results[metrics[i].__name__].append(metrics[i]().item())

            results['train_loss'].append(torch.sqrt(train_loss).cpu().detach().numpy())
            results['val_loss'].append(val_rmse)
            results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
            results['reg'].append(reg_.cpu().detach().numpy())

            if save_fig and _ % save_fig_freq == 0:
                self.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars, title="Step {}".format(_), beta=beta)
                plt.savefig(img_folder + '/' + str(_) + '.jpg', bbox_inches='tight', dpi=200)
                plt.close()

            # Checking early stopping criteria
            if no_fit_change_steps==n_iter_no_change:
                print(f'Early stopping criteria raised')
                break
        
        # load best model
        self.load_ckpt(s_ckpt)
        self(dataset['train_input'])
        self.clear_ckpts()

        return results