from pathlib import Path
import os, sys
import numpy as np
import json
from tqdm import tqdm
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

from sympy.printing import latex
import sympy

import torch
from kan import KAN
from kan.LBFGS import LBFGS


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
        #print(x[:5], y[:5])
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


        # Early stopping stuff preparation
        no_fit_change_steps = 0
        best_val_rmse = np.inf
        # Remembering first model
        best_model_dict = deepcopy(self.state_dict())

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
                # Remembering best_val_fit and best_model
                best_val_rmse = val_rmse
                best_model_dict = deepcopy(self.state_dict())


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
        
        # Load best model
        self.load_state_dict(best_model_dict)
        self(dataset['train_input'])
        val_loss = loss_fn_eval(self.forward(dataset['val_input'][val_id].to(device)), dataset['val_label'][val_id].to(device))
        val_rmse = torch.sqrt(val_loss).cpu().detach().numpy()

        return results
    
    
class KAN_es_2(KAN):
    """
    KAN class with early stopping training. Early sropping was made closly to skl.MLPRegressor.
    Added interpretable plot function.
    """
    def fit(self, dataset, tol=0.001, n_iter_no_change=10,
            opt="LBFGS", steps=100, log=1, lamb=0., lamb_l1=1., lamb_entropy=2., lamb_coef=0., lamb_coefdiff=0., update_grid=True, grid_update_num=10, loss_fn=None, lr=1.,start_grid_update_step=-1, stop_grid_update_step=50, batch=-1,
            metrics=None, save_fig=False, in_vars=None, out_vars=None, beta=3, save_fig_freq=1, img_folder='./video', singularity_avoiding=False, y_th=1000., reg_metric='edge_forward_spline_n', display_metrics=None):
        '''
        training

        Args:
        -----
            dataset : dic
                contains dataset['train_input'], dataset['train_label'], dataset['test_input'], dataset['test_label']
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
            start_grid_update_step : int
                no grid updates before this training step
            stop_grid_update_step : int
                no grid updates after this training step
            loss_fn : function
                loss function
            lr : float
                learning rate
            batch : int
                batch size, if -1 then full.
            save_fig_freq : int
                save figure every (save_fig_freq) steps
            singularity_avoiding : bool
                indicate whether to avoid singularity for the symbolic part
            y_th : float
                singularity threshold (anything above the threshold is considered singular and is softened in some ways)
            reg_metric : str
                regularization metric. Choose from {'edge_forward_spline_n', 'edge_forward_spline_u', 'edge_forward_sum', 'edge_backward', 'node_backward'}
            metrics : a list of metrics (as functions)
                the metrics to be computed in training
            display_metrics : a list of functions
                the metric to be displayed in tqdm progress bar

        Returns:
        --------
            results : dic
                results['train_loss'], 1D array of training losses (RMSE)
                results['test_loss'], 1D array of test losses (RMSE)
                results['reg'], 1D array of regularization
                other metrics specified in metrics

        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[2,5,1], grid=5, k=3, noise_scale=0.3, seed=2)
        >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
        >>> dataset = create_dataset(f, n_var=2)
        >>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
        >>> model.plot()
        # Most examples in toturals involve the fit() method. Please check them for useness.
        '''

        # Early stopping stuff preparation
        no_fit_change_steps = 0
        best_test_loss = np.inf
        
        if lamb > 0. and not self.save_act:
            print('setting lamb=0. If you want to set lamb > 0, set self.save_act=True')
            
        old_save_act, old_symbolic_enabled = self.disable_symbolic_in_fit(lamb)

        pbar = tqdm(range(steps), desc='description', ncols=100)

        if loss_fn == None:
            loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
        else:
            loss_fn = loss_fn_eval = loss_fn

        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        if opt == "Adam":
            optimizer = torch.optim.Adam(self.get_params(), lr=lr)
        elif opt == "LBFGS":
            optimizer = LBFGS(self.get_params(), lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

        results = {}
        results['train_loss'] = []
        results['test_loss'] = []
        results['reg'] = []
        if metrics != None:
            for i in range(len(metrics)):
                results[metrics[i].__name__] = []

        if batch == -1 or batch > dataset['train_input'].shape[0]:
            batch_size = dataset['train_input'].shape[0]
            batch_size_test = dataset['test_input'].shape[0]
        else:
            batch_size = batch
            batch_size_test = batch

        global train_loss, reg_

        def closure():
            global train_loss, reg_
            optimizer.zero_grad()
            pred = self.forward(dataset['train_input'][train_id], singularity_avoiding=singularity_avoiding, y_th=y_th)
            train_loss = loss_fn(pred, dataset['train_label'][train_id])
            if self.save_act:
                if reg_metric == 'edge_backward':
                    self.attribute()
                if reg_metric == 'node_backward':
                    self.node_attribute()
                reg_ = self.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
            else:
                reg_ = torch.tensor(0.)
            objective = train_loss + lamb * reg_
            objective.backward()
            return objective

        if save_fig:
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)

        for _ in pbar:
            
            if _ == steps-1 and old_save_act:
                self.save_act = True
                
            if save_fig and _ % save_fig_freq == 0:
                save_act = self.save_act
                self.save_act = True
            
            train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
            test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)

            if _ % grid_update_freq == 0 and _ < stop_grid_update_step and update_grid and _ >= start_grid_update_step:
                self.update_grid(dataset['train_input'][train_id])

            if opt == "LBFGS":
                optimizer.step(closure)

            if opt == "Adam":
                pred = self.forward(dataset['train_input'][train_id], singularity_avoiding=singularity_avoiding, y_th=y_th)
                train_loss = loss_fn(pred, dataset['train_label'][train_id])
                if self.save_act:
                    if reg_metric == 'edge_backward':
                        self.attribute()
                    if reg_metric == 'node_backward':
                        self.node_attribute()
                    reg_ = self.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
                else:
                    reg_ = torch.tensor(0.)
                loss = train_loss + lamb * reg_
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            test_loss = loss_fn_eval(self.forward(dataset['test_input'][test_id]), dataset['test_label'][test_id])
            
            
            if metrics != None:
                for i in range(len(metrics)):
                    results[metrics[i].__name__].append(metrics[i]().item())

            results['train_loss'].append(torch.sqrt(train_loss).cpu().detach().numpy())
            results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
            results['reg'].append(reg_.cpu().detach().numpy())

            # Early stopping processing stuff
            if (test_loss > best_test_loss - tol):
                no_fit_change_steps += 1
            else:
                no_fit_change_steps = 0

            if test_loss < best_test_loss:
                # Remembering best_val_fit and best_model
                best_test_loss = test_loss
                best_model_dict = deepcopy(self.state_dict())


            if _ % log == 0:
                if display_metrics == None:
                    pbar.set_description("| trn_loss: %.2e | tst_loss: %.2e | e_stop: %d/%d | reg: %.2e | " % (torch.sqrt(train_loss).cpu().detach().numpy(), torch.sqrt(test_loss).cpu().detach().numpy(), no_fit_change_steps, n_iter_no_change, reg_.cpu().detach().numpy()))
                else:
                    string = ''
                    data = ()
                    for metric in display_metrics:
                        string += f' {metric}: %.2e |'
                        try:
                            results[metric]
                        except:
                            raise Exception(f'{metric} not recognized')
                        data += (results[metric][-1],)
                    pbar.set_description(string % data)
                    
            
            if save_fig and _ % save_fig_freq == 0:
                self.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars, title="Step {}".format(_), beta=beta)
                plt.savefig(img_folder + '/' + str(_) + '.jpg', bbox_inches='tight', dpi=200)
                plt.close()
                self.save_act = save_act
                
            # Checking early stopping criteria
            if no_fit_change_steps==n_iter_no_change:
                print(f'Early stopping criteria raised')
                break

        # Load best model
        self.load_state_dict(best_model_dict)
        self(dataset['train_input'])
        self.log_history('fit')
        # revert back to original state
        self.symbolic_enabled = old_symbolic_enabled
        return results


    def plot(self, reper_in_out='', reper_index=np.nan, hist_plot=False, folder="./figures", beta=3, metric='backward', scale=0.5, scale_scatter = 1, tick=False, sample=False, in_vars=None, out_vars=None, title=None, varscale=1.0, cmap='viridis', vlines_alpha=0.1, vlines_linewidth=10):
        '''
        plot KAN
        
        Args:
        -----
            reper_in_out : str ('' or 'in' or 'out')
                defines reper in input or output layer
            reper_index : np.nan or int
                defines index of reper chanel
            hist_plot : bool
                print histograms along axes
            folder : str
                the folder to store pngs
            beta : float
                positive number. control the transparency of each activation. transparency = tanh(beta*l1).
            mask : bool
                If True, plot with mask (need to run prune() first to obtain mask). If False (by default), plot all activation functions.
            mode : bool
                "supervised" or "unsupervised". If "supervised", l1 is measured by absolution value (not subtracting mean); if "unsupervised", l1 is measured by standard deviation (subtracting mean).
            scale : float
                control the size of the diagram
            in_vars: None or list of str
                the name(s) of input variables
            out_vars: None or list of str
                the name(s) of output variables
            title: None or str
                title
            varscale : float
                the size of input variables
            cmap : str
                color map from plt
            vlines_alpha : float
                transperency for colorfull vlines/hlines
            vlines_linewidth : float
                width of colorfull vlines/hlines
            
        Returns:
        --------
            Figure
            
        Example
        -------
        >>> # see more interactive examples in demos
        >>> model = KAN(width=[2,3,1], grid=3, k=3, noise_scale=1.0)
        >>> x = torch.normal(0,1,size=(100,2))
        >>> model(x) # do a forward pass to obtain model.acts
        >>> model.plot()
        '''
        global Symbol
        
        __file__ = os.path.abspath('')+'\Any_filename.py'
        
        if not self.save_act:
            print('cannot plot since data are not saved. Set save_act=True first.')
        
        # forward to obtain activations
        if self.acts == None:
            if self.cache_data == None:
                raise Exception('model hasn\'t seen any data yet.')
            self.forward(self.cache_data)
            
        if metric == 'backward':
            self.attribute()
            
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        # matplotlib.use('Agg')
        depth = len(self.width) - 1
        
        w_large = 2.0
            
        def prep_ax(ax, tick, alpha_mask, axis_off=False):
            if tick == True:
                ax.tick_params(axis="y", direction="out", labelsize=10*w_large)#, pad=-50
                ax.tick_params(axis="x", direction="out", labelsize=10*w_large)#, pad=-25
                x_min, x_max, y_min, y_max = self.get_range(l, i, j, verbose=False)
                ax.set_xticks([x_min, x_max], ['%2.f' % x_min, '%2.f' % x_max])
                ax.set_yticks([y_min, y_max], ['%2.f' % y_min, '%2.f' % y_max])
            else:
                ax.set_xticks([])
                ax.set_yticks([])
            if alpha_mask == 1:
                ax.patch.set_edgecolor('black')
            else:
                ax.patch.set_edgecolor('white')
            ax.patch.set_linewidth(1.5)
            if axis_off: ax.axis('off')
            
        def plot_color_vlines(x_values, y_values, cmap = 'viridis',
                              alpha=0.7, linewidth=30, plot_object=plt):
            norm = matplotlib.colors.Normalize(vmin=min(y_values), vmax=max(y_values))

            #colormap possible values = viridis, jet, spectral
            colormap = matplotlib.colormaps.get_cmap(cmap) 
            rgba = colormap(norm(y_values))

            for x, c in zip(x_values, rgba):
                plot_object.axvline(x, color=c, alpha=alpha, linewidth=linewidth, zorder=0)
            
        def plot_color_hlines(x_values, y_values, cmap = 'viridis',
                              alpha=0.7, linewidth=30, plot_object=plt):
            norm = matplotlib.colors.Normalize(vmin=min(y_values), vmax=max(y_values))

            #colormap possible values = viridis, jet, spectral
            colormap = matplotlib.colormaps.get_cmap(cmap) 
            rgba = colormap(norm(y_values))

            for x, c in zip(x_values, rgba):
                plot_object.axhline(x, color=c, alpha=alpha, linewidth=linewidth, zorder=0)
            
        if reper_in_out=='in': reper_layer=0
        elif reper_in_out=='out': reper_layer=depth
        if (not reper_in_out=='' and not np.isnan(reper_index)): reper_color = self.acts[reper_layer][:, reper_index]
            
        for l in range(depth):
            for i in range(self.width_in[l]):
                for j in range(self.width_out[l+1]):
                    rank = torch.argsort(self.acts[l][:, i])
                    #rank = np.arange(len(self.acts[l][:, i]))
                    #fig, ax = plt.subplots(figsize=(w_large, w_large))

                    num = rank.shape[0]

                    #print(self.width_in[l])
                    #print(self.width_out[l+1])
                    symbolic_mask = self.symbolic_fun[l].mask[j][i]
                    numeric_mask = self.act_fun[l].mask[i][j]
                    if symbolic_mask > 0. and numeric_mask > 0.:
                        color = 'purple'
                        alpha_mask = 1
                    if symbolic_mask > 0. and numeric_mask == 0.:
                        color = "red"
                        alpha_mask = 1
                    if symbolic_mask == 0. and numeric_mask > 0.:
                        color = "black"
                        alpha_mask = 1
                    if symbolic_mask == 0. and numeric_mask == 0.:
                        color = "white"
                        alpha_mask = 0
                        
                        
                    '''
                    if tick == True:
                        ax.tick_params(axis="y", direction="in", pad=-22, labelsize=50)
                        ax.tick_params(axis="x", direction="in", pad=-15, labelsize=50)
                        x_min, x_max, y_min, y_max = self.get_range(l, i, j, verbose=False)
                        plt.xticks([x_min, x_max], ['%2.f' % x_min, '%2.f' % x_max])
                        plt.yticks([y_min, y_max], ['%2.f' % y_min, '%2.f' % y_max])
                    else:
                        plt.xticks([])
                        plt.yticks([])
                    if alpha_mask == 1:
                        plt.gca().patch.set_edgecolor('black')
                    else:
                        plt.gca().patch.set_edgecolor('white')
                    plt.gca().patch.set_linewidth(1.5)
                    # plt.axis('off')
                    '''
                    # Default plot
                    if (not (reper_in_out in {'in', 'out'}) or np.isnan(reper_index)) and not hist_plot:
                        fig, ax = plt.subplots(figsize=(w_large, w_large))
                        prep_ax(ax, tick, alpha_mask)
                        
                        plt.plot(self.acts[l][:, i][rank].cpu().detach().numpy(), self.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color, lw=5)
                        if sample==True:
                            plt.scatter(self.acts[l][:, i][rank].cpu().detach().numpy(), self.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color, s=400 * scale_scatter ** 2)
                    
                    # Colored scatter
                    elif ((reper_in_out in {'in', 'out'}) and not np.isnan(reper_index)) and not hist_plot:
                        fig, ax = plt.subplots(figsize=(w_large, w_large))
                        prep_ax(ax, tick, alpha_mask)
                        
                        plt.plot(self.acts[l][:, i][rank].cpu().detach().numpy(), self.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color, lw=5, zorder=10)
                        
                        
                        if reper_in_out=='in' and l==0 and i==reper_index:
                            plot_color_vlines(self.acts[l][:, i].cpu().detach().numpy(), reper_color, cmap, vlines_alpha, vlines_linewidth)
                            # All code commented as one bellow draw colorfull scatters instead of vertical/horizontal lines
                            #plt.scatter(self.acts[l][:, i].cpu().detach().numpy(), self.spline_postacts[l][:, j, i].cpu().detach().numpy(), c=reper_color, s=400 * scale_scatter ** 2, cmap=cmap)
                            plt.scatter(self.acts[l][:, i].cpu().detach().numpy(), self.spline_postacts[l][:, j, i].cpu().detach().numpy(), c=reper_color, s=0, cmap=cmap, alpha=1, zorder=-10)
                            plt.colorbar(orientation="horizontal")
                        
                        elif reper_in_out=='out' and l==depth-1 and i==reper_index:
                            plot_color_hlines(self.spline_postacts[l][:, j, i].cpu().detach().numpy(), reper_color, cmap, vlines_alpha, vlines_linewidth)
                            #plt.scatter(self.acts[l][:, i].cpu().detach().numpy(), self.spline_postacts[l][:, j, i].cpu().detach().numpy(), c=reper_color, s=400 * scale_scatter ** 2, cmap=cmap)
                            plt.scatter(self.acts[l][:, i].cpu().detach().numpy(), self.spline_postacts[l][:, j, i].cpu().detach().numpy(), c=reper_color, s=0, cmap=cmap, alpha=1, zorder=-10)
                            plt.colorbar(orientation="vertical")
                            
                        else:
                            plot_color_vlines(self.acts[l][:, i].cpu().detach().numpy(), reper_color, cmap, vlines_alpha, vlines_linewidth)
                            #plt.scatter(self.acts[l][:, i].cpu().detach().numpy(), self.spline_postacts[l][:, j, i].cpu().detach().numpy(), c=reper_color, s=400 * scale_scatter ** 2, cmap=cmap)
                            plt.scatter(self.acts[l][:, i].cpu().detach().numpy(), self.spline_postacts[l][:, j, i].cpu().detach().numpy(), c=reper_color, s=0, cmap=cmap, alpha=1, zorder=-10)
                            
                    # Histogram plots
                    elif (not (reper_in_out in {'in', 'out'}) or np.isnan(reper_index)) and hist_plot:
                        # Create Fig and gridspec
                        fig = plt.figure(figsize=(w_large, w_large))
                        grid = plt.GridSpec(5, 5, hspace=0.1, wspace=0.2)

                        # Define the axes
                        ax_main = fig.add_subplot(grid[:-1, :-1], xticklabels=[], yticklabels=[])
                        ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
                        ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])
                        
                        prep_ax(ax_main, tick, alpha_mask)
                        prep_ax(ax_right, tick, alpha_mask, axis_off=True)
                        prep_ax(ax_bottom, tick, alpha_mask, axis_off=True)

                        # Plot on main ax
                        ax_main.plot(self.acts[l][:, i][rank].cpu().detach().numpy(), self.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color, lw=5)
                        if sample == True:
                            mappable = ax_main.scatter(self.acts[l][:, i][rank].cpu().detach().numpy(), self.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color, s=400 * scale_scatter ** 2)

                        # histogram on the right
                        ax_bottom.hist(self.acts[l][:, i].cpu().detach().numpy(), orientation='vertical')
                        ax_bottom.invert_yaxis()

                        # histogram in the bottom
                        ax_right.hist(self.spline_postacts[l][:, j, i].cpu().detach().numpy(), orientation='horizontal')
                    
                    # Histogram plots and colored scatter
                    elif ((reper_in_out in {'in', 'out'}) and not np.isnan(reper_index)) and hist_plot:
                        
                        if reper_in_out=='in' and l==0 and i==reper_index:
                            # Create Fig and gridspec
                            fig = plt.figure(figsize=(w_large, w_large))
                            grid = plt.GridSpec(5, 6, hspace=0.1, wspace=0.2)
                            
                            # Define the axes
                            ax_main = fig.add_subplot(grid[:-2, :-1], xticklabels=[], yticklabels=[])
                            ax_right = fig.add_subplot(grid[0:-2, -1], xticklabels=[], yticklabels=[])
                            ax_bottom = fig.add_subplot(grid[-2, 0:-1], xticklabels=[], yticklabels=[])
                            ax_colorbar = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])
                            
                            prep_ax(ax_main, tick, alpha_mask)
                            prep_ax(ax_right, tick, alpha_mask, axis_off=True)
                            prep_ax(ax_bottom, tick, alpha_mask, axis_off=True)
                            prep_ax(ax_colorbar, tick, alpha_mask, axis_off=True)
                            
                            # Plot on main ax
                            ax_main.plot(self.acts[l][:, i][rank].cpu().detach().numpy(), self.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color, lw=5, zorder=10)
                            if sample == True:
                                plot_color_vlines(self.acts[l][:, i].cpu().detach().numpy(), reper_color, cmap, vlines_alpha, vlines_linewidth, plot_object=ax_main)
                                #mappable = ax_main.scatter(self.acts[l][:, i].cpu().detach().numpy(), self.spline_postacts[l][:, j, i].cpu().detach().numpy(), c=reper_color, s=400 * scale_scatter ** 2, cmap=cmap)
                                mappable = ax_main.scatter(self.acts[l][:, i].cpu().detach().numpy(), self.spline_postacts[l][:, j, i].cpu().detach().numpy(), c=reper_color, s=0, cmap=cmap, alpha=1, zorder=-10)
                            fig.colorbar(orientation="horizontal", mappable=mappable, cax=ax_colorbar)
                            
                            # histogram on the right
                            ax_bottom.hist(self.acts[l][:, i].cpu().detach().numpy(), orientation='vertical')
                            ax_bottom.invert_yaxis()
                            
                            # histogram in the bottom
                            ax_right.hist(self.spline_postacts[l][:, j, i].cpu().detach().numpy(), orientation='horizontal')

                        
                        elif reper_in_out=='out' and l==depth-1 and self.width_in[l]==1 and i==reper_index:
                            # Create Fig and gridspec
                            fig = plt.figure(figsize=(w_large, w_large))
                            grid = plt.GridSpec(6, 5, hspace=0.1, wspace=0.2)
                            
                            # Define the axes
                            ax_main = fig.add_subplot(grid[:-1, :-2], xticklabels=[], yticklabels=[])
                            ax_right = fig.add_subplot(grid[0:-1, -2], xticklabels=[], yticklabels=[])
                            ax_bottom = fig.add_subplot(grid[-1, 0:-2], xticklabels=[], yticklabels=[])
                            ax_colorbar = fig.add_subplot(grid[0:-1, -1], xticklabels=[], yticklabels=[])
                            
                            prep_ax(ax_main, tick, alpha_mask)
                            prep_ax(ax_right, tick, alpha_mask, axis_off=True)
                            prep_ax(ax_bottom, tick, alpha_mask, axis_off=True)
                            prep_ax(ax_colorbar, tick, alpha_mask, axis_off=True)
                            
                            # Plot on main ax
                            ax_main.plot(self.acts[l][:, i][rank].cpu().detach().numpy(), self.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color, lw=5, zorder=10)
                            if sample == True:
                                plot_color_hlines(self.spline_postacts[l][:, j, i].cpu().detach().numpy(), reper_color, cmap, vlines_alpha, vlines_linewidth, plot_object=ax_main)
                                #mappable = ax_main.scatter(self.acts[l][:, i].cpu().detach().numpy(), self.spline_postacts[l][:, j, i].cpu().detach().numpy(), c=reper_color, s=400 * scale_scatter ** 2, cmap=cmap)
                                mappable = ax_main.scatter(self.acts[l][:, i].cpu().detach().numpy(), self.spline_postacts[l][:, j, i].cpu().detach().numpy(), c=reper_color, s=0, cmap=cmap, alpha=1, zorder=-10)
                            fig.colorbar(orientation="vertical", mappable=mappable, cax=ax_colorbar)
                            
                            # histogram on the right
                            ax_bottom.hist(self.acts[l][:, i].cpu().detach().numpy(), orientation='vertical')
                            ax_bottom.invert_yaxis()
                            
                            # histogram in the bottom
                            ax_right.hist(self.spline_postacts[l][:, j, i].cpu().detach().numpy(), orientation='horizontal')
                            
                        
                        else:
                            # Create Fig and gridspec
                            fig = plt.figure(figsize=(w_large, w_large))
                            grid = plt.GridSpec(5, 5, hspace=0.1, wspace=0.2)

                            # Define the axes
                            ax_main = fig.add_subplot(grid[:-1, :-1], xticklabels=[], yticklabels=[])
                            ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
                            ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

                            prep_ax(ax_main, tick, alpha_mask)
                            prep_ax(ax_right, tick, alpha_mask, axis_off=True)
                            prep_ax(ax_bottom, tick, alpha_mask, axis_off=True)

                            # Plot on main ax
                            ax_main.plot(self.acts[l][:, i][rank].cpu().detach().numpy(), self.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color, lw=5, zorder=10)
                            if sample == True:
                                plot_color_vlines(self.acts[l][:, i].cpu().detach().numpy(), reper_color, cmap, vlines_alpha, vlines_linewidth, plot_object=ax_main)
                                #mappable = ax_main.scatter(self.acts[l][:, i].cpu().detach().numpy(), self.spline_postacts[l][:, j, i].cpu().detach().numpy(), c=reper_color, s=400 * scale_scatter ** 2, cmap=cmap)
                                #mappable = ax_main.scatter(self.acts[l][:, i].cpu().detach().numpy(), self.spline_postacts[l][:, j, i].cpu().detach().numpy(), c=reper_color, s=400 * scale_scatter ** 2, cmap=cmap, alpha=1, zorder=-10)

                            # histogram on the right
                            ax_bottom.hist(self.acts[l][:, i].cpu().detach().numpy(), orientation='vertical')
                            ax_bottom.invert_yaxis()

                            # histogram in the bottom
                            ax_right.hist(self.spline_postacts[l][:, j, i].cpu().detach().numpy(), orientation='horizontal')

                    plt.gca().spines[:].set_color(color)

                    plt.savefig(f'{folder}/sp_{l}_{i}_{j}.png', bbox_inches="tight", dpi=400)
                    plt.close()

        def score2alpha(score):
            return np.tanh(beta * score)

        
        if metric == 'forward_n':
            scores = self.acts_scale
        elif metric == 'forward_u':
            scores = self.edge_actscale
        elif metric == 'backward':
            scores = self.edge_scores
        else:
            raise Exception(f'metric = \'{metric}\' not recognized')
        
        alpha = [score2alpha(score.cpu().detach().numpy()) for score in scores]
            
        # draw skeleton
        width = np.array(self.width)
        width_in = np.array(self.width_in)
        width_out = np.array(self.width_out)
        A = 1
        y0 = 0.3  # height: from input to pre-mult
        z0 = 0.1  # height: from pre-mult to post-mult (input of next layer)

        neuron_depth = len(width)
        min_spacing = A / np.maximum(np.max(width_out), 5)

        max_neuron = np.max(width_out)
        max_num_weights = np.max(width_in[:-1] * width_out[1:])
        y1 = 0.4 / np.maximum(max_num_weights, 5) # size (height/width) of 1D function diagrams
        y2 = 0.15 / np.maximum(max_neuron, 5) # size (height/width) of operations (sum and mult)

        fig, ax = plt.subplots(figsize=(10 * scale, 10 * scale * (neuron_depth - 1) * (y0+z0)))
        # fig, ax = plt.subplots(figsize=(5,5*(neuron_depth-1)*y0))

        # -- Transformation functions
        DC_to_FC = ax.transData.transform
        FC_to_NFC = fig.transFigure.inverted().transform
        # -- Take data coordinates and transform them to normalized figure coordinates
        DC_to_NFC = lambda x: FC_to_NFC(DC_to_FC(x))
        
        # plot scatters and lines
        for l in range(neuron_depth):
            
            n = width_in[l]
            
            # scatters
            for i in range(n):
                plt.scatter(1 / (2 * n) + i / n, l * (y0+z0), s=min_spacing ** 2 * 10000 * scale ** 2, color='black')
                
            # plot connections (input to pre-mult)
            for i in range(n):
                if l < neuron_depth - 1:
                    n_next = width_out[l+1]
                    N = n * n_next
                    for j in range(n_next):
                        id_ = i * n_next + j

                        symbol_mask = self.symbolic_fun[l].mask[j][i]
                        numerical_mask = self.act_fun[l].mask[i][j]
                        if symbol_mask == 1. and numerical_mask > 0.:
                            color = 'purple'
                            alpha_mask = 1.
                        if symbol_mask == 1. and numerical_mask == 0.:
                            color = "red"
                            alpha_mask = 1.
                        if symbol_mask == 0. and numerical_mask == 1.:
                            color = "black"
                            alpha_mask = 1.
                        if symbol_mask == 0. and numerical_mask == 0.:
                            color = "white"
                            alpha_mask = 0.
                        
                        plt.plot([1 / (2 * n) + i / n, 1 / (2 * N) + id_ / N], [l * (y0+z0), l * (y0+z0) + y0/2 - y1], color=color, lw=2 * scale, alpha=alpha[l][j][i] * alpha_mask)
                        plt.plot([1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next], [l * (y0+z0) + y0/2 + y1, l * (y0+z0)+y0], color=color, lw=2 * scale, alpha=alpha[l][j][i] * alpha_mask)
                            
                            
            # plot connections (pre-mult to post-mult, post-mult = next-layer input)
            if l < neuron_depth - 1:
                n_in = width_out[l+1]
                n_out = width_in[l+1]
                mult_id = 0
                for i in range(n_in):
                    if i < width[l+1][0]:
                        j = i
                    else:
                        if i == width[l+1][0]:
                            if isinstance(self.mult_arity,int):
                                ma = self.mult_arity
                            else:
                                ma = self.mult_arity[l+1][mult_id]
                            current_mult_arity = ma
                        if current_mult_arity == 0:
                            mult_id += 1
                            if isinstance(self.mult_arity,int):
                                ma = self.mult_arity
                            else:
                                ma = self.mult_arity[l+1][mult_id]
                            current_mult_arity = ma
                        j = width[l+1][0] + mult_id
                        current_mult_arity -= 1
                        #j = (i-width[l+1][0])//self.mult_arity + width[l+1][0]
                    plt.plot([1 / (2 * n_in) + i / n_in, 1 / (2 * n_out) + j / n_out], [l * (y0+z0) + y0, (l+1) * (y0+z0)], color='black', lw=2 * scale)

                    
                    
            plt.xlim(0, 1)
            plt.ylim(-0.1 * (y0+z0), (neuron_depth - 1 + 0.1) * (y0+z0))


        plt.axis('off')

        for l in range(neuron_depth - 1):
            # plot splines
            n = width_in[l]
            for i in range(n):
                n_next = width_out[l + 1]
                N = n * n_next
                for j in range(n_next):
                    id_ = i * n_next + j
                    im = plt.imread(f'{folder}/sp_{l}_{i}_{j}.png')
                    left = DC_to_NFC([1 / (2 * N) + id_ / N - y1, 0])[0]
                    right = DC_to_NFC([1 / (2 * N) + id_ / N + y1, 0])[0]
                    bottom = DC_to_NFC([0, l * (y0+z0) + y0/2 - y1])[1]
                    up = DC_to_NFC([0, l * (y0+z0) + y0/2 + y1])[1]
                    newax = fig.add_axes([left, bottom, right - left, up - bottom])
                    # newax = fig.add_axes([1/(2*N)+id_/N-y1, (l+1/2)*y0-y1, y1, y1], anchor='NE')
                    if reper_in_out in {'in', 'out'}: # Colorfull lines ON
                        if alpha[l][j][i]==0.0: newax.imshow(im, alpha=alpha[l][j][i])
                        else: newax.imshow(im, alpha=1)
                    else: # Colorfull lines OFF
                        newax.imshow(im, alpha=alpha[l][j][i])
                    newax.axis('off')
                    
              
            # plot sum symbols
            N = n = width_out[l+1]
            for j in range(n):
                id_ = j
                path = os.path.dirname(os.path.abspath(__file__)) + "/assets/img/sum_symbol.png"
                im = plt.imread(path)
                left = DC_to_NFC([1 / (2 * N) + id_ / N - y2, 0])[0]
                right = DC_to_NFC([1 / (2 * N) + id_ / N + y2, 0])[0]
                bottom = DC_to_NFC([0, l * (y0+z0) + y0 - y2])[1]
                up = DC_to_NFC([0, l * (y0+z0) + y0 + y2])[1]
                newax = fig.add_axes([left, bottom, right - left, up - bottom])
                newax.imshow(im)
                newax.axis('off')
                
            # plot mult symbols
            N = n = width_in[l+1]
            n_sum = width[l+1][0]
            n_mult = width[l+1][1]
            for j in range(n_mult):
                id_ = j + n_sum
                path = os.path.dirname(os.path.abspath(__file__)) + "/assets/img/mult_symbol.png"
                im = plt.imread(path)
                left = DC_to_NFC([1 / (2 * N) + id_ / N - y2, 0])[0]
                right = DC_to_NFC([1 / (2 * N) + id_ / N + y2, 0])[0]
                bottom = DC_to_NFC([0, (l+1) * (y0+z0) - y2])[1]
                up = DC_to_NFC([0, (l+1) * (y0+z0) + y2])[1]
                newax = fig.add_axes([left, bottom, right - left, up - bottom])
                newax.imshow(im)
                newax.axis('off')

        if in_vars != None:
            n = self.width_in[0]
            for i in range(n):
                if isinstance(in_vars[i], sympy.Expr):
                    plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), -0.1, f'${latex(in_vars[i])}$', fontsize=40 * scale * varscale, horizontalalignment='center', verticalalignment='center')
                else:
                    plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), -0.1, in_vars[i], fontsize=40 * scale * varscale, horizontalalignment='center', verticalalignment='center')
                
                
                
        if out_vars != None:
            n = self.width_in[-1]
            for i in range(n):
                if isinstance(out_vars[i], sympy.Expr):
                    plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), (y0+z0) * (len(self.width) - 1) + 0.15, f'${latex(out_vars[i])}$', fontsize=40 * scale * varscale, horizontalalignment='center', verticalalignment='center')
                else:
                    plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), (y0+z0) * (len(self.width) - 1) + 0.15, out_vars[i], fontsize=40 * scale * varscale, horizontalalignment='center', verticalalignment='center')

        if title != None:
            plt.gcf().get_axes()[0].text(0.5, (y0+z0) * (len(self.width) - 1) + 0.3, title, fontsize=40 * scale, horizontalalignment='center', verticalalignment='center')
            
        plt.show()