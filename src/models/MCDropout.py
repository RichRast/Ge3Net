import torch
from src.main.evaluation import modelOuts

class MC_Dropout(object):
    def __init__(self, mc_samples, n_win, variance):
        self.mc_samples = mc_samples
        self.variance = variance
        self.n_win = n_win

    def _getFromMcSamples(self, outs_list):
        cat_outs = torch.cat(outs_list, 0).contiguous()
        mean_outs = cat_outs.view(self.mc_samples, -1, self.n_win , cat_outs.shape[-1]).mean(0)
        var_outs=None
        if self.variance:
            var_outs = cat_outs.view(self.mc_samples, -1, self.n_win , cat_outs.shape[-1]).var(0)
        return mean_outs, var_outs

    def __call__(self, forward_fn, x):
        main_list, x_nxt_list, aux_list=[],[],[]
        for _ in range(self.mc_samples):
            outs, x_nxt = forward_fn(x)
            # only collect and mc dropout for the main network
            main_list.append(outs.coord_main)
            aux_list.append(outs.coord_aux)
            x_nxt_list.append(x_nxt)

        outs_main, y_var = self._getFromMcSamples(main_list)
        outs_aux, _ = self._getFromMcSamples(aux_list)
        x_nxt, _ = self._getFromMcSamples(x_nxt_list)
        return modelOuts(coord_main=outs_main, coord_aux=outs_aux, y_var=y_var), x_nxt, main_list