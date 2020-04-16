from dataclasses import asdict
import copy
import numpy as np
from curvefit.core.model import CurveModel

class APModelLean:

    def __init__(self, df, peaked_groups, model_info, fit_args):

        self.df = df
        self.peaked_groups = peaked_groups 
        self.model_info = model_info
        self.fit_args = fit_args

    def run_init_model(self):
        if len(self.peaked_groups) > 0:
            self.peaked_model = self.run_joint_model(self.peaked_groups, **asdict(self.fit_args))
            
            self.fe_peaked, re_peaked = self.peaked_model.unzip_x(self.peaked_model.result.x)
            self.beta_fe_mean = self.fe_peaked[1]
            self.beta_fe_std = np.std(re_peaked[:, 1])
    
    def run_joint_model(self, groups, fit_dict):
        data = self.df[self.df[self.model_info.group_col].isin(groups)]
        model = CurveModel(data, **asdict(self.model_info))
        model.fit_params(**fit_dict)
        return model 

    def run_single_group_model(self, group, max_time=np.inf):
        data = self.df[
            (self.df[self.model_info.group_col] == group) & 
            (self.df[self.model_info.time_col] <= max_time)
        ]
        model = CurveModel(data, **asdict(self.model_info))
        fit_dict = self._get_single_group_fit_dict(model.num_obs)
        model.fit_params(**fit_dict)
        return model

    def _get_single_group_fit_dict(self, num_obs):
        fit_args = copy.deepcopy(self.fit_args)
        
        fit_args.smart_initialize = False
        fit_args.smart_init_options = None
        fit_args.re_bounds = [[0.0, 0.0]] * self.model_info.num_params
        fit_args.options.maxiter = fit_args.single_group_fit_maxiter
        
        fit_args.fe_gprior[1] = [
            self.beta_fe_mean, 
            self.model_info.prior_modifier(num_obs) * self.beta_fe_std,
        ]
        return asdict(fit_args)

    def fit(self, groups=None):
        self.single_group_models = []
        if groups is None:
            groups = self.df[self.model_info.group_col].unique()
        for group in groups:
            self.single_group_models.append(self.run_single_group_model(group))

    def predict(self, times, group, predict_space):
        predictions = self.single_group_models[group](
            t=times,
            group_name=group,
            prediction_functional_form=predict_space,
        )
        return predictions 


class PVLean:

    def __init__(self, apmodel, pipeline_info):
        self.apmodel = apmodel 
        self.pipeline_info = pipeline_info
        self.df = apmodel.df

    def get_pv_residuals_for_single_group(self, group, theta=1):
        model_full = self.apmodel.run_single_group_model(group)
        times = model_full.t 
        pred_obs = model_full.df[self.pipeline_info.observation_col_pred].to_numpy()
        
        predictions = np.zeros((len(times), len(times)))
        for i in range(len(times)):
            if i < len(times) - 1:
                model = self.apmodel.run_single_group_model(group, max_time=times[i])
            else:
                model = model_full
            predictions[i, :] = model.predict(
                times, 
                prediction_functional_form=self.pipeline_info.predict_space_fun,
            )
        return (predictions - pred_obs) / (predictions**theta)

    def run_pv(self, theta=1, groups=None):
        residuals = {}
        if groups is None:
            groups = self.apmodel.df[self.apmodel.model_info.group_col].unique()
        for group in groups:
            residuals[group].append(self.get_pv_residuals_for_single_group(group, theta))
        return residuals







    



    


    

