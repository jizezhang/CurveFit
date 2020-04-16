from dataclasses import dataclass
from typing import List
import numpy as np
from curvefit.core.functions import normal_loss

@dataclass
class ModelInfo:
    time_col: str
    observation_col_fit: str 
    group_col: str 
    covariates_cols: List[str]
    observation_se_col: str = None
    params_names: List[str] = ['alpha', 'beta', 'p']
    params_link_fun: List[callable] = [np.exp, lambda x: x, np.exp]
    variable_link_fun: List[callable]
    fit_space_fun: callable
    loss_fun: callable = normal_loss
    observation_se_fun: callable = None
    prior_modifier: callable
    num_params: int = 3


@dataclass
class PipelineInfo:
    observation_col_pred: str
    predict_space_fun: callable
    all_covariates_names: List[str]


@dataclass
class OptimizerOptions:
    ftol: float 
    gtol: float 
    maxiter: int 
    disp: bool = True


@dataclass
class FitArgs:
    fe_init: List[float]
    fe_bounds: List[List[float]]
    fe_gpriors: List[List[float]]
    re_bounds: List[List[float]]
    re_gpriors: List[List[float]]
    smart_initialize: bool 
    smart_init_options: OptimizerOptions = None
    options: OptimizerOptions 
    fun_gprior: callable
    single_group_fit_maxiter: int = 500






