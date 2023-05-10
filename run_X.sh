
# X8
# python find_critical_temp_run.py --filename data/X8/hotrg_gilt_X8_Tc.pth --nLayers 60 --max_dim 8 --model Ising2D --param_name beta --param_min 0.4406 --param_max 0.4407 --tol 0 --observable_name magnetization --gilt_enabled --mcf_enabled 
python HOTRG_run.py --filename data/X8/hotrg_gilt_X8.pth --nLayers 60 --max_dim 8 --gilt_enabled --mcf_enabled --mcf_eps 0 --mcf_max_iter 500 --mcf_phase_iter2 50 --model Ising2D --params_file data/X8/hotrg_gilt_X8_Tc.pth
python scDim_plot.py --filename data/X8/hotrg_gilt_X8 --tensor_path data/X8/hotrg_gilt_X8.pth --is_HOTRG
python linearized_run.py --filename data/X8/hotrg_gilt_X8_lTRG_gilt_L30.pth --tensor_path data/X8/hotrg_gilt_X8.pth --iLayer 30 --mcf_enabled --mcf_eps 0 --mcf_max_iter 500 --mcf_phase_iter2 50 --gilt_enabled  --svd_num_eigvecs 32


# X10
# python find_critical_temp_run.py --filename data/X10/hotrg_gilt_X10_Tc.pth --nLayers 60 --max_dim 10 --model Ising2D --param_name beta --param_min 0.4406 --param_max 0.4407 --tol 0 --observable_name magnetization --gilt_enabled --mcf_enabled
python HOTRG_run.py --filename data/X10/hotrg_gilt_X10.pth --nLayers 60 --max_dim 10 --gilt_enabled --mcf_enabled --mcf_eps 0 --mcf_max_iter 500 --mcf_phase_iter2 50 --model Ising2D --params_file data/X10/hotrg_gilt_X10_Tc.pth
python scDim_plot.py --filename data/X10/hotrg_gilt_X10 --tensor_path data/X10/hotrg_gilt_X10.pth --is_HOTRG
python linearized_run.py --filename data/X10/hotrg_gilt_X10_lTRG_gilt_L30.pth --tensor_path data/X10/hotrg_gilt_X10.pth --iLayer 30 --mcf_enabled --mcf_eps 0 --mcf_max_iter 500 --mcf_phase_iter2 50 --gilt_enabled  --svd_num_eigvecs 32

# X12
# python find_critical_temp_run.py --filename data/X12/hotrg_gilt_X12_Tc.pth --nLayers 60 --max_dim 12 --model Ising2D --param_name beta --param_min 0.4406 --param_max 0.4407 --tol 0 --observable_name magnetization --gilt_enabled --mcf_enabled
python HOTRG_run.py --filename data/X12/hotrg_gilt_X12.pth --nLayers 60 --max_dim 12 --gilt_enabled --mcf_enabled --mcf_eps 0 --mcf_max_iter 500 --mcf_phase_iter2 50 --model Ising2D --params_file data/X12/hotrg_gilt_X12_Tc.pth
python scDim_plot.py --filename data/X12/hotrg_gilt_X12 --tensor_path data/X12/hotrg_gilt_X12.pth --is_HOTRG
python linearized_run.py --filename data/X12/hotrg_gilt_X12_lTRG_gilt_L30.pth --tensor_path data/X12/hotrg_gilt_X12.pth --iLayer 30 --mcf_enabled --mcf_eps 0 --mcf_max_iter 500 --mcf_phase_iter2 50 --gilt_enabled  --svd_num_eigvecs 32

# X14
# python find_critical_temp_run.py --filename data/X14/hotrg_gilt_X14_Tc.pth --nLayers 60 --max_dim 14 --model Ising2D --param_name beta --param_min 0.4406 --param_max 0.4407 --tol 0 --observable_name magnetization --gilt_enabled --mcf_enabled
python HOTRG_run.py --filename data/X14/hotrg_gilt_X14.pth --nLayers 60 --max_dim 14 --gilt_enabled --mcf_enabled --mcf_eps 0 --mcf_max_iter 500 --mcf_phase_iter2 50 --model Ising2D --params_file data/X14/hotrg_gilt_X14_Tc.pth
python scDim_plot.py --filename data/X14/hotrg_gilt_X14 --tensor_path data/X14/hotrg_gilt_X14.pth --is_HOTRG
python linearized_run.py --filename data/X14/hotrg_gilt_X14_lTRG_gilt_L30.pth --tensor_path data/X14/hotrg_gilt_X14.pth --iLayer 30 --mcf_enabled --mcf_eps 0 --mcf_max_iter 500 --mcf_phase_iter2 50 --gilt_enabled  --svd_num_eigvecs 32

# X16
# python find_critical_temp_run.py --filename data/X16/hotrg_gilt_X16_Tc.pth --nLayers 60 --max_dim 16 --model Ising2D --param_name beta --param_min 0.4406 --param_max 0.4407 --tol 0 --observable_name magnetization --gilt_enabled --mcf_enabled
python HOTRG_run.py --filename data/X16/hotrg_gilt_X16.pth --nLayers 60 --max_dim 16 --gilt_enabled --mcf_enabled --mcf_eps 0 --mcf_max_iter 500 --mcf_phase_iter2 50 --model Ising2D --params_file data/X16/hotrg_gilt_X16_Tc.pth
python scDim_plot.py --filename data/X16/hotrg_gilt_X16 --tensor_path data/X16/hotrg_gilt_X16.pth --is_HOTRG
python linearized_run.py --filename data/X16/hotrg_gilt_X16_lTRG_gilt_L30.pth --tensor_path data/X16/hotrg_gilt_X16.pth --iLayer 30 --mcf_enabled --mcf_eps 0 --mcf_max_iter 500 --mcf_phase_iter2 50 --gilt_enabled  --svd_num_eigvecs 32

# X18
# python find_critical_temp_run.py --filename data/X18/hotrg_gilt_X18_Tc.pth --nLayers 60 --max_dim 18 --model Ising2D --param_name beta --param_min 0.4406 --param_max 0.4407 --tol 0 --observable_name magnetization --gilt_enabled --mcf_enabled
python HOTRG_run.py --filename data/X18/hotrg_gilt_X18.pth --nLayers 60 --max_dim 18 --gilt_enabled --mcf_enabled --mcf_eps 0 --mcf_max_iter 500 --mcf_phase_iter2 50 --model Ising2D --params_file data/X18/hotrg_gilt_X18_Tc.pth
python scDim_plot.py --filename data/X18/hotrg_gilt_X18 --tensor_path data/X18/hotrg_gilt_X18.pth --is_HOTRG
python linearized_run.py --filename data/X18/hotrg_gilt_X18_lTRG_gilt_L30.pth --tensor_path data/X18/hotrg_gilt_X18.pth --iLayer 30 --mcf_enabled --mcf_eps 0 --mcf_max_iter 500 --mcf_phase_iter2 50 --gilt_enabled  --svd_num_eigvecs 32

# X20
# python find_critical_temp_run.py --filename data/X20/hotrg_gilt_X20_Tc.pth --nLayers 60 --max_dim 20 --model Ising2D --param_name beta --param_min 0.4406 --param_max 0.4407 --tol 0 --observable_name magnetization --gilt_enabled --mcf_enabled
python HOTRG_run.py --filename data/X20/hotrg_gilt_X20.pth --nLayers 60 --max_dim 20 --gilt_enabled --mcf_enabled --mcf_eps 0 --mcf_max_iter 500 --mcf_phase_iter2 50 --model Ising2D --params_file data/X20/hotrg_gilt_X20_Tc.pth
python scDim_plot.py --filename data/X20/hotrg_gilt_X20 --tensor_path data/X20/hotrg_gilt_X20.pth --is_HOTRG
python linearized_run.py --filename data/X20/hotrg_gilt_X20_lTRG_gilt_L30.pth --tensor_path data/X20/hotrg_gilt_X20.pth --iLayer 30 --mcf_enabled --mcf_eps 0 --mcf_max_iter 500 --mcf_phase_iter2 50 --gilt_enabled  --svd_num_eigvecs 32

# X22
# python find_critical_temp_run.py --filename data/X22/hotrg_gilt_X22_Tc.pth --nLayers 60 --max_dim 22 --model Ising2D --param_name beta --param_min 0.4406 --param_max 0.4407 --tol 0 --observable_name magnetization --gilt_enabled --mcf_enabled
python HOTRG_run.py --filename data/X22/hotrg_gilt_X22.pth --nLayers 60 --max_dim 22 --gilt_enabled --mcf_enabled --mcf_eps 0 --mcf_max_iter 500 --mcf_phase_iter2 50 --model Ising2D --params_file data/X22/hotrg_gilt_X22_Tc.pth
python scDim_plot.py --filename data/X22/hotrg_gilt_X22 --tensor_path data/X22/hotrg_gilt_X22.pth --is_HOTRG
python linearized_run.py --filename data/X22/hotrg_gilt_X22_lTRG_gilt_L30.pth --tensor_path data/X22/hotrg_gilt_X22.pth --iLayer 30 --mcf_enabled --mcf_eps 0 --mcf_max_iter 500 --mcf_phase_iter2 50 --gilt_enabled  --svd_num_eigvecs 32

# X24
# python find_critical_temp_run.py --filename data/X24/hotrg_gilt_X24_Tc.pth --nLayers 60 --max_dim 24 --model Ising2D --param_name beta --param_min 0.4406 --param_max 0.4407 --tol 0 --observable_name magnetization --gilt_enabled --mcf_enabled
python HOTRG_run.py --filename data/X24/hotrg_gilt_X24.pth --nLayers 60 --max_dim 24 --gilt_enabled --mcf_enabled --mcf_eps 0 --mcf_max_iter 500 --mcf_phase_iter2 50 --model Ising2D --params_file data/X24/hotrg_gilt_X24_Tc.pth
python scDim_plot.py --filename data/X24/hotrg_gilt_X24 --tensor_path data/X24/hotrg_gilt_X24.pth --is_HOTRG
python linearized_run.py --filename data/X24/hotrg_gilt_X24_lTRG_gilt_L30.pth --tensor_path data/X24/hotrg_gilt_X24.pth --iLayer 30 --mcf_enabled --mcf_eps 0 --mcf_max_iter 500 --mcf_phase_iter2 50 --gilt_enabled  --svd_num_eigvecs 32

# X26
# python find_critical_temp_run.py --filename data/X26/hotrg_gilt_X26_Tc.pth --nLayers 60 --max_dim 26 --model Ising2D --param_name beta --param_min 0.4406 --param_max 0.4407 --tol 0 --observable_name magnetization --gilt_enabled --mcf_enabled
python HOTRG_run.py --filename data/X26/hotrg_gilt_X26.pth --nLayers 60 --max_dim 26 --gilt_enabled --mcf_enabled --mcf_eps 0 --mcf_max_iter 500 --mcf_phase_iter2 50 --model Ising2D --params_file data/X26/hotrg_gilt_X26_Tc.pth
python scDim_plot.py --filename data/X26/hotrg_gilt_X26 --tensor_path data/X26/hotrg_gilt_X26.pth --is_HOTRG
python linearized_run.py --filename data/X26/hotrg_gilt_X26_lTRG_gilt_L30.pth --tensor_path data/X26/hotrg_gilt_X26.pth --iLayer 30 --mcf_enabled --mcf_eps 0 --mcf_max_iter 500 --mcf_phase_iter2 50 --gilt_enabled  --svd_num_eigvecs 32

# X28
# python find_critical_temp_run.py --filename data/X28/hotrg_gilt_X28_Tc.pth --nLayers 60 --max_dim 28 --model Ising2D --param_name beta --param_min 0.4406 --param_max 0.4407 --tol 0 --observable_name magnetization --gilt_enabled --mcf_enabled
python HOTRG_run.py --filename data/X28/hotrg_gilt_X28.pth --nLayers 60 --max_dim 28 --gilt_enabled --mcf_enabled --mcf_eps 0 --mcf_max_iter 500 --mcf_phase_iter2 50 --model Ising2D --params_file data/X28/hotrg_gilt_X28_Tc.pth
python scDim_plot.py --filename data/X28/hotrg_gilt_X28 --tensor_path data/X28/hotrg_gilt_X28.pth --is_HOTRG
python linearized_run.py --filename data/X28/hotrg_gilt_X28_lTRG_gilt_L30.pth --tensor_path data/X28/hotrg_gilt_X28.pth --iLayer 30 --mcf_enabled --mcf_eps 0 --mcf_max_iter 500 --mcf_phase_iter2 50 --gilt_enabled  --svd_num_eigvecs 32

# X30
# python find_critical_temp_run.py --filename data/X30/hotrg_gilt_X30_Tc.pth --nLayers 60 --max_dim 30 --model Ising2D --param_name beta --param_min 0.4406 --param_max 0.4407 --tol 0 --observable_name magnetization --gilt_enabled --mcf_enabled
python HOTRG_run.py --filename data/X30/hotrg_gilt_X30.pth --nLayers 60 --max_dim 30 --gilt_enabled --mcf_enabled --mcf_eps 0 --mcf_max_iter 500 --mcf_phase_iter2 50 --model Ising2D --params_file data/X30/hotrg_gilt_X30_Tc.pth
python scDim_plot.py --filename data/X30/hotrg_gilt_X30 --tensor_path data/X30/hotrg_gilt_X30.pth --is_HOTRG
python linearized_run.py --filename data/X30/hotrg_gilt_X30_lTRG_gilt_L30.pth --tensor_path data/X30/hotrg_gilt_X30.pth --iLayer 30 --mcf_enabled --mcf_eps 0 --mcf_max_iter 500 --mcf_phase_iter2 50 --gilt_enabled  --svd_num_eigvecs 32

