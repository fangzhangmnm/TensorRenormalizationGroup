
# X32
# python find_critical_temp_run.py --filename data/X32/hotrg_gilt_X32_Tc.pth --nLayers 60 --max_dim 32 --model Ising2D --param_name beta --param_min 0.4406838 --param_max 0.4406842 --tol 0 --observable_name magnetization --gilt_enabled --mcf_enabled
python HOTRG_run.py --filename data/X32/hotrg_gilt_X32.pth --nLayers 60 --max_dim 32 --gilt_enabled --mcf_enabled --mcf_eps 0 --mcf_max_iter 500 --mcf_phase_iter2 50 --model Ising2D --params_file data/X32/hotrg_gilt_X32_Tc.pth
python scDim_plot.py --filename data/X32/hotrg_gilt_X32 --tensor_path data/X32/hotrg_gilt_X32.pth --is_HOTRG
python linearized_run.py --filename data/X32/hotrg_gilt_X32_lTRG_gilt_L30.pth --tensor_path data/X32/hotrg_gilt_X32.pth --iLayer 30 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32

# X34
# python find_critical_temp_run.py --filename data/X34/hotrg_gilt_X34_Tc.pth --nLayers 60 --max_dim 34 --model Ising2D --param_name beta --param_min 0.4406838 --param_max 0.4406842 --tol 0 --observable_name magnetization --gilt_enabled --mcf_enabled
python HOTRG_run.py --filename data/X34/hotrg_gilt_X34.pth --nLayers 60 --max_dim 34 --gilt_enabled --mcf_enabled --model Ising2D --params_file data/X34/hotrg_gilt_X34_Tc.pth
python scDim_plot.py --filename data/X34/hotrg_gilt_X34 --tensor_path data/X34/hotrg_gilt_X34.pth --is_HOTRG
python linearized_run.py --filename data/X34/hotrg_gilt_X34_lTRG_gilt_L30.pth --tensor_path data/X34/hotrg_gilt_X34.pth --iLayer 30 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32

# X36
# python find_critical_temp_run.py --filename data/X36/hotrg_gilt_X36_Tc.pth --nLayers 60 --max_dim 36 --model Ising2D --param_name beta --param_min 0.4406838 --param_max 0.4406842 --tol 0 --observable_name magnetization --gilt_enabled --mcf_enabled
python HOTRG_run.py --filename data/X36/hotrg_gilt_X36.pth --nLayers 60 --max_dim 36 --gilt_enabled --mcf_enabled --model Ising2D --params_file data/X36/hotrg_gilt_X36_Tc.pth
python scDim_plot.py --filename data/X36/hotrg_gilt_X36 --tensor_path data/X36/hotrg_gilt_X36.pth --is_HOTRG
python linearized_run.py --filename data/X36/hotrg_gilt_X36_lTRG_gilt_L30.pth --tensor_path data/X36/hotrg_gilt_X36.pth --iLayer 30 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32

# X38
# python find_critical_temp_run.py --filename data/X38/hotrg_gilt_X38_Tc.pth --nLayers 60 --max_dim 38 --model Ising2D --param_name beta --param_min 0.4406838 --param_max 0.4406842 --tol 0 --observable_name magnetization --gilt_enabled --mcf_enabled
python HOTRG_run.py --filename data/X38/hotrg_gilt_X38.pth --nLayers 60 --max_dim 38 --gilt_enabled --mcf_enabled --model Ising2D --params_file data/X38/hotrg_gilt_X38_Tc.pth
python scDim_plot.py --filename data/X38/hotrg_gilt_X38 --tensor_path data/X38/hotrg_gilt_X38.pth --is_HOTRG
python linearized_run.py --filename data/X38/hotrg_gilt_X38_lTRG_gilt_L30.pth --tensor_path data/X38/hotrg_gilt_X38.pth --iLayer 30 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32

# X40
# python find_critical_temp_run.py --filename data/X40/hotrg_gilt_X40_Tc.pth --nLayers 60 --max_dim 40 --model Ising2D --param_name beta --param_min 0.4406838 --param_max 0.4406842 --tol 0 --observable_name magnetization --gilt_enabled --mcf_enabled
# python HOTRG_run.py --filename data/X40/hotrg_gilt_X40.pth --nLayers 60 --max_dim 40 --gilt_enabled --mcf_enabled --model Ising2D --params_file data/X40/hotrg_gilt_X40_Tc.pth
# python scDim_plot.py --filename data/X40/hotrg_gilt_X40 --tensor_path data/X40/hotrg_gilt_X40.pth --is_HOTRG
# python linearized_run.py --filename data/X40/hotrg_gilt_X40_lTRG_gilt_L30.pth --tensor_path data/X40/hotrg_gilt_X40.pth --iLayer 30 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32

# # X42
# # python find_critical_temp_run.py --filename data/X42/hotrg_gilt_X42_Tc.pth --nLayers 60 --max_dim 42 --model Ising2D --param_name beta --param_min 0.4406838 --param_max 0.4406842 --tol 0 --observable_name magnetization --gilt_enabled --mcf_enabled
# python HOTRG_run.py --filename data/X42/hotrg_gilt_X42.pth --nLayers 60 --max_dim 42 --gilt_enabled --mcf_enabled --model Ising2D --params_file data/X42/hotrg_gilt_X42_Tc.pth
# python scDim_plot.py --filename data/X42/hotrg_gilt_X42 --tensor_path data/X42/hotrg_gilt_X42.pth --is_HOTRG
# python linearized_run.py --filename data/X42/hotrg_gilt_X42_lTRG_gilt_L30.pth --tensor_path data/X42/hotrg_gilt_X42.pth --iLayer 30 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32