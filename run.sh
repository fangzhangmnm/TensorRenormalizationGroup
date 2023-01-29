# Ising2D 0.44068679350977147
python find_critical_temp_run.py --filename data/hotrg_X16_Tc --nLayers 60 --max_dim 16 --model Ising2D --param_name beta --param_min 0.43 --param_max 0.45 --observable_name magnetization
python find_critical_temp_run.py --filename data/hotrg_gilt_X16_Tc --nLayers 60 --max_dim 16 --model Ising2D --param_name beta --param_min 0.43 --param_max 0.45 --observable_name magnetization --gilt_enabled --mcf_enabled
# 0.44068255424499514
python find_critical_temp_run.py --filename data/hotrg_gilt_X24_Tc --nLayers 60 --max_dim 24 --model Ising2D --param_name beta --param_min 0.43 --param_max 0.45 --observable_name magnetization --gilt_enabled --mcf_enabled
# 0.44068381309509275

#Ising3D ~0.2216544
python find_critical_temp_run.py --filename data/ising3d_hotrg_X8_Tc --nLayers 60 --max_dim 8 --model Ising3D --param_name beta --param_min 0.20 --param_max 0.24 --observable_name magnetization
# 0.22138627052307125
python find_critical_temp_run.py --filename data/ising3d_hotrg_X9_Tc --nLayers 60 --max_dim 9 --model Ising3D --param_name beta --param_min 0.20 --param_max 0.24 --observable_name magnetization
# 0.22182593345642088
python find_critical_temp_run.py --filename data/ising3d_hotrg_X10_Tc --nLayers 60 --max_dim 10 --model Ising3D --param_name beta --param_min 0.20 --param_max 0.24 --observable_name magnetization
# 0.2221596431732178


python HOTRG_run.py --filename data/hotrg_gilt_X16 --nLayers 60 --max_dim 16 --gilt_enabled --mcf_enabled
python HOTRG_run.py --filename data/hotrg_gilt_X24 --nLayers 60 --max_dim 24 --gilt_enabled --mcf_enabled
python HOTRG_run.py --filename data/hotrg_gilt_X32 --nLayers 60 --max_dim 32 --gilt_enabled --mcf_enabled
python HOTRG_run.py --filename data/hotrg_gilt_X44 --nLayers 60 --max_dim 44 --gilt_enabled --mcf_enabled  --device cuda:1

python HOTRG_run.py --filename data/hotrg_X16 --nLayers 60 --max_dim 16 --mcf_enabled
python HOTRG_run.py --filename data/hotrg_X24 --nLayers 60 --max_dim 24 --mcf_enabled
python HOTRG_run.py --filename data/hotrg_X32 --nLayers 60 --max_dim 32 --mcf_enabled
python HOTRG_run.py --filename data/hotrg_X44 --nLayers 60 --max_dim 44 --mcf_enabled

python TNR_run.py --filename data/tnr_X16 --nLayers 10 --tnr_max_dim_TRG 16 --tnr_max_dim_TNR 8 --device cuda:1
python TNR_run.py --filename data/tnr_X24 --nLayers 10 --tnr_max_dim_TRG 24 --tnr_max_dim_TNR 12 --device cuda:1

python HOTRG_run.py --filename data/ising3d_hotrg_X8 --model Ising3D --nLayers 60 --max_dim 16 --mcf_enabled


python scDim_plot.py --filename data/hotrg_gilt_X24 --tensor_path data/hotrg_gilt_X24.pkl --is_HOTRG
python scDim_plot.py --filename data/hotrg_gilt_X32 --tensor_path data/hotrg_gilt_X32.pkl --is_HOTRG
python scDim_plot.py --filename data/hotrg_gilt_X44 --tensor_path data/hotrg_gilt_X44.pkl --is_HOTRG
python scDim_plot.py --filename data/tnr_X16 --tensor_path data/tnr_X16.pkl
python scDim_plot.py --filename data/tnr_X24 --tensor_path data/tnr_X24.pkl --num_scaling_dims 128

python linearized_run.py --filename data/hotrg_gilt_X24_lTRG_L10 --tensor_path data/hotrg_gilt_X24.pkl --iLayer 10 --mcf_enabled
python linearized_run.py --filename data/hotrg_gilt_X24_lTRG_gilt_L10 --tensor_path data/hotrg_gilt_X24.pkl --iLayer 10 --mcf_enabled --gilt_enabled
python linearized_run.py --filename data/hotrg_gilt_X24_lTRG_L20 --tensor_path data/hotrg_gilt_X24.pkl --iLayer 20 --mcf_enabled
python linearized_run.py --filename data/hotrg_gilt_X24_lTRG_gilt_L20 --tensor_path data/hotrg_gilt_X24.pkl --iLayer 20 --mcf_enabled --gilt_enabled

python linearized_run.py --filename data/hotrg_gilt_X16_lTRG_gilt_L20 --tensor_path data/hotrg_gilt_X16.pkl --iLayer 20 --mcf_enabled --gilt_enabled
python linearized_run.py --filename data/hotrg_gilt_X32_lTRG_gilt_L20 --tensor_path data/hotrg_gilt_X32.pkl --iLayer 20 --mcf_enabled --gilt_enabled --device cuda:1
python linearized_run.py --filename data/hotrg_gilt_X44_lTRG_gilt_L20 --tensor_path data/hotrg_gilt_X44.pkl --iLayer 20 --mcf_enabled --gilt_enabled --device cpu
python linearized_run.py --filename data/tnr_X16_lTRG_L10 --tensor_path data/tnr_X16.pkl --iLayer 10 --mcf_enabled
python linearized_run.py --filename data/tnr_X16_lTRG_gilt_L10 --tensor_path data/tnr_X16.pkl --iLayer 10 --mcf_enabled --gilt_enabled



python linearized_run.py --filename data/hotrg_gilt_X24_lTRG_gilt_jax_L20 --tensor_path data/hotrg_gilt_X24.pkl --iLayer 20 --mcf_enabled --gilt_enabled --linearized_use_jax
python linearized_run.py --filename data/hotrg_gilt_X24_lTRG_full_jax_L20 --tensor_path data/hotrg_gilt_X24.pkl --iLayer 20 --mcf_enabled --gilt_enabled --linearized_full --linearized_use_jax


python linearized_cyl_run.py --filename data/hotrg_gilt_X24_cyl_L20 --tensor_path data/hotrg_gilt_X24.pkl --iLayer 20

python linearized_cyl_run.py --filename data/tnr_X16_cyl_L10 --tensor_path data/tnr_X16.pkl --iLayer 10 --svd_num_eigvecs 32
        # 0.0000, 0.1135, 1.0016, 1.1382, 1.1382, 2.0007, 2.0008, 2.0013, 2.0013,
        # 2.1146, 2.1372, 2.1381, 2.9999, 2.9999, 3.0002, 3.0022, 3.0024, 3.1128,
        # 3.1128, 3.1153, 3.1153, 3.1381, 3.1381, 3.9991, 3.9992, 3.9998, 4.0002,
        # 4.0009, 4.0009, 4.0050, 4.0064, 4.0064

python linearized_run.py --filename data/hotrg_gilt_X24_lTRG_L15 --tensor_path data/hotrg_gilt_X24.pkl --iLayer 15


python linearized_run.py --filename data/hotrg_gilt_X24_lTRG_gilt_L20 --tensor_path data/hotrg_gilt_X24.pkl --iLayer 20 --mcf_enabled --gilt_enabled --svd_num_eigvecs 24
python linearized_run.py --filename data/hotrg_gilt_X24_lTRG_gilt_bigger_L20 --tensor_path data/hotrg_gilt_X24.pkl --iLayer 20 --mcf_enabled --gilt_enabled --svd_num_eigvecs 24 --gilt_eps 8e-6

python linearized_run.py --filename data/hotrg_gilt_X16_lTRG_full_L20 --tensor_path data/hotrg_gilt_X16.pkl --iLayer 20 --mcf_enabled --gilt_enabled --linearized_full --svd_num_eigvecs 32