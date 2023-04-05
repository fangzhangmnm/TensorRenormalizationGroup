# Ising2D 0.44068679350977147

python find_critical_temp_run.py --filename data/hotrg_gilt_X24_Tc.pth --nLayers 60 --max_dim 24 --model Ising2D --param_name beta --param_min 0.43 --param_max 0.45 --observable_name magnetization --gilt_enabled --mcf_enabled
# 0.44068381309509275 X24 diff=0.00000298041
# 0.44068394660949706 X30 diff=0.0000028469
python find_critical_temp_run.py --filename data/hotrg_X24_Tc.pth --nLayers 60 --max_dim 24 --model Ising2D --param_name beta --param_min 0.43 --param_max 0.45 --observable_name magnetization --mcf_enabled
# 0.44069609642028806 X24 diff=0.00000930291
# 0.44068705558776855 X30 diff=


python HOTRG_run.py --filename data/hotrg_X24.pth --nLayers 60 --max_dim 24 --mcf_enabled  --model Ising2D --params_file data/hotrg_X24_Tc.pth
python HOTRG_run.py --filename data/hotrg_X24_lowB.pth --nLayers 60 --max_dim 24 --mcf_enabled --model Ising2D --params '{"beta":0.440686}'
python HOTRG_run.py --filename data/hotrg_X24_highB.pth --nLayers 60 --max_dim 24 --mcf_enabled --model Ising2D --params '{"beta":0.440706}'

python HOTRG_run.py --filename data/hotrg_gilt_X24.pth --nLayers 60 --max_dim 24 --gilt_enabled --mcf_enabled --model Ising2D --params_file data/hotrg_gilt_X24_Tc.pth
python HOTRG_run.py --filename data/hotrg_gilt_X24_lowB.pth --nLayers 60 --max_dim 24 --gilt_enabled --mcf_enabled --model Ising2D --params '{"beta":0.440674}'
python HOTRG_run.py --filename data/hotrg_gilt_X24_highB.pth --nLayers 60 --max_dim 24 --gilt_enabled --mcf_enabled --model Ising2D --params '{"beta":0.440694}'



python scDim_plot.py --filename data/hotrg_gilt_X24 --tensor_path data/hotrg_gilt_X24.pth --is_HOTRG
python scDim_plot.py --filename data/hotrg_X24 --tensor_path data/hotrg_X24.pth --is_HOTRG

python linearized_cyl_run.py --filename data/hotrg_gilt_X24_cyl_L20.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 20 --svd_num_eigvecs 32
# 0.0000, 0.1256, 1.0059, 1.1320, 1.1320, 2.0146, 2.0151, 2.0165, 2.0166,
# 2.1411, 2.1463, 2.1471, 3.0289, 3.0372, 3.0375, 3.0377, 3.0421, 3.1534,
# 3.1535, 3.1587, 3.1587, 3.1787, 3.1788, 4.0380, 4.0467, 4.0475, 4.0512,
# 4.0515, 4.0815, 4.0840, 4.0842, 4.0857
python linearized_run.py --filename data/hotrg_gilt_X24_lTRG_gilt_L20.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 20 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
# 0.0000, 0.1275, 1.0027, 1.1269, 1.1448, 1.9985, 2.0004, 2.0042, 2.0434,
# 2.1215, 2.1981, 2.2049, 2.5006, 2.6803, 2.6884, 2.7914, 2.8247, 2.8255,
# 2.8467, 2.8467, 3.0117, 3.0952, 3.1032, 3.1032, 3.1158, 3.1158, 3.1594,
# 3.1594, 3.2014, 3.2044, 3.2044, 3.2255

python linearized_cyl_run.py --filename data/hotrg_X24_cyl_L20.pth --tensor_path data/hotrg_X24.pth --iLayer 20 --svd_num_eigvecs 32
# 0.0000, 0.1217, 1.0085, 1.1422, 1.1422, 2.0496, 2.0525, 2.0701, 2.0701,
# 2.1795, 2.2202, 2.2752, 2.9658, 2.9680, 2.9680, 2.9707, 3.1250, 3.1269,
# 3.1308, 3.1308, 3.1360, 3.1625, 3.2200, 3.2200, 3.2411, 3.2411, 3.2661,
# 3.2898, 3.2898, 3.3869, 3.3869, 3.4483

python linearized_run.py --filename data/hotrg_X24_lTRG_L20.pth --tensor_path data/hotrg_X24.pth --iLayer 20 --mcf_enabled --svd_num_eigvecs 32
# 0.0000, 0.1266, 1.0836, 1.1857, 1.1911, 2.0098, 2.0936, 2.1000, 2.1085,
# 2.1085, 2.1140, 2.1140, 2.1165, 2.1165, 2.1251, 2.1251, 2.1430, 2.1430,
# 2.1463, 2.1463, 2.1653, 2.1653, 2.1883, 2.1926, 2.1926, 2.1961, 2.1961,
# 2.1982, 2.1982, 2.2196, 2.2196, 2.2305

python linearized_run.py --filename data/hotrg_gilt_X24_lTRG_L20.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 20 --mcf_enabled --svd_num_eigvecs 32

# 0.0000, 0.1292, 1.0350, 1.1553, 1.1560, 2.0125, 2.0135, 2.0167, 2.0553,
# 2.0567, 2.0951, 2.1714, 2.2281, 2.2281, 2.2517, 2.2517, 2.3449, 2.3449,
# 2.3704, 2.3818, 2.3951, 2.4809, 2.4948, 2.5126, 2.5204, 2.5379, 2.5427,
# 2.5525, 2.6167, 2.6526, 2.6682, 2.6788





python linearized_run.py --filename data/hotrg_X24_lTRG_L20.pth --tensor_path data/hotrg_X24.pth --iLayer 20 --mcf_enabled --svd_num_eigvecs 32


# python linearized_cyl_run.py --filename data/hotrg_gilt_X24_cyl_L24 --tensor_path data/hotrg_gilt_X24.pth --iLayer 24 --svd_num_eigvecs 32
# # 0.0000, 0.1261, 1.0090, 1.1353, 1.1354, 2.0208, 2.0212, 2.0227, 2.0228,
# # 2.1478, 2.1528, 2.1536, 3.0382, 3.0466, 3.0469, 3.0473, 3.0516, 3.1632,
# # 3.1635, 3.1685, 3.1685, 3.1885, 3.1888, 4.0504, 4.0593, 4.0600, 4.0639,
# # 4.0641, 4.0944, 4.0970, 4.0974, 4.0988
# python linearized_run.py --filename data/hotrg_gilt_X24_lTRG_gilt_L24 --tensor_path data/hotrg_gilt_X24.pth --iLayer 24 --mcf_enabled --gilt_enabled --svd_num_eigvecs 32
# # 0.0000, 0.1276, 1.0024, 1.1264, 1.1444, 1.9984, 2.0011, 2.0035, 2.0442,
# # 2.1201, 2.1974, 2.2041, 2.4995, 2.6826, 2.6826, 2.7900, 2.8226, 2.8250,
# # 2.8470, 2.8470, 3.0132, 3.0944, 3.1041, 3.1041, 3.1177, 3.1177, 3.1625,
# # 3.1625, 3.2028, 3.2028, 3.2032, 3.2245



python correlation_run.py --filename data/hotrg_X24_torus_correlation_y_10.pkl --points_filename data/torus_correlation_points_y_10.pkl --tensors_filename data/hotrg_X24.pth --log2Size 10

python correlation_run.py --filename data/hotrg_gilt_X24_torus_correlation_y_10.pkl --points_filename data/torus_correlation_points_y_10.pkl --tensors_filename data/hotrg_gilt_X24.pth --log2Size 10


python correlation_run.py --filename data/hotrg_X24_smearing_between_edge_10.pkl --points_filename data/smearing_between_edge_10.pkl --tensors_filename data/hotrg_X24.pth --log2Size 10

python correlation_run.py --filename data/hotrg_gilt_X24_smearing_between_edge_10.pkl --points_filename data/smearing_between_edge_10.pkl --tensors_filename data/hotrg_gilt_X24.pth --log2Size 10

python correlation_run.py --filename data/hotrg_X24_smearing_corner_10.pkl --points_filename data/smearing_corner_10.pkl --tensors_filename data/hotrg_X24.pth --log2Size 10

python correlation_run.py --filename data/hotrg_gilt_X24_smearing_corner_10.pkl --points_filename data/smearing_corner_10.pkl --tensors_filename data/hotrg_gilt_X24.pth --log2Size 10

python correlation_run.py --filename data/hotrg_gilt_X24_4pt_correlation.pkl --points_filename data/4pt_correlation_points.pkl --tensors_filename data/hotrg_gilt_X24.pth --log2Size 10

python correlation_run.py --filename data/hotrg_gilt_X24_4pt_correlation_30.pkl --points_filename data/4pt_correlation_points_30.pkl --tensors_filename data/hotrg_gilt_X24.pth --log2Size 30

python correlation_run.py --filename data/hotrg_gilt_X24_sigma_sigma_epsilon_correlation.pkl --points_filename data/sigma_sigma_epsilon_correlation_points.pkl --tensors_filename data/hotrg_gilt_X24.pth --log2Size 10



python find_critical_temp_run.py --filename data/hotrg_gilt_nomcf_X24_Tc.pth --nLayers 60 --max_dim 24 --model Ising2D --param_name beta --param_min 0.43 --param_max 0.45 --observable_name magnetization --gilt_enabled
# 0.44068388938903813

python HOTRG_run.py --filename data/hotrg_nomcf_X24.pth --nLayers 60 --max_dim 24 --mcf_enabled  --model Ising2D --params_file data/hotrg_gilt_nomcf_X24_Tc.pth