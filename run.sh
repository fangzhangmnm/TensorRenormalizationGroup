python HOTRG_run.py --filename data/hotrg_gilt_X16 --nLayers 60 --max_dim 16 --gilt_enabled --mcf_enabled
python HOTRG_run.py --filename data/hotrg_gilt_X24 --nLayers 60 --max_dim 24 --gilt_enabled --mcf_enabled
python HOTRG_run.py --filename data/hotrg_gilt_X32 --nLayers 60 --max_dim 32 --gilt_enabled --mcf_enabled
python HOTRG_run.py --filename data/hotrg_gilt_X44 --nLayers 60 --max_dim 44 --gilt_enabled --mcf_enabled

python HOTRG_run.py --filename data/hotrg_X16 --nLayers 60 --max_dim 16 --mcf_enabled
python HOTRG_run.py --filename data/hotrg_X24 --nLayers 60 --max_dim 24 --mcf_enabled
python HOTRG_run.py --filename data/hotrg_X32 --nLayers 60 --max_dim 32 --mcf_enabled
python HOTRG_run.py --filename data/hotrg_X44 --nLayers 60 --max_dim 44 --mcf_enabled

python TNR_run.py --filename data/tnr_X16 --nLayers 20 --tnr_max_dim_TRG 16 --tnr_max_dim_TNR 8