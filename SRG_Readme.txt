python SRG.py create foo1 --model Ising2D --beta 0.4 --override
python SRG.py create foo2 --model Ising2D --beta 0.42 --override
python SRG.py create foo3 --model Ising2D --beta 0.44 --override
python SRG.py create foo4 --model Ising2D --beta 0.46 --override
python SRG.py create foo5 --model Ising2D --beta 0.48 --override
python SRG.py create foo6 --model Ising2D --beta 0.5 --override

python SRG.py train foo1 foo2 foo3 foo4 foo5 foo6 --device cuda:0 --iter 10