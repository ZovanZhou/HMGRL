CUDA_VISIBLE_DEVICES=$1 python main.py --task 1 --mode train --eta 10.0 --zeta 10.0
CUDA_VISIBLE_DEVICES=$1 python main.py --task 1 --mode test