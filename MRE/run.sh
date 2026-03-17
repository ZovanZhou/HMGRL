CUDA_VISIBLE_DEVICES=$1 python main.py --mode train --eta 1.0 --zeta 1.0
CUDA_VISIBLE_DEVICES=$1 python main.py --mode test