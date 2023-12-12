import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", "-g", type=int, required=True)
parser.add_argument("--seed", "-s", type=int, required=True)
parser.add_argument("--lambda_", "-l", type=float, required=True) # 0.01 for {weight_mag, regenerative_reg}, 0.001 = {singular_val}

parser.add_argument("--regularizer", "-r", type=str, required=True)

args = parser.parse_args()

for layer_discount in [0.75, 1.0, 1.25]:
    for update_with_target in [False, True]:
        subprocess.run(f"CUDA_VISIBLE_DEVICES={args.gpu} python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed {args.seed} --regularizer \"{args.regularizer}\" --weight_plot_freq 1000 --lambda_ {args.lambda_} --layer_discount {layer_discount} --update_with_target {update_with_target}", shell=True)
