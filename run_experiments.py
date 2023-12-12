import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", "-s", type=int, required=True)

args = parser.parse_args()

for regularizer in ["none", "weight_mag", "regenerative_reg", "singular_loss"]:
    for lambda_ in [0.01, 0.1, 1.0]:
        for layer_discount in [0.5, 0.75, 1.0, 1.25, 1.5]:
            for update_with_target in [False, True]:
                subprocess.run(f"CUDA_VISIBLE_DEVICES={args.seed} python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed {args.seed} --regularizer \"{regularizer}\" --weight_plot_freq 1000 --lambda_ {lambda_} --layer_discount {layer_discount} --update_with_target {update_with_target}", shell=True)
