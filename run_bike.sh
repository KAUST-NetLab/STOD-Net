#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J stgcn
#SBATCH -o stgcn.%J.out
#SBATCH -e stgcn.%J.err
#SBATCH --array=1-20
#SBATCH --time=01:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1


# load the module
#module load pytorch/1.2.0-cuda10.0-cudnn7.6-py3.7
conda activate /ibex/scratch/zhanc0c/projects/st_dense_gcn/env

loss_values=( 'l1' 'l2')
depth_values=( 40)
lr_values=(  1e-4 )
step_values=( 1 )
gcn_values=( 'gcn' )
file_values=( 'bike' )
transfer_values=( ff )
fusion_values=( no-fusion )
norm_values=( '01_sigmoid' )
c_values=( 5 )
p_values=( 4 )
t_values=( 1 )

exp_values=( $(seq 1 10) )
#exp_values=( 1 )

gcn_layer_values=( 2 )
gate_values=( 3 )

alpha_values=( 1.0 )
beta_values=( 1.0 )

trial=${SLURM_ARRAY_TASK_ID}
loss=${loss_values[$(( trial % ${#loss_values[@]} ))]}
trial=$(( trial / ${#loss_values[@]} ))
depth=${depth_values[$(( trial % ${#depth_values[@]} ))]}
trial=$(( trial / ${#depth_values[@]} ))
lr=${lr_values[$(( trial % ${#lr_values[@]} ))]}
trial=$(( trial / ${#lr_values[@]} ))

steps=${step_values[$(( trial % ${#step_values[@]} ))]}
trial=$(( trial / ${#step_values[@]} ))

c=${c_values[$(( trial % ${#c_values[@]} ))]}
trial=$(( trial / ${#c_values[@]} ))
p=${p_values[$(( trial % ${#p_values[@]} ))]}
trial=$(( trial / ${#p_values[@]} ))
t=${t_values[$(( trial % ${#t_values[@]} ))]}
trial=$(( trial / ${#t_values[@]} ))

exp=${exp_values[$(( trial % ${#exp_values[@]} ))]}
trial=$(( trial / ${#exp_values[@]} ))

gcn_layer=${gcn_layer_values[$(( trial % ${#gcn_layer_values[@]} ))]}
trial=$(( trial / ${#gcn_layer_values[@]} ))

gate=${gate_values[$(( trial % ${#gate_values[@]} ))]}
trial=$(( trial / ${#gate_values[@]} ))

alpha=${alpha_values[$(( trial % ${#alpha_values[@]} ))]}
trial=$(( trial / ${#alpha_values[@]} ))

beta=${beta_values[$(( trial % ${#beta_values[@]} ))]}
trial=$(( trial / ${#beta_values[@]} ))

gcn=${gcn_values[$(( trial % ${#gcn_values[@]} ))]}
trial=$(( trial / ${#gcn_values[@]} ))

norm=${norm_values[$(( trial % ${#norm_values[@]} ))]}
trial=$(( trial / ${#norm_values[@]} ))

fusion=${fusion_values[$(( trial % ${#fusion_values[@]} ))]}
trial=$(( trial / ${#fusion_values[@]} ))

transfer=${transfer_values[$(( trial % ${#transfer_values[@]} ))]}
trial=$(( trial / ${#transfer_values[@]} ))


file=${file_values[$(( trial % ${#file_values[@]} ))]}

python st_gcn_v2.py -traffic ${file} -gcn_type ${gcn} -lr ${lr} -depth ${depth} -loss ${loss} -epoch_size 100 -ibex -close_size ${c} -period_size ${p} -trend_size ${t} -norm_type ${norm} -${fusion} -${transfer} -gcn_layer ${gcn_layer} -gate_type ${gate} -beta ${beta} -alpha ${alpha} -exp ${exp} -batch_size 128