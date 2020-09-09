#!bin/bash
# Var assignment
LR=2.5e-4
GPU=0
du=300
dc=300
echo ========= lr=$LR ==============
for iter in 1 # 2 3 4 5
do
echo --- $Enc - $Dec $iter ---
python Main.py \
-lr $LR \
-gpu $GPU \
-epochs 100 \
-d_h1 $du \
-d_h2 $dc \
-report_loss 720 \
-data_path ./IEMOCAP.pt \
-dataset IEMOCAP
done