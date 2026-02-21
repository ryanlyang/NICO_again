import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paste ALL your log lines for trial_27, trial_34, trial_36 (and any others) here.
# Needs lines like:
# [trial_34][Epoch 12] val_acc=... ig_fwd_kl=... test={'...','overall': ...}
LOG_TEXT = r"""
asks: /home/ryreu/guided_cnn/code/HaveNicoLearn/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap
Images: /home/ryreu/guided_cnn/code/NICO-plus/data/Unzip_DG_Bench/DG_Benchmark/NICO_DG
Output: /home/ryreu/guided_cnn/NICO_runs/output/debug_metric_autumn_rock
Targets: autumn rock
IG steps: 16
/home/ryreu/miniconda3/envs/gals_a100/bin/python
Val split: 16% split from train (no *_val.txt found)
Running held-out targets=['autumn', 'rock'], sources=['dim', 'grass', 'outdoor', 'water'], train=46171, val=8794, ig_steps=16, device=cuda:0

=== trial_36 start params={'base_lr': 0.012972276913609016, 'classifier_lr': 6.79761541785652e-05, 'lr2_mult': 0.018463196348278824, 'attention_epoch': 11, 'kl_lambda_start': 0.46479079030852405, 'kl_increment': 0.0} ===
[trial_36][Epoch 0] val_acc=0.563111 gxi_rev_kl=6.078101 gxi_fwd_kl=2.008956 ig_rev_kl=6.075740 ig_fwd_kl=2.006624 cam_rev_kl=11.293195 cam_fwd_kl=1.691293 test={'autumn': 50.02748763056624, 'rock': 48.395842747401716, 'overall': 49.13194444444444}
[trial_36][Epoch 1] val_acc=0.654196 gxi_rev_kl=6.079356 gxi_fwd_kl=2.010396 ig_rev_kl=6.075320 ig_fwd_kl=2.006928 cam_rev_kl=11.281657 cam_fwd_kl=1.684280 test={'autumn': 58.87850467289719, 'rock': 56.9814731134207, 'overall': 57.83730158730159}
[trial_36][Epoch 2] val_acc=0.690812 gxi_rev_kl=6.079427 gxi_fwd_kl=2.010498 ig_rev_kl=6.075775 ig_fwd_kl=2.007256 cam_rev_kl=11.235908 cam_fwd_kl=1.667308 test={'autumn': 62.56184716877405, 'rock': 58.92453682783552, 'overall': 60.56547619047619}
[trial_36][Epoch 3] val_acc=0.717194 gxi_rev_kl=6.079966 gxi_fwd_kl=2.010712 ig_rev_kl=6.076766 ig_fwd_kl=2.007695 cam_rev_kl=11.250613 cam_fwd_kl=1.670781 test={'autumn': 63.60637713029137, 'rock': 63.08178942611839, 'overall': 63.31845238095238}
[trial_36][Epoch 4] val_acc=0.735729 gxi_rev_kl=6.081247 gxi_fwd_kl=2.011476 ig_rev_kl=6.076276 ig_fwd_kl=2.007161 cam_rev_kl=11.266981 cam_fwd_kl=1.674067 test={'autumn': 67.4546454095657, 'rock': 64.66335291459558, 'overall': 65.92261904761905}
[trial_36][Epoch 5] val_acc=0.737889 gxi_rev_kl=6.082497 gxi_fwd_kl=2.012821 ig_rev_kl=6.077467 ig_fwd_kl=2.008343 cam_rev_kl=11.292500 cam_fwd_kl=1.683070 test={'autumn': 65.58548653106102, 'rock': 63.80478987799368, 'overall': 64.60813492063492}
[trial_36][Epoch 6] val_acc=0.717307 gxi_rev_kl=6.080559 gxi_fwd_kl=2.011471 ig_rev_kl=6.076212 ig_fwd_kl=2.007551 cam_rev_kl=11.273872 cam_fwd_kl=1.680728 test={'autumn': 63.71632765255635, 'rock': 63.39810212381383, 'overall': 63.541666666666664}
[trial_36][Epoch 7] val_acc=0.777917 gxi_rev_kl=6.082206 gxi_fwd_kl=2.012689 ig_rev_kl=6.076216 ig_fwd_kl=2.007405 cam_rev_kl=11.262112 cam_fwd_kl=1.672156 test={'autumn': 70.53326003298515, 'rock': 69.00135562584727, 'overall': 69.69246031746032}
[trial_36][Epoch 8] val_acc=0.782806 gxi_rev_kl=6.083505 gxi_fwd_kl=2.013648 ig_rev_kl=6.077476 ig_fwd_kl=2.008173 cam_rev_kl=11.281094 cam_fwd_kl=1.678798 test={'autumn': 70.53326003298515, 'rock': 69.3176683235427, 'overall': 69.86607142857143}
[trial_36][Epoch 9] val_acc=0.779395 gxi_rev_kl=6.084338 gxi_fwd_kl=2.014856 ig_rev_kl=6.076919 ig_fwd_kl=2.008163 cam_rev_kl=11.271382 cam_fwd_kl=1.675525 test={'autumn': 71.68774051676745, 'rock': 67.78129236330773, 'overall': 69.5436507936508}
[trial_36][Epoch 10] val_acc=0.783261 gxi_rev_kl=6.083577 gxi_fwd_kl=2.014063 ig_rev_kl=6.077579 ig_fwd_kl=2.008546 cam_rev_kl=11.301662 cam_fwd_kl=1.686615 test={'autumn': 70.47828477185267, 'rock': 68.91098056936285, 'overall': 69.61805555555556}
[trial_36][Epoch 11] val_acc=0.823402 gxi_rev_kl=6.082555 gxi_fwd_kl=2.013369 ig_rev_kl=6.076069 ig_fwd_kl=2.007463 cam_rev_kl=11.144699 cam_fwd_kl=1.630641 test={'autumn': 74.10665200659703, 'rock': 72.02892001807501, 'overall': 72.96626984126983}
[trial_36][Epoch 12] val_acc=0.829543 gxi_rev_kl=6.080988 gxi_fwd_kl=2.012077 ig_rev_kl=6.075103 ig_fwd_kl=2.006767 cam_rev_kl=11.085487 cam_fwd_kl=1.610890 test={'autumn': 75.20615722924684, 'rock': 72.97785811116132, 'overall': 73.98313492063492}
[trial_36][Epoch 13] val_acc=0.834774 gxi_rev_kl=6.079855 gxi_fwd_kl=2.011174 ig_rev_kl=6.074255 ig_fwd_kl=2.006123 cam_rev_kl=11.046595 cam_fwd_kl=1.598833 test={'autumn': 75.53600879604178, 'rock': 73.6556710347944, 'overall': 74.50396825396825}
[trial_36][Epoch 14] val_acc=0.835115 gxi_rev_kl=6.079882 gxi_fwd_kl=2.011344 ig_rev_kl=6.074007 ig_fwd_kl=2.006069 cam_rev_kl=11.038529 cam_fwd_kl=1.596511 test={'autumn': 76.30566245189665, 'rock': 73.97198373248983, 'overall': 75.02480158730158}
[trial_36][Epoch 15] val_acc=0.836025 gxi_rev_kl=6.079033 gxi_fwd_kl=2.010643 ig_rev_kl=6.073592 ig_fwd_kl=2.005744 cam_rev_kl=11.025865 cam_fwd_kl=1.592763 test={'autumn': 76.36063771302913, 'rock': 74.24310890194306, 'overall': 75.1984126984127}
[trial_36][Epoch 16] val_acc=0.839663 gxi_rev_kl=6.079496 gxi_fwd_kl=2.011113 ig_rev_kl=6.073606 ig_fwd_kl=2.005812 cam_rev_kl=11.020363 cam_fwd_kl=1.591126 test={'autumn': 76.08576140736669, 'rock': 74.69498418436511, 'overall': 75.32242063492063}
[trial_36][Epoch 17] val_acc=0.838071 gxi_rev_kl=6.079764 gxi_fwd_kl=2.011357 ig_rev_kl=6.073546 ig_fwd_kl=2.005796 cam_rev_kl=11.012862 cam_fwd_kl=1.588736 test={'autumn': 76.96536558548654, 'rock': 75.01129688206055, 'overall': 75.89285714285714}
[trial_36][Epoch 18] val_acc=0.838754 gxi_rev_kl=6.079000 gxi_fwd_kl=2.010725 ig_rev_kl=6.073199 ig_fwd_kl=2.005505 cam_rev_kl=11.007188 cam_fwd_kl=1.586745 test={'autumn': 77.18526663001649, 'rock': 75.3727971079982, 'overall': 76.19047619047619}
[trial_36][Epoch 19] val_acc=0.841710 gxi_rev_kl=6.079277 gxi_fwd_kl=2.011052 ig_rev_kl=6.073398 ig_fwd_kl=2.005734 cam_rev_kl=11.004055 cam_fwd_kl=1.585839 test={'autumn': 76.91039032435404, 'rock': 75.05648441030276, 'overall': 75.89285714285714}
[trial_36][Epoch 20] val_acc=0.842961 gxi_rev_kl=6.079530 gxi_fwd_kl=2.011240 ig_rev_kl=6.073403 ig_fwd_kl=2.005737 cam_rev_kl=10.999419 cam_fwd_kl=1.584299 test={'autumn': 77.07531610775152, 'rock': 75.41798463624039, 'overall': 76.16567460317461}
[trial_36][Epoch 21] val_acc=0.843302 gxi_rev_kl=6.079210 gxi_fwd_kl=2.010963 ig_rev_kl=6.073115 ig_fwd_kl=2.005543 cam_rev_kl=10.988075 cam_fwd_kl=1.580709 test={'autumn': 77.24024189114898, 'rock': 75.28242205151378, 'overall': 76.16567460317461}
[trial_36][Epoch 22] val_acc=0.843871 gxi_rev_kl=6.079157 gxi_fwd_kl=2.010898 ig_rev_kl=6.073183 ig_fwd_kl=2.005590 cam_rev_kl=10.989288 cam_fwd_kl=1.581041 test={'autumn': 77.62506871907641, 'rock': 74.96610935381834, 'overall': 76.16567460317461}
[trial_36][Epoch 23] val_acc=0.843302 gxi_rev_kl=6.079464 gxi_fwd_kl=2.011229 ig_rev_kl=6.073176 ig_fwd_kl=2.005630 cam_rev_kl=10.987703 cam_fwd_kl=1.580808 test={'autumn': 77.130291368884, 'rock': 75.73429733393583, 'overall': 76.3640873015873}
[trial_36][Epoch 24] val_acc=0.844212 gxi_rev_kl=6.078678 gxi_fwd_kl=2.010578 ig_rev_kl=6.072803 ig_fwd_kl=2.005301 cam_rev_kl=10.982637 cam_fwd_kl=1.578843 test={'autumn': 77.24024189114898, 'rock': 75.68910980569363, 'overall': 76.38888888888889}
[trial_36][Epoch 25] val_acc=0.844894 gxi_rev_kl=6.078843 gxi_fwd_kl=2.010683 ig_rev_kl=6.072807 ig_fwd_kl=2.005334 cam_rev_kl=10.979954 cam_fwd_kl=1.577882 test={'autumn': 77.62506871907641, 'rock': 75.68910980569363, 'overall': 76.5625}
[trial_36][Epoch 26] val_acc=0.846031 gxi_rev_kl=6.079182 gxi_fwd_kl=2.011051 ig_rev_kl=6.072707 ig_fwd_kl=2.005255 cam_rev_kl=10.980565 cam_fwd_kl=1.578356 test={'autumn': 77.24024189114898, 'rock': 75.50835969272481, 'overall': 76.28968253968254}
[trial_36][Epoch 27] val_acc=0.845349 gxi_rev_kl=6.079160 gxi_fwd_kl=2.010986 ig_rev_kl=6.073074 ig_fwd_kl=2.005548 cam_rev_kl=10.982197 cam_fwd_kl=1.578853 test={'autumn': 77.18526663001649, 'rock': 75.64392227745142, 'overall': 76.33928571428571}
[trial_36][Epoch 28] val_acc=0.845349 gxi_rev_kl=6.079260 gxi_fwd_kl=2.011140 ig_rev_kl=6.073023 ig_fwd_kl=2.005513 cam_rev_kl=10.979171 cam_fwd_kl=1.577982 test={'autumn': 77.18526663001649, 'rock': 75.77948486217804, 'overall': 76.41369047619048}
[trial_36][Epoch 29] val_acc=0.845008 gxi_rev_kl=6.079033 gxi_fwd_kl=2.010933 ig_rev_kl=6.072872 ig_fwd_kl=2.005436 cam_rev_kl=10.975481 cam_fwd_kl=1.576716 test={'autumn': 77.57009345794393, 'rock': 76.00542250338907, 'overall': 76.71130952380952}

=== trial_36 best summary ===
[trial_36][BEST val_acc] value=0.846031 epoch=26 test={'autumn': 77.24024189114898, 'rock': 75.50835969272481, 'overall': 76.28968253968254}
[trial_36][BEST gxi_rev_kl] value=6.078101 epoch=0 test={'autumn': 50.02748763056624, 'rock': 48.395842747401716, 'overall': 49.13194444444444}
[trial_36][BEST gxi_fwd_kl] value=2.008956 epoch=0 test={'autumn': 50.02748763056624, 'rock': 48.395842747401716, 'overall': 49.13194444444444}
[trial_36][BEST ig_rev_kl] value=6.072707 epoch=26 test={'autumn': 77.24024189114898, 'rock': 75.50835969272481, 'overall': 76.28968253968254}
[trial_36][BEST ig_fwd_kl] value=2.005255 epoch=26 test={'autumn': 77.24024189114898, 'rock': 75.50835969272481, 'overall': 76.28968253968254}
[trial_36][BEST cam_rev_kl] value=10.975481 epoch=29 test={'autumn': 77.57009345794393, 'rock': 76.00542250338907, 'overall': 76.71130952380952}
[trial_36][BEST cam_fwd_kl] value=1.576716 epoch=29 test={'autumn': 77.57009345794393, 'rock': 76.00542250338907, 'overall': 76.71130952380952}
[trial_36] saved: /home/ryreu/guided_cnn/NICO_runs/output/debug_metric_autumn_rock/trial_36_best_metrics.json

=== trial_34 start params={'base_lr': 0.00021433680976098852, 'classifier_lr': 0.0007434605528635105, 'lr2_mult': 0.045599132525539994, 'attention_epoch': 8, 'kl_lambda_start': 0.14669902456222098, 'kl_increment': 0.0} ===
[trial_34][Epoch 0] val_acc=0.800318 gxi_rev_kl=6.077325 gxi_fwd_kl=2.008755 ig_rev_kl=6.069864 ig_fwd_kl=2.002456 cam_rev_kl=11.106245 cam_fwd_kl=1.614737 test={'autumn': 78.44969763606377, 'rock': 78.21961138725712, 'overall': 78.3234126984127}
[trial_34][Epoch 1] val_acc=0.827041 gxi_rev_kl=6.078052 gxi_fwd_kl=2.009182 ig_rev_kl=6.070085 ig_fwd_kl=2.002748 cam_rev_kl=11.119319 cam_fwd_kl=1.617882 test={'autumn': 81.0885101704233, 'rock': 79.48486217803887, 'overall': 80.20833333333333}
[trial_34][Epoch 2] val_acc=0.840914 gxi_rev_kl=6.078566 gxi_fwd_kl=2.009472 ig_rev_kl=6.069964 ig_fwd_kl=2.002485 cam_rev_kl=11.131816 cam_fwd_kl=1.621213 test={'autumn': 81.74821330401319, 'rock': 80.61455038409399, 'overall': 81.12599206349206}
[trial_34][Epoch 3] val_acc=0.851945 gxi_rev_kl=6.080263 gxi_fwd_kl=2.010859 ig_rev_kl=6.070972 ig_fwd_kl=2.003333 cam_rev_kl=11.140839 cam_fwd_kl=1.623781 test={'autumn': 81.36338647608576, 'rock': 81.11161319475825, 'overall': 81.22519841269842}
[trial_34][Epoch 4] val_acc=0.854901 gxi_rev_kl=6.080428 gxi_fwd_kl=2.010863 ig_rev_kl=6.071213 ig_fwd_kl=2.003484 cam_rev_kl=11.136736 cam_fwd_kl=1.621814 test={'autumn': 81.91313908741067, 'rock': 81.15680072300046, 'overall': 81.49801587301587}
[trial_34][Epoch 5] val_acc=0.861838 gxi_rev_kl=6.080917 gxi_fwd_kl=2.011371 ig_rev_kl=6.071472 ig_fwd_kl=2.003780 cam_rev_kl=11.143781 cam_fwd_kl=1.624254 test={'autumn': 81.91313908741067, 'rock': 81.60867600542251, 'overall': 81.74603174603175}
[trial_34][Epoch 6] val_acc=0.862520 gxi_rev_kl=6.081254 gxi_fwd_kl=2.011688 ig_rev_kl=6.071689 ig_fwd_kl=2.004106 cam_rev_kl=11.142811 cam_fwd_kl=1.624185 test={'autumn': 82.02308960967565, 'rock': 81.60867600542251, 'overall': 81.79563492063492}
[trial_34][Epoch 7] val_acc=0.867068 gxi_rev_kl=6.080603 gxi_fwd_kl=2.011199 ig_rev_kl=6.071425 ig_fwd_kl=2.003828 cam_rev_kl=11.147093 cam_fwd_kl=1.624836 test={'autumn': 81.80318856514569, 'rock': 81.74423859014912, 'overall': 81.77083333333333}
[trial_34][Epoch 8] val_acc=0.867978 gxi_rev_kl=6.080423 gxi_fwd_kl=2.011065 ig_rev_kl=6.071200 ig_fwd_kl=2.003725 cam_rev_kl=11.137784 cam_fwd_kl=1.621619 test={'autumn': 82.2979659153381, 'rock': 82.28648892905558, 'overall': 82.29166666666667}
[trial_34][Epoch 9] val_acc=0.869570 gxi_rev_kl=6.080581 gxi_fwd_kl=2.011230 ig_rev_kl=6.071194 ig_fwd_kl=2.003748 cam_rev_kl=11.126840 cam_fwd_kl=1.618236 test={'autumn': 82.40791643760308, 'rock': 82.24130140081337, 'overall': 82.31646825396825}
[trial_34][Epoch 10] val_acc=0.869115 gxi_rev_kl=6.080289 gxi_fwd_kl=2.010917 ig_rev_kl=6.071065 ig_fwd_kl=2.003658 cam_rev_kl=11.120446 cam_fwd_kl=1.616169 test={'autumn': 82.73776800439802, 'rock': 82.01536375960235, 'overall': 82.34126984126983}
[trial_34][Epoch 11] val_acc=0.869798 gxi_rev_kl=6.080725 gxi_fwd_kl=2.011319 ig_rev_kl=6.071159 ig_fwd_kl=2.003756 cam_rev_kl=11.110883 cam_fwd_kl=1.613262 test={'autumn': 82.57284222100056, 'rock': 82.01536375960235, 'overall': 82.26686507936508}
[trial_34][Epoch 12] val_acc=0.870594 gxi_rev_kl=6.079946 gxi_fwd_kl=2.010818 ig_rev_kl=6.070777 ig_fwd_kl=2.003443 cam_rev_kl=11.103604 cam_fwd_kl=1.610907 test={'autumn': 82.62781748213304, 'rock': 82.4672390420244, 'overall': 82.53968253968254}
[trial_34][Epoch 13] val_acc=0.871503 gxi_rev_kl=6.079766 gxi_fwd_kl=2.010629 ig_rev_kl=6.070407 ig_fwd_kl=2.003210 cam_rev_kl=11.095406 cam_fwd_kl=1.608280 test={'autumn': 82.68279274326554, 'rock': 81.83461364663353, 'overall': 82.2172619047619}
[trial_34][Epoch 14] val_acc=0.870821 gxi_rev_kl=6.080587 gxi_fwd_kl=2.011302 ig_rev_kl=6.071308 ig_fwd_kl=2.003918 cam_rev_kl=11.094317 cam_fwd_kl=1.607778 test={'autumn': 82.90269378779549, 'rock': 82.24130140081337, 'overall': 82.53968253968254}
[trial_34][Epoch 15] val_acc=0.869002 gxi_rev_kl=6.079572 gxi_fwd_kl=2.010490 ig_rev_kl=6.070478 ig_fwd_kl=2.003244 cam_rev_kl=11.087342 cam_fwd_kl=1.605577 test={'autumn': 82.68279274326554, 'rock': 82.15092634432897, 'overall': 82.39087301587301}
[trial_34][Epoch 16] val_acc=0.870139 gxi_rev_kl=6.080029 gxi_fwd_kl=2.010846 ig_rev_kl=6.070629 ig_fwd_kl=2.003411 cam_rev_kl=11.083656 cam_fwd_kl=1.604517 test={'autumn': 82.07806487080813, 'rock': 81.92498870311793, 'overall': 81.99404761904762}
[trial_34][Epoch 17] val_acc=0.870707 gxi_rev_kl=6.080186 gxi_fwd_kl=2.010994 ig_rev_kl=6.070883 ig_fwd_kl=2.003604 cam_rev_kl=11.077103 cam_fwd_kl=1.602568 test={'autumn': 83.23254535459043, 'rock': 82.4220515137822, 'overall': 82.78769841269842}
[trial_34][Epoch 18] val_acc=0.868888 gxi_rev_kl=6.079525 gxi_fwd_kl=2.010472 ig_rev_kl=6.070398 ig_fwd_kl=2.003215 cam_rev_kl=11.078733 cam_fwd_kl=1.602929 test={'autumn': 82.847718526663, 'rock': 81.97017623136014, 'overall': 82.36607142857143}
[trial_34][Epoch 19] val_acc=0.868774 gxi_rev_kl=6.080077 gxi_fwd_kl=2.011003 ig_rev_kl=6.070660 ig_fwd_kl=2.003471 cam_rev_kl=11.076718 cam_fwd_kl=1.602339 test={'autumn': 82.90269378779549, 'rock': 82.28648892905558, 'overall': 82.56448412698413}
[trial_34][Epoch 20] val_acc=0.872527 gxi_rev_kl=6.080130 gxi_fwd_kl=2.011026 ig_rev_kl=6.070694 ig_fwd_kl=2.003519 cam_rev_kl=11.068711 cam_fwd_kl=1.600115 test={'autumn': 82.57284222100056, 'rock': 82.33167645729779, 'overall': 82.44047619047619}
[trial_34][Epoch 21] val_acc=0.871390 gxi_rev_kl=6.079973 gxi_fwd_kl=2.010886 ig_rev_kl=6.070598 ig_fwd_kl=2.003427 cam_rev_kl=11.064972 cam_fwd_kl=1.598760 test={'autumn': 82.847718526663, 'rock': 81.83461364663353, 'overall': 82.29166666666667}
[trial_34][Epoch 22] val_acc=0.872186 gxi_rev_kl=6.079532 gxi_fwd_kl=2.010543 ig_rev_kl=6.070179 ig_fwd_kl=2.003118 cam_rev_kl=11.066705 cam_fwd_kl=1.599361 test={'autumn': 82.79274326553052, 'rock': 81.97017623136014, 'overall': 82.34126984126983}
[trial_34][Epoch 23] val_acc=0.870594 gxi_rev_kl=6.080006 gxi_fwd_kl=2.010960 ig_rev_kl=6.070565 ig_fwd_kl=2.003385 cam_rev_kl=11.061332 cam_fwd_kl=1.597815 test={'autumn': 82.57284222100056, 'rock': 82.33167645729779, 'overall': 82.44047619047619}
[trial_34][Epoch 24] val_acc=0.872072 gxi_rev_kl=6.079330 gxi_fwd_kl=2.010323 ig_rev_kl=6.070332 ig_fwd_kl=2.003200 cam_rev_kl=11.059404 cam_fwd_kl=1.597149 test={'autumn': 82.62781748213304, 'rock': 82.24130140081337, 'overall': 82.41567460317461}
[trial_34][Epoch 25] val_acc=0.871958 gxi_rev_kl=6.079551 gxi_fwd_kl=2.010582 ig_rev_kl=6.070389 ig_fwd_kl=2.003272 cam_rev_kl=11.057166 cam_fwd_kl=1.596353 test={'autumn': 82.68279274326554, 'rock': 82.37686398553998, 'overall': 82.51488095238095}
[trial_34][Epoch 26] val_acc=0.871503 gxi_rev_kl=6.079747 gxi_fwd_kl=2.010632 ig_rev_kl=6.070605 ig_fwd_kl=2.003380 cam_rev_kl=11.056896 cam_fwd_kl=1.596218 test={'autumn': 83.23254535459043, 'rock': 82.4220515137822, 'overall': 82.78769841269842}
[trial_34][Epoch 27] val_acc=0.871048 gxi_rev_kl=6.079726 gxi_fwd_kl=2.010702 ig_rev_kl=6.070361 ig_fwd_kl=2.003224 cam_rev_kl=11.061232 cam_fwd_kl=1.597504 test={'autumn': 82.51786695986806, 'rock': 82.19611387257117, 'overall': 82.34126984126983}
[trial_34][Epoch 28] val_acc=0.870139 gxi_rev_kl=6.080043 gxi_fwd_kl=2.010963 ig_rev_kl=6.070513 ig_fwd_kl=2.003371 cam_rev_kl=11.059408 cam_fwd_kl=1.597028 test={'autumn': 82.68279274326554, 'rock': 82.10573881608676, 'overall': 82.36607142857143}
[trial_34][Epoch 29] val_acc=0.869229 gxi_rev_kl=6.079656 gxi_fwd_kl=2.010654 ig_rev_kl=6.070070 ig_fwd_kl=2.003085 cam_rev_kl=11.055502 cam_fwd_kl=1.595691 test={'autumn': 82.79274326553052, 'rock': 82.33167645729779, 'overall': 82.53968253968254}

=== trial_34 best summary ===
[trial_34][BEST val_acc] value=0.872527 epoch=20 test={'autumn': 82.57284222100056, 'rock': 82.33167645729779, 'overall': 82.44047619047619}
[trial_34][BEST gxi_rev_kl] value=6.077325 epoch=0 test={'autumn': 78.44969763606377, 'rock': 78.21961138725712, 'overall': 78.3234126984127}
[trial_34][BEST gxi_fwd_kl] value=2.008755 epoch=0 test={'autumn': 78.44969763606377, 'rock': 78.21961138725712, 'overall': 78.3234126984127}
[trial_34][BEST ig_rev_kl] value=6.069864 epoch=0 test={'autumn': 78.44969763606377, 'rock': 78.21961138725712, 'overall': 78.3234126984127}
[trial_34][BEST ig_fwd_kl] value=2.002456 epoch=0 test={'autumn': 78.44969763606377, 'rock': 78.21961138725712, 'overall': 78.3234126984127}
[trial_34][BEST cam_rev_kl] value=11.055502 epoch=29 test={'autumn': 82.79274326553052, 'rock': 82.33167645729779, 'overall': 82.53968253968254}
[trial_34][BEST cam_fwd_kl] value=1.595691 epoch=29 test={'autumn': 82.79274326553052, 'rock': 82.33167645729779, 'overall': 82.53968253968254}
[trial_34] saved: /home/ryreu/guided_cnn/NICO_runs/output/debug_metric_autumn_rock/trial_34_best_metrics.json

=== trial_27 start params={'base_lr': 0.0014669918347337053, 'classifier_lr': 0.004683650727475706, 'lr2_mult': 0.21931433491573496, 'attention_epoch': 12, 'kl_lambda_start': 0.10018509707172872, 'kl_increment': 0.0} ===
[trial_27][Epoch 0] val_acc=0.803844 gxi_rev_kl=6.075611 gxi_fwd_kl=2.007799 ig_rev_kl=6.070469 ig_fwd_kl=2.003094 cam_rev_kl=11.091684 cam_fwd_kl=1.612806 test={'autumn': 75.48103353490929, 'rock': 75.55354722096702, 'overall': 75.52083333333333}
[trial_27][Epoch 1] val_acc=0.806004 gxi_rev_kl=6.078144 gxi_fwd_kl=2.009578 ig_rev_kl=6.071724 ig_fwd_kl=2.004358 cam_rev_kl=11.143856 cam_fwd_kl=1.628964 test={'autumn': 76.30566245189665, 'rock': 73.7008585630366, 'overall': 74.87599206349206}
[trial_27][Epoch 2] val_acc=0.826018 gxi_rev_kl=6.078316 gxi_fwd_kl=2.009402 ig_rev_kl=6.071841 ig_fwd_kl=2.003995 cam_rev_kl=11.134835 cam_fwd_kl=1.624729 test={'autumn': 77.89994502473887, 'rock': 76.32173520108451, 'overall': 77.03373015873017}
[trial_27][Epoch 3] val_acc=0.829315 gxi_rev_kl=6.078740 gxi_fwd_kl=2.009819 ig_rev_kl=6.072113 ig_fwd_kl=2.004255 cam_rev_kl=11.121089 cam_fwd_kl=1.622437 test={'autumn': 76.25068719076415, 'rock': 77.13511070944419, 'overall': 76.73611111111111}
[trial_27][Epoch 4] val_acc=0.837844 gxi_rev_kl=6.079247 gxi_fwd_kl=2.010090 ig_rev_kl=6.072709 ig_fwd_kl=2.004566 cam_rev_kl=11.157866 cam_fwd_kl=1.631668 test={'autumn': 77.78999450247389, 'rock': 77.40623587889742, 'overall': 77.57936507936508}
[trial_27][Epoch 5] val_acc=0.829315 gxi_rev_kl=6.079359 gxi_fwd_kl=2.010342 ig_rev_kl=6.073062 ig_fwd_kl=2.005100 cam_rev_kl=11.153646 cam_fwd_kl=1.633602 test={'autumn': 76.14073666849917, 'rock': 75.82467239042025, 'overall': 75.9672619047619}
[trial_27][Epoch 6] val_acc=0.844667 gxi_rev_kl=6.080045 gxi_fwd_kl=2.010999 ig_rev_kl=6.073169 ig_fwd_kl=2.005219 cam_rev_kl=11.142367 cam_fwd_kl=1.627999 test={'autumn': 78.17482133040131, 'rock': 75.77948486217804, 'overall': 76.86011904761905}
[trial_27][Epoch 7] val_acc=0.846941 gxi_rev_kl=6.082343 gxi_fwd_kl=2.012723 ig_rev_kl=6.074699 ig_fwd_kl=2.006448 cam_rev_kl=11.155665 cam_fwd_kl=1.633135 test={'autumn': 77.95492028587135, 'rock': 76.86398553999096, 'overall': 77.3561507936508}
[trial_27][Epoch 8] val_acc=0.852854 gxi_rev_kl=6.082684 gxi_fwd_kl=2.013155 ig_rev_kl=6.074434 ig_fwd_kl=2.006346 cam_rev_kl=11.158833 cam_fwd_kl=1.633411 test={'autumn': 78.11984606926883, 'rock': 78.35517397198373, 'overall': 78.24900793650794}
[trial_27][Epoch 9] val_acc=0.854560 gxi_rev_kl=6.082542 gxi_fwd_kl=2.013383 ig_rev_kl=6.074293 ig_fwd_kl=2.006261 cam_rev_kl=11.136445 cam_fwd_kl=1.627447 test={'autumn': 78.77954920285872, 'rock': 76.32173520108451, 'overall': 77.43055555555556}
[trial_27][Epoch 10] val_acc=0.855128 gxi_rev_kl=6.082386 gxi_fwd_kl=2.012981 ig_rev_kl=6.073901 ig_fwd_kl=2.005946 cam_rev_kl=11.136232 cam_fwd_kl=1.627760 test={'autumn': 78.99945024738868, 'rock': 76.50248531405332, 'overall': 77.62896825396825}
[trial_27][Epoch 11] val_acc=0.857630 gxi_rev_kl=6.083814 gxi_fwd_kl=2.014244 ig_rev_kl=6.074889 ig_fwd_kl=2.006817 cam_rev_kl=11.165002 cam_fwd_kl=1.636128 test={'autumn': 79.87905442550853, 'rock': 77.58698599186624, 'overall': 78.62103174603175}
[trial_27][Epoch 12] val_acc=0.871048 gxi_rev_kl=6.081332 gxi_fwd_kl=2.012402 ig_rev_kl=6.073218 ig_fwd_kl=2.005621 cam_rev_kl=11.049522 cam_fwd_kl=1.596493 test={'autumn': 80.37383177570094, 'rock': 79.34929959331225, 'overall': 79.81150793650794}
[trial_27][Epoch 13] val_acc=0.871731 gxi_rev_kl=6.080088 gxi_fwd_kl=2.011441 ig_rev_kl=6.071930 ig_fwd_kl=2.004693 cam_rev_kl=10.989683 cam_fwd_kl=1.577894 test={'autumn': 80.92358438702584, 'rock': 79.8011748757343, 'overall': 80.30753968253968}
[trial_27][Epoch 14] val_acc=0.874915 gxi_rev_kl=6.079841 gxi_fwd_kl=2.011523 ig_rev_kl=6.071984 ig_fwd_kl=2.004849 cam_rev_kl=10.960173 cam_fwd_kl=1.568613 test={'autumn': 80.70368334249588, 'rock': 79.71079981924989, 'overall': 80.15873015873017}
[trial_27][Epoch 15] val_acc=0.875256 gxi_rev_kl=6.078763 gxi_fwd_kl=2.010579 ig_rev_kl=6.071531 ig_fwd_kl=2.004440 cam_rev_kl=10.938706 cam_fwd_kl=1.562010 test={'autumn': 80.92358438702584, 'rock': 80.16267510167194, 'overall': 80.50595238095238}
[trial_27][Epoch 16] val_acc=0.878212 gxi_rev_kl=6.079182 gxi_fwd_kl=2.010966 ig_rev_kl=6.071575 ig_fwd_kl=2.004527 cam_rev_kl=10.927465 cam_fwd_kl=1.558671 test={'autumn': 80.81363386476086, 'rock': 79.25892453682783, 'overall': 79.96031746031746}
[trial_27][Epoch 17] val_acc=0.874232 gxi_rev_kl=6.079506 gxi_fwd_kl=2.011303 ig_rev_kl=6.071723 ig_fwd_kl=2.004703 cam_rev_kl=10.918116 cam_fwd_kl=1.556111 test={'autumn': 80.81363386476086, 'rock': 79.57523723452327, 'overall': 80.13392857142857}
[trial_27][Epoch 18] val_acc=0.874232 gxi_rev_kl=6.078545 gxi_fwd_kl=2.010480 ig_rev_kl=6.071334 ig_fwd_kl=2.004373 cam_rev_kl=10.911803 cam_fwd_kl=1.553818 test={'autumn': 80.86860912589334, 'rock': 80.07230004518753, 'overall': 80.43154761904762}
[trial_27][Epoch 19] val_acc=0.875711 gxi_rev_kl=6.079341 gxi_fwd_kl=2.011100 ig_rev_kl=6.071796 ig_fwd_kl=2.004747 cam_rev_kl=10.907993 cam_fwd_kl=1.552741 test={'autumn': 80.97855964815832, 'rock': 80.25305015815636, 'overall': 80.58035714285714}
[trial_27][Epoch 20] val_acc=0.877303 gxi_rev_kl=6.078890 gxi_fwd_kl=2.010899 ig_rev_kl=6.071432 ig_fwd_kl=2.004613 cam_rev_kl=10.900235 cam_fwd_kl=1.550743 test={'autumn': 81.58328752061573, 'rock': 80.20786262991415, 'overall': 80.82837301587301}
[trial_27][Epoch 21] val_acc=0.875370 gxi_rev_kl=6.078918 gxi_fwd_kl=2.010941 ig_rev_kl=6.071353 ig_fwd_kl=2.004515 cam_rev_kl=10.899057 cam_fwd_kl=1.550434 test={'autumn': 81.1434854315558, 'rock': 80.20786262991415, 'overall': 80.62996031746032}
[trial_27][Epoch 22] val_acc=0.877416 gxi_rev_kl=6.078490 gxi_fwd_kl=2.010518 ig_rev_kl=6.071224 ig_fwd_kl=2.004396 cam_rev_kl=10.895592 cam_fwd_kl=1.549448 test={'autumn': 81.03353490929082, 'rock': 80.34342521464076, 'overall': 80.6547619047619}
[trial_27][Epoch 23] val_acc=0.877530 gxi_rev_kl=6.078703 gxi_fwd_kl=2.010803 ig_rev_kl=6.071250 ig_fwd_kl=2.004400 cam_rev_kl=10.891611 cam_fwd_kl=1.548407 test={'autumn': 81.03353490929082, 'rock': 80.07230004518753, 'overall': 80.50595238095238}
[trial_27][Epoch 24] val_acc=0.875824 gxi_rev_kl=6.078226 gxi_fwd_kl=2.010410 ig_rev_kl=6.070896 ig_fwd_kl=2.004188 cam_rev_kl=10.886860 cam_fwd_kl=1.546985 test={'autumn': 80.64870808136338, 'rock': 80.20786262991415, 'overall': 80.40674603174604}
[trial_27][Epoch 25] val_acc=0.877530 gxi_rev_kl=6.078200 gxi_fwd_kl=2.010396 ig_rev_kl=6.070961 ig_fwd_kl=2.004212 cam_rev_kl=10.885390 cam_fwd_kl=1.546294 test={'autumn': 81.19846069268829, 'rock': 80.29823768639855, 'overall': 80.70436507936508}
[trial_27][Epoch 26] val_acc=0.878326 gxi_rev_kl=6.078566 gxi_fwd_kl=2.010718 ig_rev_kl=6.070866 ig_fwd_kl=2.004123 cam_rev_kl=10.883552 cam_fwd_kl=1.545789 test={'autumn': 81.47333699835075, 'rock': 80.25305015815636, 'overall': 80.80357142857143}
[trial_27][Epoch 27] val_acc=0.876052 gxi_rev_kl=6.078291 gxi_fwd_kl=2.010492 ig_rev_kl=6.070926 ig_fwd_kl=2.004148 cam_rev_kl=10.881477 cam_fwd_kl=1.545162 test={'autumn': 81.47333699835075, 'rock': 79.93673746046092, 'overall': 80.62996031746032}
[trial_27][Epoch 28] val_acc=0.878895 gxi_rev_kl=6.078679 gxi_fwd_kl=2.010825 ig_rev_kl=6.070991 ig_fwd_kl=2.004224 cam_rev_kl=10.880373 cam_fwd_kl=1.544727 test={'autumn': 81.36338647608576, 'rock': 80.34342521464076, 'overall': 80.80357142857143}
[trial_27][Epoch 29] val_acc=0.876848 gxi_rev_kl=6.078294 gxi_fwd_kl=2.010505 ig_rev_kl=6.070800 ig_fwd_kl=2.004115 cam_rev_kl=10.877304 cam_fwd_kl=1.544158 test={'autumn': 82.02308960967565, 'rock': 80.07230004518753, 'overall': 80.95238095238095}

=== trial_27 best summary ===
[trial_27][BEST val_acc] value=0.878895 epoch=28 test={'autumn': 81.36338647608576, 'rock': 80.34342521464076, 'overall': 80.80357142857143}
[trial_27][BEST gxi_rev_kl] value=6.075611 epoch=0 test={'autumn': 75.48103353490929, 'rock': 75.55354722096702, 'overall': 75.52083333333333}
[trial_27][BEST gxi_fwd_kl] value=2.007799 epoch=0 test={'autumn': 75.48103353490929, 'rock': 75.55354722096702, 'overall': 75.52083333333333}
[trial_27][BEST ig_rev_kl] value=6.070469 epoch=0 test={'autumn': 75.48103353490929, 'rock': 75.55354722096702, 'overall': 75.52083333333333}
[trial_27][BEST ig_fwd_kl] value=2.003094 epoch=0 test={'autumn': 75.48103353490929, 'rock': 75.55354722096702, 'overall': 75.52083333333333}
[trial_27][BEST cam_rev_kl] value=10.877304 epoch=29 test={'autumn': 82.02308960967565, 'rock': 80.07230004518753, 'overall': 80.95238095238095}
[trial_27][BEST cam_fwd_kl] value=1.544158 epoch=29 test={'autumn': 82.02308960967565, 'rock': 80.07230004518753, 'overall': 80.95238095238095}
[trial_27] saved: /home/ryreu/guided_cnn/NICO_runs/output/debug_metric_autumn_rock/trial_27_best_metrics.json

[TRIAL 0] start params={'base_lr': 0.026180514403573385, 'classifier_lr': 3.8367061829644805e-05, 'lr2_mult': 0.39879089159809655, 'attention_epoch': 3, 'kl_lambda_start': 3.53230477355179, 'kl_increment': 0.0} (elapsed=0.00h)
[TRIAL 0][Epoch 0] val_acc=0.474642 ig_fwd_kl=2.009264 log_optim_num=-20.837833 optim_num=8.917522e-10 test={'autumn': 39.802089059923034, 'rock': 39.22277451423407, 'overall': 39.48412698412698}
[TRIAL 0][Epoch 1] val_acc=0.566636 ig_fwd_kl=2.006799 log_optim_num=-20.636027 optim_num=1.091158e-09 test={'autumn': 51.89664650907092, 'rock': 50.15815634884772, 'overall': 50.942460317460316}
[TRIAL 0][Epoch 2] val_acc=0.634296 ig_fwd_kl=2.007876 log_optim_num=-20.533996 optim_num=1.208366e-09 test={'autumn': 56.89939527212754, 'rock': 53.95390872119295, 'overall': 55.282738095238095}
[TRIAL 0][Epoch 3] val_acc=0.722652 ig_fwd_kl=2.003427 log_optim_num=-20.359101 optim_num=1.439312e-09 test={'autumn': 61.62726772952171, 'rock': 61.680976050610035, 'overall': 61.65674603174603}
[TRIAL 0][Epoch 4] val_acc=0.745849 ig_fwd_kl=2.003208 log_optim_num=-20.325313 optim_num=1.488774e-09 test={'autumn': 69.92853216052777, 'rock': 65.07004066877542, 'overall': 67.26190476190476}
[TRIAL 0][Epoch 5] val_acc=0.731180 ig_fwd_kl=2.004068 log_optim_num=-20.353773 optim_num=1.447001e-09 test={'autumn': 63.71632765255635, 'rock': 65.61229100768188, 'overall': 64.75694444444444}
[TRIAL 0][Epoch 6] val_acc=0.732545 ig_fwd_kl=2.003166 log_optim_num=-20.342889 optim_num=1.462836e-09 test={'autumn': 65.64046179219352, 'rock': 66.56122910076819, 'overall': 66.14583333333333}
[TRIAL 0][Epoch 7] val_acc=0.767341 ig_fwd_kl=2.004426 log_optim_num=-20.309081 optim_num=1.513137e-09 test={'autumn': 69.59868059373282, 'rock': 68.32354270221418, 'overall': 68.89880952380952}
[TRIAL 0][Epoch 8] val_acc=0.773141 ig_fwd_kl=2.004101 log_optim_num=-20.298302 optim_num=1.529535e-09 test={'autumn': 67.3996701484332, 'rock': 68.4591052869408, 'overall': 67.9811507936508}
[TRIAL 0][Epoch 9] val_acc=0.785877 ig_fwd_kl=2.004338 log_optim_num=-20.284339 optim_num=1.551042e-09 test={'autumn': 71.24793842770754, 'rock': 69.85991866244916, 'overall': 70.48611111111111}
[TRIAL 0][Epoch 10] val_acc=0.778485 ig_fwd_kl=2.004106 log_optim_num=-20.291467 optim_num=1.540025e-09 test={'autumn': 69.37877954920286, 'rock': 69.76954360596476, 'overall': 69.59325396825396}
[TRIAL 0][Epoch 11] val_acc=0.795088 ig_fwd_kl=2.004576 log_optim_num=-20.275067 optim_num=1.565490e-09 test={'autumn': 70.03848268279275, 'rock': 70.31179394487121, 'overall': 70.18849206349206}
[TRIAL 0][Epoch 12] val_acc=0.790084 ig_fwd_kl=2.004173 log_optim_num=-20.277342 optim_num=1.561933e-09 test={'autumn': 69.54370533260033, 'rock': 69.3176683235427, 'overall': 69.41964285714286}
[TRIAL 0][Epoch 13] val_acc=0.788833 ig_fwd_kl=2.004435 log_optim_num=-20.281553 optim_num=1.555369e-09 test={'autumn': 70.53326003298515, 'rock': 70.04066877541798, 'overall': 70.26289682539682}
[TRIAL 0][Epoch 14] val_acc=0.808392 ig_fwd_kl=2.003904 log_optim_num=-20.251745 optim_num=1.602429e-09 test={'autumn': 71.85266630016493, 'rock': 70.80885675553547, 'overall': 71.2797619047619}
[TRIAL 0][Epoch 15] val_acc=0.801001 ig_fwd_kl=2.004144 log_optim_num=-20.263331 optim_num=1.583970e-09 test={'autumn': 71.79769103903243, 'rock': 71.80298237686398, 'overall': 71.80059523809524}
[TRIAL 0][Epoch 16] val_acc=0.801228 ig_fwd_kl=2.004785 log_optim_num=-20.269457 optim_num=1.574297e-09 test={'autumn': 69.92853216052777, 'rock': 69.45323090826932, 'overall': 69.66765873015873}
[TRIAL 0][Epoch 17] val_acc=0.818854 ig_fwd_kl=2.004498 log_optim_num=-20.244827 optim_num=1.613554e-09 test={'autumn': 73.7218251786696, 'rock': 71.57704473565296, 'overall': 72.54464285714286}
[TRIAL 0][Epoch 18] val_acc=0.817717 ig_fwd_kl=2.004553 log_optim_num=-20.246766 optim_num=1.610428e-09 test={'autumn': 73.7218251786696, 'rock': 71.57704473565296, 'overall': 72.54464285714286}
[TRIAL 0][Epoch 19] val_acc=0.826814 ig_fwd_kl=2.004472 log_optim_num=-20.234894 optim_num=1.629660e-09 test={'autumn': 73.55689939527213, 'rock': 73.47492092182557, 'overall': 73.51190476190476}
[TRIAL 0][Epoch 20] val_acc=0.824994 ig_fwd_kl=2.004052 log_optim_num=-20.232901 optim_num=1.632911e-09 test={'autumn': 74.21660252886201, 'rock': 73.74604609127881, 'overall': 73.95833333333333}
[TRIAL 0][Epoch 21] val_acc=0.830111 ig_fwd_kl=2.004244 log_optim_num=-20.228639 optim_num=1.639886e-09 test={'autumn': 76.14073666849917, 'rock': 73.56529597830999, 'overall': 74.72718253968254}
[TRIAL 0][Epoch 22] val_acc=0.830794 ig_fwd_kl=2.003954 log_optim_num=-20.224910 optim_num=1.646012e-09 test={'autumn': 75.70093457943925, 'rock': 74.96610935381834, 'overall': 75.29761904761905}
[TRIAL 0][Epoch 23] val_acc=0.832272 ig_fwd_kl=2.004654 log_optim_num=-20.230137 optim_num=1.637432e-09 test={'autumn': 75.48103353490929, 'rock': 74.28829643018527, 'overall': 74.82638888888889}
[TRIAL 0][Epoch 24] val_acc=0.835797 ig_fwd_kl=2.003983 log_optim_num=-20.219202 optim_num=1.655434e-09 test={'autumn': 76.47058823529412, 'rock': 74.83054676909173, 'overall': 75.5704365079365}
[TRIAL 0][Epoch 25] val_acc=0.833523 ig_fwd_kl=2.003770 log_optim_num=-20.219790 optim_num=1.654461e-09 test={'autumn': 76.96536558548654, 'rock': 74.74017171260732, 'overall': 75.74404761904762}
[TRIAL 0][Epoch 26] val_acc=0.836707 ig_fwd_kl=2.003630 log_optim_num=-20.214582 optim_num=1.663100e-09 test={'autumn': 76.36063771302913, 'rock': 75.05648441030276, 'overall': 75.64484126984127}
[TRIAL 0][Epoch 27] val_acc=0.839209 ig_fwd_kl=2.004344 log_optim_num=-20.218734 optim_num=1.656210e-09 test={'autumn': 77.02034084661902, 'rock': 74.87573429733393, 'overall': 75.84325396825396}
[TRIAL 0][Epoch 28] val_acc=0.835570 ig_fwd_kl=2.004107 log_optim_num=-20.220708 optim_num=1.652944e-09 test={'autumn': 77.07531610775152, 'rock': 74.64979665612292, 'overall': 75.74404761904762}
[TRIAL 0][Epoch 29] val_acc=0.838526 ig_fwd_kl=2.004003 log_optim_num=-20.216141 optim_num=1.660510e-09 test={'autumn': 77.18526663001649, 'rock': 75.4631721644826, 'overall': 76.24007936507937}

[TRIAL 1] start params={'base_lr': 0.00011037907255768818, 'classifier_lr': 8.87103098231687e-05, 'lr2_mult': 0.049197296261198876, 'attention_epoch': 7, 'kl_lambda_start': 5.631292451465149, 'kl_increment': 0.0} (elapsed=2.29h)
[TRIAL 1][Epoch 0] val_acc=0.657949 ig_fwd_kl=2.004756 log_optim_num=-20.466185 optim_num=1.293150e-09 test={'autumn': 67.01484332050578, 'rock': 65.65747853592408, 'overall': 66.26984126984127}
[TRIAL 1][Epoch 1] val_acc=0.751649 ig_fwd_kl=2.002693 log_optim_num=-20.312415 optim_num=1.508101e-09 test={'autumn': 74.71137987905442, 'rock': 74.51423407139629, 'overall': 74.60317460317461}
[TRIAL 1][Epoch 2] val_acc=0.785422 ig_fwd_kl=2.002858 log_optim_num=-20.270113 optim_num=1.573264e-09 test={'autumn': 77.62506871907641, 'rock': 76.86398553999096, 'overall': 77.20734126984127}
[TRIAL 1][Epoch 3] val_acc=0.806232 ig_fwd_kl=2.002180 log_optim_num=-20.237186 optim_num=1.625929e-09 test={'autumn': 78.94447498625618, 'rock': 78.44554902846814, 'overall': 78.67063492063492}
[TRIAL 1][Epoch 4] val_acc=0.814305 ig_fwd_kl=2.002497 log_optim_num=-20.230395 optim_num=1.637010e-09 test={'autumn': 79.4392523364486, 'rock': 79.12336195210122, 'overall': 79.26587301587301}
[TRIAL 1][Epoch 5] val_acc=0.828406 ig_fwd_kl=2.003221 log_optim_num=-20.220461 optim_num=1.653351e-09 test={'autumn': 80.26388125343595, 'rock': 79.62042476276548, 'overall': 79.91071428571429}
[TRIAL 1][Epoch 6] val_acc=0.834887 ig_fwd_kl=2.002943 log_optim_num=-20.209891 optim_num=1.670921e-09 test={'autumn': 79.71412864211105, 'rock': 80.11748757342973, 'overall': 79.93551587301587}
[TRIAL 1][Epoch 7] val_acc=0.828747 ig_fwd_kl=1.999620 log_optim_num=-20.184037 optim_num=1.714685e-09 test={'autumn': 79.54920285871358, 'rock': 80.25305015815636, 'overall': 79.93551587301587}
[TRIAL 1][Epoch 8] val_acc=0.823630 ig_fwd_kl=1.998595 log_optim_num=-20.179988 optim_num=1.721642e-09 test={'autumn': 78.99945024738868, 'rock': 79.7559873474921, 'overall': 79.41468253968254}
[TRIAL 1][Epoch 9] val_acc=0.818967 ig_fwd_kl=1.997976 log_optim_num=-20.179467 optim_num=1.722538e-09 test={'autumn': 78.66959868059374, 'rock': 79.21373700858562, 'overall': 78.96825396825396}
[TRIAL 1][Epoch 10] val_acc=0.812258 ig_fwd_kl=1.997552 log_optim_num=-20.183456 optim_num=1.715680e-09 test={'autumn': 78.55964815832876, 'rock': 79.39448712155445, 'overall': 79.01785714285714}
[TRIAL 1][Epoch 11] val_acc=0.813168 ig_fwd_kl=1.997598 log_optim_num=-20.182801 optim_num=1.716804e-09 test={'autumn': 78.11984606926883, 'rock': 79.07817442385901, 'overall': 78.64583333333333}
[TRIAL 1][Epoch 12] val_acc=0.814987 ig_fwd_kl=1.997568 log_optim_num=-20.180258 optim_num=1.721176e-09 test={'autumn': 78.2847718526663, 'rock': 78.44554902846814, 'overall': 78.37301587301587}
[TRIAL 1][Epoch 13] val_acc=0.812031 ig_fwd_kl=1.997145 log_optim_num=-20.179668 optim_num=1.722193e-09 test={'autumn': 78.66959868059374, 'rock': 78.80704925440578, 'overall': 78.74503968253968}
[TRIAL 1][Epoch 14] val_acc=0.814191 ig_fwd_kl=1.997035 log_optim_num=-20.175915 optim_num=1.728668e-09 test={'autumn': 77.68004398020891, 'rock': 78.49073655671035, 'overall': 78.125}
[TRIAL 1][Epoch 15] val_acc=0.813282 ig_fwd_kl=1.997040 log_optim_num=-20.177073 optim_num=1.726667e-09 test={'autumn': 78.11984606926883, 'rock': 78.35517397198373, 'overall': 78.24900793650794}
[TRIAL 1][Epoch 16] val_acc=0.814078 ig_fwd_kl=1.996984 log_optim_num=-20.175541 optim_num=1.729314e-09 test={'autumn': 78.77954920285872, 'rock': 78.53592408495255, 'overall': 78.64583333333333}
[TRIAL 1][Epoch 17] val_acc=0.814305 ig_fwd_kl=1.996926 log_optim_num=-20.174684 optim_num=1.730797e-09 test={'autumn': 78.00989554700385, 'rock': 77.90329868956168, 'overall': 77.95138888888889}
[TRIAL 1][Epoch 18] val_acc=0.814419 ig_fwd_kl=1.996388 log_optim_num=-20.169161 optim_num=1.740382e-09 test={'autumn': 78.11984606926883, 'rock': 78.53592408495255, 'overall': 78.34821428571429}
[TRIAL 1][Epoch 19] val_acc=0.814191 ig_fwd_kl=1.996969 log_optim_num=-20.175250 optim_num=1.729817e-09 test={'autumn': 78.17482133040131, 'rock': 78.44554902846814, 'overall': 78.3234126984127}
[TRIAL 1][Epoch 20] val_acc=0.813168 ig_fwd_kl=1.996827 log_optim_num=-20.175085 optim_num=1.730103e-09 test={'autumn': 78.3397471137988, 'rock': 78.40036150022594, 'overall': 78.37301587301587}
[TRIAL 1][Epoch 21] val_acc=0.812372 ig_fwd_kl=1.996968 log_optim_num=-20.177474 optim_num=1.725974e-09 test={'autumn': 78.11984606926883, 'rock': 78.35517397198373, 'overall': 78.24900793650794}
[TRIAL 1][Epoch 22] val_acc=0.814305 ig_fwd_kl=1.996729 log_optim_num=-20.172711 optim_num=1.734215e-09 test={'autumn': 78.17482133040131, 'rock': 78.0388612742883, 'overall': 78.10019841269842}
[TRIAL 1][Epoch 23] val_acc=0.813168 ig_fwd_kl=1.996823 log_optim_num=-20.175044 optim_num=1.730174e-09 test={'autumn': 78.17482133040131, 'rock': 78.44554902846814, 'overall': 78.3234126984127}
[TRIAL 1][Epoch 24] val_acc=0.813737 ig_fwd_kl=1.996859 log_optim_num=-20.174706 optim_num=1.730759e-09 test={'autumn': 78.22979659153381, 'rock': 78.40036150022594, 'overall': 78.3234126984127}
[TRIAL 1][Epoch 25] val_acc=0.813168 ig_fwd_kl=1.996912 log_optim_num=-20.175941 optim_num=1.728623e-09 test={'autumn': 78.22979659153381, 'rock': 77.99367374604608, 'overall': 78.10019841269842}
[TRIAL 1][Epoch 26] val_acc=0.811690 ig_fwd_kl=1.996812 log_optim_num=-20.176757 optim_num=1.727213e-09 test={'autumn': 78.22979659153381, 'rock': 78.0388612742883, 'overall': 78.125}
[TRIAL 1][Epoch 27] val_acc=0.815101 ig_fwd_kl=1.996681 log_optim_num=-20.171255 optim_num=1.736741e-09 test={'autumn': 78.06487080813633, 'rock': 78.21961138725712, 'overall': 78.14980158730158}
[TRIAL 1][Epoch 28] val_acc=0.813850 ig_fwd_kl=1.996967 log_optim_num=-20.175646 optim_num=1.729133e-09 test={'autumn': 78.77954920285872, 'rock': 78.80704925440578, 'overall': 78.79464285714286}
[TRIAL 1][Epoch 29] val_acc=0.813509 ig_fwd_kl=1.996844 log_optim_num=-20.174836 optim_num=1.730534e-09 test={'autumn': 77.78999450247389, 'rock': 77.94848621780389, 'overall': 77.87698412698413}

[TRIAL 2] start params={'base_lr': 0.002099117333436227, 'classifier_lr': 0.0003312699304112277, 'lr2_mult': 0.016129840526323925, 'attention_epoch': 14, 'kl_lambda_start': 0.42790722869763587, 'kl_increment': 0.0} (elapsed=4.56h)
[TRIAL 2][Epoch 0] val_acc=0.759040 ig_fwd_kl=2.003129 log_optim_num=-20.306991 optim_num=1.516302e-09 test={'autumn': 73.06212204507972, 'rock': 72.93267058291912, 'overall': 72.99107142857143}
[TRIAL 2][Epoch 1] val_acc=0.798044 ig_fwd_kl=2.003366 log_optim_num=-20.259253 optim_num=1.590444e-09 test={'autumn': 75.31610775151182, 'rock': 73.74604609127881, 'overall': 74.45436507936508}
[TRIAL 2][Epoch 2] val_acc=0.826245 ig_fwd_kl=2.004414 log_optim_num=-20.235006 optim_num=1.629479e-09 test={'autumn': 75.97581088510171, 'rock': 75.59873474920921, 'overall': 75.7688492063492}
[TRIAL 2][Epoch 3] val_acc=0.815556 ig_fwd_kl=2.003984 log_optim_num=-20.243727 optim_num=1.615329e-09 test={'autumn': 75.59098405717427, 'rock': 75.14685946678716, 'overall': 75.34722222222223}
[TRIAL 2][Epoch 4] val_acc=0.826700 ig_fwd_kl=2.005907 log_optim_num=-20.249383 optim_num=1.606218e-09 test={'autumn': 74.76635514018692, 'rock': 75.3727971079982, 'overall': 75.09920634920636}
[TRIAL 2][Epoch 5] val_acc=0.827041 ig_fwd_kl=2.005003 log_optim_num=-20.239928 optim_num=1.621477e-09 test={'autumn': 76.80043980208906, 'rock': 75.64392227745142, 'overall': 76.16567460317461}
[TRIAL 2][Epoch 6] val_acc=0.838526 ig_fwd_kl=2.006295 log_optim_num=-20.239061 optim_num=1.622885e-09 test={'autumn': 78.3397471137988, 'rock': 75.4631721644826, 'overall': 76.7609126984127}
[TRIAL 2][Epoch 7] val_acc=0.839663 ig_fwd_kl=2.005832 log_optim_num=-20.233077 optim_num=1.632625e-09 test={'autumn': 76.91039032435404, 'rock': 76.2765476728423, 'overall': 76.5625}
[TRIAL 2][Epoch 8] val_acc=0.845008 ig_fwd_kl=2.006536 log_optim_num=-20.233768 optim_num=1.631498e-09 test={'autumn': 77.40516767454645, 'rock': 75.86985991866246, 'overall': 76.5625}
[TRIAL 2][Epoch 9] val_acc=0.843075 ig_fwd_kl=2.006732 log_optim_num=-20.238018 optim_num=1.624577e-09 test={'autumn': 78.99945024738868, 'rock': 77.08992318120198, 'overall': 77.95138888888889}
[TRIAL 2][Epoch 10] val_acc=0.850921 ig_fwd_kl=2.007410 log_optim_num=-20.235535 optim_num=1.628617e-09 test={'autumn': 78.3397471137988, 'rock': 76.63804789877993, 'overall': 77.40575396825396}
[TRIAL 2][Epoch 11] val_acc=0.851149 ig_fwd_kl=2.007411 log_optim_num=-20.235276 optim_num=1.629038e-09 test={'autumn': 79.10940076965366, 'rock': 76.77361048350656, 'overall': 77.82738095238095}
[TRIAL 2][Epoch 12] val_acc=0.858654 ig_fwd_kl=2.008245 log_optim_num=-20.234841 optim_num=1.629747e-09 test={'autumn': 79.27432655305113, 'rock': 78.49073655671035, 'overall': 78.84424603174604}
[TRIAL 2][Epoch 13] val_acc=0.861383 ig_fwd_kl=2.008272 log_optim_num=-20.231939 optim_num=1.634483e-09 test={'autumn': 80.75865860362836, 'rock': 79.16854948034343, 'overall': 79.8859126984127}
[TRIAL 2][Epoch 14] val_acc=0.866955 ig_fwd_kl=2.007036 log_optim_num=-20.213133 optim_num=1.665512e-09 test={'autumn': 80.86860912589334, 'rock': 79.16854948034343, 'overall': 79.93551587301587}
[TRIAL 2][Epoch 15] val_acc=0.865704 ig_fwd_kl=2.005988 log_optim_num=-20.204096 optim_num=1.680632e-09 test={'autumn': 81.52831225948323, 'rock': 79.53004970628106, 'overall': 80.43154761904762}
[TRIAL 2][Epoch 16] val_acc=0.869115 ig_fwd_kl=2.005646 log_optim_num=-20.196739 optim_num=1.693041e-09 test={'autumn': 81.47333699835075, 'rock': 79.71079981924989, 'overall': 80.50595238095238}
[TRIAL 2][Epoch 17] val_acc=0.869911 ig_fwd_kl=2.004961 log_optim_num=-20.188974 optim_num=1.706239e-09 test={'autumn': 81.47333699835075, 'rock': 79.62042476276548, 'overall': 80.4563492063492}
[TRIAL 2][Epoch 18] val_acc=0.870821 ig_fwd_kl=2.004851 log_optim_num=-20.186827 optim_num=1.709907e-09 test={'autumn': 81.63826278174821, 'rock': 79.66561229100768, 'overall': 80.55555555555556}
[TRIAL 2][Epoch 19] val_acc=0.869115 ig_fwd_kl=2.004482 log_optim_num=-20.185095 optim_num=1.712871e-09 test={'autumn': 81.52831225948323, 'rock': 80.02711251694532, 'overall': 80.70436507936508}
[TRIAL 2][Epoch 20] val_acc=0.868888 ig_fwd_kl=2.004068 log_optim_num=-20.181218 optim_num=1.719525e-09 test={'autumn': 81.47333699835075, 'rock': 79.48486217803887, 'overall': 80.38194444444444}
[TRIAL 2][Epoch 21] val_acc=0.869002 ig_fwd_kl=2.003912 log_optim_num=-20.179528 optim_num=1.722433e-09 test={'autumn': 82.02308960967565, 'rock': 79.53004970628106, 'overall': 80.6547619047619}
[TRIAL 2][Epoch 22] val_acc=0.869684 ig_fwd_kl=2.003681 log_optim_num=-20.176439 optim_num=1.727763e-09 test={'autumn': 81.36338647608576, 'rock': 79.71079981924989, 'overall': 80.4563492063492}
[TRIAL 2][Epoch 23] val_acc=0.868888 ig_fwd_kl=2.003763 log_optim_num=-20.178174 optim_num=1.724767e-09 test={'autumn': 81.25343595382078, 'rock': 79.8011748757343, 'overall': 80.4563492063492}
[TRIAL 2][Epoch 24] val_acc=0.870025 ig_fwd_kl=2.003556 log_optim_num=-20.174789 optim_num=1.730615e-09 test={'autumn': 81.52831225948323, 'rock': 79.89154993221871, 'overall': 80.62996031746032}
[TRIAL 2][Epoch 25] val_acc=0.869798 ig_fwd_kl=2.003506 log_optim_num=-20.174555 optim_num=1.731020e-09 test={'autumn': 81.25343595382078, 'rock': 79.39448712155445, 'overall': 80.23313492063492}
[TRIAL 2][Epoch 26] val_acc=0.869343 ig_fwd_kl=2.003219 log_optim_num=-20.172208 optim_num=1.735087e-09 test={'autumn': 81.30841121495327, 'rock': 79.98192498870311, 'overall': 80.58035714285714}
[TRIAL 2][Epoch 27] val_acc=0.870252 ig_fwd_kl=2.003338 log_optim_num=-20.172353 optim_num=1.734836e-09 test={'autumn': 81.25343595382078, 'rock': 80.02711251694532, 'overall': 80.58035714285714}
[TRIAL 2][Epoch 28] val_acc=0.869229 ig_fwd_kl=2.003477 log_optim_num=-20.174923 optim_num=1.730384e-09 test={'autumn': 81.41836173721825, 'rock': 79.39448712155445, 'overall': 80.30753968253968}
[TRIAL 2][Epoch 29] val_acc=0.870366 ig_fwd_kl=2.003098 log_optim_num=-20.169819 optim_num=1.739238e-09 test={'autumn': 81.47333699835075, 'rock': 80.07230004518753, 'overall': 80.70436507936508}

[TRIAL 3] start params={'base_lr': 3.1397055166901684e-05, 'classifier_lr': 2.4396604480953775e-05, 'lr2_mult': 0.02624041429398575, 'attention_epoch': 4, 'kl_lambda_start': 0.5916055464009692, 'kl_increment': 0.0} (elapsed=6.82h)
[TRIAL 3][Epoch 0] val_acc=0.273027 ig_fwd_kl=2.019101 log_optim_num=-21.489193 optim_num=4.649027e-10 test={'autumn': 27.102803738317757, 'rock': 24.401265250790782, 'overall': 25.620039682539684}
[TRIAL 3][Epoch 1] val_acc=0.509438 ig_fwd_kl=2.010026 log_optim_num=-20.774703 optim_num=9.498635e-10 test={'autumn': 53.1061022539857, 'rock': 49.66109353818346, 'overall': 51.21527777777778}
[TRIAL 3][Epoch 2] val_acc=0.623152 ig_fwd_kl=2.006354 log_optim_num=-20.536500 optim_num=1.205345e-09 test={'autumn': 62.2869708631116, 'rock': 61.22910076818798, 'overall': 61.70634920634921}
[TRIAL 3][Epoch 3] val_acc=0.678986 ig_fwd_kl=2.003580 log_optim_num=-20.422952 optim_num=1.350282e-09 test={'autumn': 67.83947223749313, 'rock': 66.87754179846362, 'overall': 67.31150793650794}
[TRIAL 3][Epoch 4] val_acc=0.684558 ig_fwd_kl=2.003848 log_optim_num=-20.417461 optim_num=1.357717e-09 test={'autumn': 67.4546454095657, 'rock': 67.37460460912789, 'overall': 67.41071428571429}
[TRIAL 3][Epoch 5] val_acc=0.680691 ig_fwd_kl=2.003492 log_optim_num=-20.419571 optim_num=1.354855e-09 test={'autumn': 68.16932380428807, 'rock': 68.14279258924537, 'overall': 68.1547619047619}
[TRIAL 3][Epoch 6] val_acc=0.685467 ig_fwd_kl=2.003957 log_optim_num=-20.417220 optim_num=1.358045e-09 test={'autumn': 67.50962067069818, 'rock': 67.82647989154994, 'overall': 67.68353174603175}
[TRIAL 3][Epoch 7] val_acc=0.686718 ig_fwd_kl=2.003826 log_optim_num=-20.414090 optim_num=1.362302e-09 test={'autumn': 68.49917537108301, 'rock': 67.5553547220967, 'overall': 67.9811507936508}
[TRIAL 3][Epoch 8] val_acc=0.687628 ig_fwd_kl=2.003465 log_optim_num=-20.409159 optim_num=1.369036e-09 test={'autumn': 68.27927432655305, 'rock': 68.00723000451875, 'overall': 68.12996031746032}
[TRIAL 3][Epoch 9] val_acc=0.688651 ig_fwd_kl=2.003082 log_optim_num=-20.403836 optim_num=1.376343e-09 test={'autumn': 68.44420010995053, 'rock': 68.27835517397199, 'overall': 68.35317460317461}
[TRIAL 3][Epoch 10] val_acc=0.690584 ig_fwd_kl=2.003321 log_optim_num=-20.403423 optim_num=1.376912e-09 test={'autumn': 68.49917537108301, 'rock': 68.18798011748757, 'overall': 68.32837301587301}
[TRIAL 3][Epoch 11] val_acc=0.688083 ig_fwd_kl=2.003417 log_optim_num=-20.408020 optim_num=1.370596e-09 test={'autumn': 68.38922484881803, 'rock': 68.27835517397199, 'overall': 68.32837301587301}
[TRIAL 3][Epoch 12] val_acc=0.689106 ig_fwd_kl=2.002945 log_optim_num=-20.401811 optim_num=1.379133e-09 test={'autumn': 67.67454645409566, 'rock': 68.36873023045639, 'overall': 68.05555555555556}
[TRIAL 3][Epoch 13] val_acc=0.692176 ig_fwd_kl=2.002938 log_optim_num=-20.397290 optim_num=1.385382e-09 test={'autumn': 68.16932380428807, 'rock': 67.82647989154994, 'overall': 67.9811507936508}
[TRIAL 3][Epoch 14] val_acc=0.696156 ig_fwd_kl=2.003173 log_optim_num=-20.393913 optim_num=1.390068e-09 test={'autumn': 68.93897746014294, 'rock': 68.36873023045639, 'overall': 68.62599206349206}
[TRIAL 3][Epoch 15] val_acc=0.695360 ig_fwd_kl=2.002577 log_optim_num=-20.389094 optim_num=1.396783e-09 test={'autumn': 68.99395272127542, 'rock': 69.49841843651153, 'overall': 69.27083333333333}
[TRIAL 3][Epoch 16] val_acc=0.692745 ig_fwd_kl=2.002991 log_optim_num=-20.397004 optim_num=1.385778e-09 test={'autumn': 68.66410115448048, 'rock': 68.09760506100316, 'overall': 68.35317460317461}
[TRIAL 3][Epoch 17] val_acc=0.695588 ig_fwd_kl=2.003548 log_optim_num=-20.398480 optim_num=1.383734e-09 test={'autumn': 69.37877954920286, 'rock': 68.82060551287844, 'overall': 69.07242063492063}
[TRIAL 3][Epoch 18] val_acc=0.693314 ig_fwd_kl=2.003012 log_optim_num=-20.396397 optim_num=1.386619e-09 test={'autumn': 68.77405167674546, 'rock': 68.4139177586986, 'overall': 68.57638888888889}
[TRIAL 3][Epoch 19] val_acc=0.697180 ig_fwd_kl=2.002917 log_optim_num=-20.389884 optim_num=1.395680e-09 test={'autumn': 68.71907641561297, 'rock': 69.22729326705829, 'overall': 68.99801587301587}
[TRIAL 3][Epoch 20] val_acc=0.694678 ig_fwd_kl=2.002647 log_optim_num=-20.390777 optim_num=1.394434e-09 test={'autumn': 69.32380428807036, 'rock': 68.36873023045639, 'overall': 68.79960317460318}
[TRIAL 3][Epoch 21] val_acc=0.696725 ig_fwd_kl=2.003060 log_optim_num=-20.391960 optim_num=1.392785e-09 test={'autumn': 69.1588785046729, 'rock': 68.95616809760506, 'overall': 69.04761904761905}
[TRIAL 3][Epoch 22] val_acc=0.695133 ig_fwd_kl=2.002832 log_optim_num=-20.391974 optim_num=1.392766e-09 test={'autumn': 69.1039032435404, 'rock': 69.2724807953005, 'overall': 69.19642857142857}
[TRIAL 3][Epoch 23] val_acc=0.695929 ig_fwd_kl=2.002529 log_optim_num=-20.387799 optim_num=1.398593e-09 test={'autumn': 69.43375481033534, 'rock': 68.86579304112065, 'overall': 69.12202380952381}
[TRIAL 3][Epoch 24] val_acc=0.694792 ig_fwd_kl=2.002508 log_optim_num=-20.389228 optim_num=1.396596e-09 test={'autumn': 69.1588785046729, 'rock': 69.3176683235427, 'overall': 69.24603174603175}
[TRIAL 3][Epoch 25] val_acc=0.693427 ig_fwd_kl=2.002876 log_optim_num=-20.394865 optim_num=1.388746e-09 test={'autumn': 69.1588785046729, 'rock': 69.2724807953005, 'overall': 69.22123015873017}
[TRIAL 3][Epoch 26] val_acc=0.699113 ig_fwd_kl=2.002610 log_optim_num=-20.384043 optim_num=1.403856e-09 test={'autumn': 68.88400219901044, 'rock': 69.2724807953005, 'overall': 69.09722222222223}
[TRIAL 3][Epoch 27] val_acc=0.696498 ig_fwd_kl=2.003390 log_optim_num=-20.395594 optim_num=1.387733e-09 test={'autumn': 69.92853216052777, 'rock': 69.99548124717577, 'overall': 69.96527777777777}
[TRIAL 3][Epoch 28] val_acc=0.696952 ig_fwd_kl=2.003195 log_optim_num=-20.392987 optim_num=1.391356e-09 test={'autumn': 68.93897746014294, 'rock': 69.3176683235427, 'overall': 69.14682539682539}
[TRIAL 3][Epoch 29] val_acc=0.698886 ig_fwd_kl=2.002719 log_optim_num=-20.385458 optim_num=1.401872e-09 test={'autumn': 69.21385376580538, 'rock': 69.45323090826932, 'overall': 69.3452380952381}

[TRIAL 4] start params={'base_lr': 0.012604333729773142, 'classifier_lr': 0.0001451269843507438, 'lr2_mult': 0.0016948405087746305, 'attention_epoch': 23, 'kl_lambda_start': 0.496881985696999, 'kl_increment': 0.0} (elapsed=9.08h)
[TRIAL 4][Epoch 0] val_acc=0.585513 ig_fwd_kl=2.006817 log_optim_num=-20.603440 optim_num=1.127301e-09 test={'autumn': 54.15063221550302, 'rock': 49.75146859466787, 'overall': 51.736111111111114}
[TRIAL 4][Epoch 1] val_acc=0.638390 ig_fwd_kl=2.006807 log_optim_num=-20.516874 optim_num=1.229235e-09 test={'autumn': 54.480483782297966, 'rock': 54.54134658834162, 'overall': 54.513888888888886}
[TRIAL 4][Epoch 2] val_acc=0.685354 ig_fwd_kl=2.006176 log_optim_num=-20.439584 optim_num=1.328010e-09 test={'autumn': 61.73721825178669, 'rock': 60.28016267510167, 'overall': 60.9375}
[TRIAL 4][Epoch 3] val_acc=0.719468 ig_fwd_kl=2.006739 log_optim_num=-20.396636 optim_num=1.386288e-09 test={'autumn': 63.27652556349643, 'rock': 64.979665612291, 'overall': 64.21130952380952}
[TRIAL 4][Epoch 4] val_acc=0.736411 ig_fwd_kl=2.008012 log_optim_num=-20.386083 optim_num=1.400995e-09 test={'autumn': 66.35514018691589, 'rock': 64.43741527338454, 'overall': 65.30257936507937}
[TRIAL 4][Epoch 5] val_acc=0.728906 ig_fwd_kl=2.008621 log_optim_num=-20.402422 optim_num=1.378290e-09 test={'autumn': 64.54095656954371, 'rock': 63.89516493447808, 'overall': 64.18650793650794}
[TRIAL 4][Epoch 6] val_acc=0.747214 ig_fwd_kl=2.007983 log_optim_num=-20.371234 optim_num=1.421953e-09 test={'autumn': 65.58548653106102, 'rock': 66.19972887483054, 'overall': 65.92261904761905}
[TRIAL 4][Epoch 7] val_acc=0.758813 ig_fwd_kl=2.008292 log_optim_num=-20.358917 optim_num=1.439576e-09 test={'autumn': 65.695437053326, 'rock': 66.78716674197922, 'overall': 66.29464285714286}
[TRIAL 4][Epoch 8] val_acc=0.771662 ig_fwd_kl=2.007841 log_optim_num=-20.337620 optim_num=1.470563e-09 test={'autumn': 67.28971962616822, 'rock': 67.5101671938545, 'overall': 67.41071428571429}
[TRIAL 4][Epoch 9] val_acc=0.697180 ig_fwd_kl=2.007283 log_optim_num=-20.433546 optim_num=1.336054e-09 test={'autumn': 65.97031335898845, 'rock': 63.669227293267056, 'overall': 64.70734126984127}
[TRIAL 4][Epoch 10] val_acc=0.798044 ig_fwd_kl=2.008070 log_optim_num=-20.306289 optim_num=1.517367e-09 test={'autumn': 71.13798790544256, 'rock': 71.26073203795752, 'overall': 71.20535714285714}
[TRIAL 4][Epoch 11] val_acc=0.805322 ig_fwd_kl=2.006772 log_optim_num=-20.284232 optim_num=1.551207e-09 test={'autumn': 71.02803738317758, 'rock': 70.49254405784004, 'overall': 70.73412698412699}
[TRIAL 4][Epoch 12] val_acc=0.811121 ig_fwd_kl=2.007911 log_optim_num=-20.288449 optim_num=1.544680e-09 test={'autumn': 72.18251786695987, 'rock': 70.76366922729326, 'overall': 71.40376984126983}
[TRIAL 4][Epoch 13] val_acc=0.809870 ig_fwd_kl=2.008122 log_optim_num=-20.292103 optim_num=1.539046e-09 test={'autumn': 73.33699835074216, 'rock': 72.07410754631722, 'overall': 72.6438492063492}
[TRIAL 4][Epoch 14] val_acc=0.823630 ig_fwd_kl=2.009201 log_optim_num=-20.286048 optim_num=1.548393e-09 test={'autumn': 73.99670148433205, 'rock': 72.30004518752824, 'overall': 73.06547619047619}
[TRIAL 4][Epoch 15] val_acc=0.825108 ig_fwd_kl=2.008474 log_optim_num=-20.276985 optim_num=1.562490e-09 test={'autumn': 73.22704782847718, 'rock': 72.52598282873927, 'overall': 72.8422619047619}
[TRIAL 4][Epoch 16] val_acc=0.835797 ig_fwd_kl=2.008947 log_optim_num=-20.268838 optim_num=1.575272e-09 test={'autumn': 75.4260582737768, 'rock': 73.47492092182557, 'overall': 74.35515873015873}
[TRIAL 4][Epoch 17] val_acc=0.833068 ig_fwd_kl=2.008846 log_optim_num=-20.271102 optim_num=1.571709e-09 test={'autumn': 73.6668499175371, 'rock': 73.6556710347944, 'overall': 73.66071428571429}
[TRIAL 4][Epoch 18] val_acc=0.836366 ig_fwd_kl=2.008463 log_optim_num=-20.263323 optim_num=1.583984e-09 test={'autumn': 74.65640461792194, 'rock': 74.28829643018527, 'overall': 74.45436507936508}
[TRIAL 4][Epoch 19] val_acc=0.841142 ig_fwd_kl=2.009114 log_optim_num=-20.264139 optim_num=1.582692e-09 test={'autumn': 75.53600879604178, 'rock': 75.19204699502937, 'overall': 75.34722222222223}
[TRIAL 4][Epoch 20] val_acc=0.842620 ig_fwd_kl=2.008896 log_optim_num=-20.260195 optim_num=1.588947e-09 test={'autumn': 75.59098405717427, 'rock': 75.3727971079982, 'overall': 75.47123015873017}
[TRIAL 4][Epoch 21] val_acc=0.843530 ig_fwd_kl=2.009283 log_optim_num=-20.262988 optim_num=1.584514e-09 test={'autumn': 75.59098405717427, 'rock': 75.50835969272481, 'overall': 75.54563492063492}
[TRIAL 4][Epoch 22] val_acc=0.844326 ig_fwd_kl=2.008688 log_optim_num=-20.256095 optim_num=1.595473e-09 test={'autumn': 76.80043980208906, 'rock': 75.3727971079982, 'overall': 76.01686507936508}
[TRIAL 4][Epoch 23] val_acc=0.845804 ig_fwd_kl=2.008449 log_optim_num=-20.251962 optim_num=1.602082e-09 test={'autumn': 77.130291368884, 'rock': 75.41798463624039, 'overall': 76.19047619047619}
[TRIAL 4][Epoch 24] val_acc=0.843757 ig_fwd_kl=2.008309 log_optim_num=-20.252981 optim_num=1.600450e-09 test={'autumn': 76.91039032435404, 'rock': 75.14685946678716, 'overall': 75.94246031746032}
[TRIAL 4][Epoch 25] val_acc=0.845235 ig_fwd_kl=2.008023 log_optim_num=-20.248368 optim_num=1.607851e-09 test={'autumn': 77.35019241341396, 'rock': 75.4631721644826, 'overall': 76.31448412698413}
[TRIAL 4][Epoch 26] val_acc=0.844098 ig_fwd_kl=2.007957 log_optim_num=-20.249058 optim_num=1.606740e-09 test={'autumn': 77.07531610775152, 'rock': 75.23723452327158, 'overall': 76.06646825396825}
[TRIAL 4][Epoch 27] val_acc=0.842165 ig_fwd_kl=2.007750 log_optim_num=-20.249275 optim_num=1.606393e-09 test={'autumn': 76.96536558548654, 'rock': 74.87573429733393, 'overall': 75.81845238095238}
[TRIAL 4][Epoch 28] val_acc=0.845577 ig_fwd_kl=2.007603 log_optim_num=-20.243765 optim_num=1.615268e-09 test={'autumn': 77.51511819681143, 'rock': 75.28242205151378, 'overall': 76.28968253968254}
[TRIAL 4][Epoch 29] val_acc=0.844098 ig_fwd_kl=2.007375 log_optim_num=-20.243236 optim_num=1.616123e-09 test={'autumn': 77.29521715228147, 'rock': 75.14685946678716, 'overall': 76.11607142857143}

[TRIAL 5] start params={'base_lr': 0.028601066882373506, 'classifier_lr': 0.009747977558416472, 'lr2_mult': 0.01013885306160916, 'attention_epoch': 15, 'kl_lambda_start': 0.3991652042588259, 'kl_increment': 0.0} (elapsed=11.34h)
[TRIAL 5][Epoch 0] val_acc=0.383671 ig_fwd_kl=2.009929 log_optim_num=-21.057257 optim_num=7.160604e-10 test={'autumn': 32.325453545904345, 'rock': 31.089019430637144, 'overall': 31.646825396825395}
[TRIAL 5][Epoch 1] val_acc=0.536502 ig_fwd_kl=2.008782 log_optim_num=-20.710506 optim_num=1.012842e-09 test={'autumn': 46.838922484881806, 'rock': 43.78671486669679, 'overall': 45.163690476190474}
[TRIAL 5][Epoch 2] val_acc=0.590744 ig_fwd_kl=2.006720 log_optim_num=-20.593574 optim_num=1.138478e-09 test={'autumn': 51.84167124793843, 'rock': 51.42340713962946, 'overall': 51.61210317460318}
[TRIAL 5][Epoch 3] val_acc=0.618262 ig_fwd_kl=2.007570 log_optim_num=-20.556540 optim_num=1.181430e-09 test={'autumn': 53.271028037383175, 'rock': 53.637596023497515, 'overall': 53.47222222222222}
[TRIAL 5][Epoch 4] val_acc=0.659541 ig_fwd_kl=2.008344 log_optim_num=-20.499655 optim_num=1.250585e-09 test={'autumn': 55.90984057174271, 'rock': 55.2643470402169, 'overall': 55.55555555555556}
[TRIAL 5][Epoch 5] val_acc=0.686036 ig_fwd_kl=2.007759 log_optim_num=-20.454412 optim_num=1.308464e-09 test={'autumn': 59.153380978559646, 'rock': 59.46678716674198, 'overall': 59.32539682539682}
[TRIAL 5][Epoch 6] val_acc=0.700591 ig_fwd_kl=2.007328 log_optim_num=-20.429106 optim_num=1.341999e-09 test={'autumn': 60.74766355140187, 'rock': 60.00903750564844, 'overall': 60.342261904761905}
[TRIAL 5][Epoch 7] val_acc=0.714237 ig_fwd_kl=2.006691 log_optim_num=-20.403450 optim_num=1.376874e-09 test={'autumn': 62.8367234744365, 'rock': 62.3136014460009, 'overall': 62.54960317460318}
[TRIAL 5][Epoch 8] val_acc=0.722311 ig_fwd_kl=2.008434 log_optim_num=-20.409644 optim_num=1.368372e-09 test={'autumn': 62.23199560197911, 'rock': 60.551287844554906, 'overall': 61.30952380952381}
[TRIAL 5][Epoch 9] val_acc=0.748465 ig_fwd_kl=2.007655 log_optim_num=-20.366284 optim_num=1.429010e-09 test={'autumn': 67.12479384277076, 'rock': 65.07004066877542, 'overall': 65.99702380952381}
[TRIAL 5][Epoch 10] val_acc=0.754947 ig_fwd_kl=2.007568 log_optim_num=-20.356793 optim_num=1.442637e-09 test={'autumn': 65.80538757559098, 'rock': 64.75372797107998, 'overall': 65.22817460317461}
[TRIAL 5][Epoch 11] val_acc=0.749602 ig_fwd_kl=2.008173 log_optim_num=-20.369944 optim_num=1.423789e-09 test={'autumn': 66.24518966465091, 'rock': 64.39222774514234, 'overall': 65.22817460317461}
[TRIAL 5][Epoch 12] val_acc=0.766659 ig_fwd_kl=2.007607 log_optim_num=-20.341779 optim_num=1.464461e-09 test={'autumn': 67.56459593183068, 'rock': 65.29597830998644, 'overall': 66.31944444444444}
[TRIAL 5][Epoch 13] val_acc=0.778713 ig_fwd_kl=2.008062 log_optim_num=-20.330731 optim_num=1.480730e-09 test={'autumn': 68.82902693787796, 'rock': 66.83235427022142, 'overall': 67.73313492063492}
[TRIAL 5][Epoch 14] val_acc=0.784967 ig_fwd_kl=2.007488 log_optim_num=-20.316994 optim_num=1.501211e-09 test={'autumn': 70.47828477185267, 'rock': 66.56122910076819, 'overall': 68.32837301587301}
[TRIAL 5][Epoch 15] val_acc=0.803275 ig_fwd_kl=2.007329 log_optim_num=-20.292349 optim_num=1.538667e-09 test={'autumn': 72.01759208356239, 'rock': 68.86579304112065, 'overall': 70.28769841269842}
[TRIAL 5][Epoch 16] val_acc=0.805663 ig_fwd_kl=2.007149 log_optim_num=-20.287579 optim_num=1.546025e-09 test={'autumn': 72.18251786695987, 'rock': 69.18210573881609, 'overall': 70.53571428571429}
[TRIAL 5][Epoch 17] val_acc=0.809188 ig_fwd_kl=2.006940 log_optim_num=-20.281125 optim_num=1.556035e-09 test={'autumn': 71.96261682242991, 'rock': 69.58879349299593, 'overall': 70.65972222222223}
[TRIAL 5][Epoch 18] val_acc=0.809416 ig_fwd_kl=2.006703 log_optim_num=-20.278478 optim_num=1.560160e-09 test={'autumn': 71.74271577789995, 'rock': 70.35698147311342, 'overall': 70.98214285714286}
[TRIAL 5][Epoch 19] val_acc=0.810780 ig_fwd_kl=2.006749 log_optim_num=-20.277252 optim_num=1.562073e-09 test={'autumn': 72.40241891148983, 'rock': 69.72435607772255, 'overall': 70.93253968253968}
[TRIAL 5][Epoch 20] val_acc=0.811917 ig_fwd_kl=2.006688 log_optim_num=-20.275239 optim_num=1.565221e-09 test={'autumn': 72.40241891148983, 'rock': 69.90510619069137, 'overall': 71.03174603174604}
[TRIAL 5][Epoch 21] val_acc=0.812372 ig_fwd_kl=2.006586 log_optim_num=-20.273657 optim_num=1.567699e-09 test={'autumn': 72.6223199560198, 'rock': 69.58879349299593, 'overall': 70.95734126984127}
[TRIAL 5][Epoch 22] val_acc=0.815215 ig_fwd_kl=2.006481 log_optim_num=-20.269110 optim_num=1.574844e-09 test={'autumn': 72.6223199560198, 'rock': 69.45323090826932, 'overall': 70.8829365079365}
[TRIAL 5][Epoch 23] val_acc=0.815101 ig_fwd_kl=2.006388 log_optim_num=-20.268323 optim_num=1.576083e-09 test={'autumn': 72.51236943375481, 'rock': 69.95029371893358, 'overall': 71.1061507936508}
[TRIAL 5][Epoch 24] val_acc=0.816011 ig_fwd_kl=2.006360 log_optim_num=-20.266929 optim_num=1.578282e-09 test={'autumn': 73.22704782847718, 'rock': 70.13104383190239, 'overall': 71.52777777777777}
[TRIAL 5][Epoch 25] val_acc=0.814987 ig_fwd_kl=2.006407 log_optim_num=-20.268653 optim_num=1.575563e-09 test={'autumn': 72.73227047828478, 'rock': 70.13104383190239, 'overall': 71.3045634920635}
[TRIAL 5][Epoch 26] val_acc=0.816238 ig_fwd_kl=2.006309 log_optim_num=-20.266135 optim_num=1.579536e-09 test={'autumn': 72.73227047828478, 'rock': 70.04066877541798, 'overall': 71.25496031746032}
[TRIAL 5][Epoch 27] val_acc=0.817262 ig_fwd_kl=2.006120 log_optim_num=-20.262993 optim_num=1.584506e-09 test={'autumn': 72.6223199560198, 'rock': 70.1762313601446, 'overall': 71.2797619047619}
[TRIAL 5][Epoch 28] val_acc=0.815897 ig_fwd_kl=2.006320 log_optim_num=-20.266663 optim_num=1.578703e-09 test={'autumn': 73.28202308960968, 'rock': 70.0858563036602, 'overall': 71.52777777777777}
[TRIAL 5][Epoch 29] val_acc=0.819081 ig_fwd_kl=2.006310 log_optim_num=-20.262674 optim_num=1.585011e-09 test={'autumn': 73.33699835074216, 'rock': 70.26660641662902, 'overall': 71.65178571428571}

[TRIAL 6] start params={'base_lr': 0.00040389728039267914, 'classifier_lr': 0.0024187657165976495, 'lr2_mult': 0.7678921526655245, 'attention_epoch': 13, 'kl_lambda_start': 1.531746276522771, 'kl_increment': 0.0} (elapsed=13.60h)
[TRIAL 6][Epoch 0] val_acc=0.824312 ig_fwd_kl=2.002519 log_optim_num=-20.218398 optim_num=1.656766e-09 test={'autumn': 80.81363386476086, 'rock': 80.02711251694532, 'overall': 80.38194444444444}
[TRIAL 6][Epoch 1] val_acc=0.840573 ig_fwd_kl=2.003480 log_optim_num=-20.208473 optim_num=1.673292e-09 test={'autumn': 80.15393073117097, 'rock': 80.52417532760958, 'overall': 80.35714285714286}
[TRIAL 6][Epoch 2] val_acc=0.851717 ig_fwd_kl=2.004435 log_optim_num=-20.204847 optim_num=1.679371e-09 test={'autumn': 80.5937328202309, 'rock': 81.38273836421148, 'overall': 81.02678571428571}
[TRIAL 6][Epoch 3] val_acc=0.857744 ig_fwd_kl=2.003339 log_optim_num=-20.186845 optim_num=1.709877e-09 test={'autumn': 81.36338647608576, 'rock': 81.51830094893809, 'overall': 81.4484126984127}
[TRIAL 6][Epoch 4] val_acc=0.859791 ig_fwd_kl=2.004427 log_optim_num=-20.195337 optim_num=1.695417e-09 test={'autumn': 81.80318856514569, 'rock': 80.93086308178943, 'overall': 81.32440476190476}
[TRIAL 6][Epoch 5] val_acc=0.862747 ig_fwd_kl=2.004764 log_optim_num=-20.195277 optim_num=1.695518e-09 test={'autumn': 82.07806487080813, 'rock': 81.51830094893809, 'overall': 81.77083333333333}
[TRIAL 6][Epoch 6] val_acc=0.867751 ig_fwd_kl=2.005040 log_optim_num=-20.192246 optim_num=1.700666e-09 test={'autumn': 82.07806487080813, 'rock': 81.51830094893809, 'overall': 81.77083333333333}
[TRIAL 6][Epoch 7] val_acc=0.866159 ig_fwd_kl=2.005785 log_optim_num=-20.201535 optim_num=1.684942e-09 test={'autumn': 81.80318856514569, 'rock': 80.7049254405784, 'overall': 81.20039682539682}
[TRIAL 6][Epoch 8] val_acc=0.867296 ig_fwd_kl=2.005918 log_optim_num=-20.201553 optim_num=1.684911e-09 test={'autumn': 81.74821330401319, 'rock': 80.11748757342973, 'overall': 80.85317460317461}
[TRIAL 6][Epoch 9] val_acc=0.868888 ig_fwd_kl=2.006028 log_optim_num=-20.200818 optim_num=1.686149e-09 test={'autumn': 82.79274326553052, 'rock': 81.20198825124265, 'overall': 81.91964285714286}
[TRIAL 6][Epoch 10] val_acc=0.872868 ig_fwd_kl=2.005652 log_optim_num=-20.192490 optim_num=1.700252e-09 test={'autumn': 81.52831225948323, 'rock': 80.84048802530502, 'overall': 81.15079365079364}
[TRIAL 6][Epoch 11] val_acc=0.873436 ig_fwd_kl=2.006107 log_optim_num=-20.196387 optim_num=1.693637e-09 test={'autumn': 82.847718526663, 'rock': 80.93086308178943, 'overall': 81.79563492063492}
[TRIAL 6][Epoch 12] val_acc=0.873095 ig_fwd_kl=2.006454 log_optim_num=-20.200246 optim_num=1.687114e-09 test={'autumn': 82.2979659153381, 'rock': 81.33755083596927, 'overall': 81.77083333333333}
[TRIAL 6][Epoch 13] val_acc=0.869115 ig_fwd_kl=1.999646 log_optim_num=-20.136737 optim_num=1.797737e-09 test={'autumn': 81.91313908741067, 'rock': 80.93086308178943, 'overall': 81.37400793650794}
[TRIAL 6][Epoch 14] val_acc=0.867637 ig_fwd_kl=1.999225 log_optim_num=-20.134233 optim_num=1.802245e-09 test={'autumn': 81.63826278174821, 'rock': 80.7501129688206, 'overall': 81.15079365079364}
[TRIAL 6][Epoch 15] val_acc=0.864453 ig_fwd_kl=1.999334 log_optim_num=-20.138997 optim_num=1.793679e-09 test={'autumn': 81.58328752061573, 'rock': 79.7559873474921, 'overall': 80.58035714285714}
[TRIAL 6][Epoch 16] val_acc=0.867182 ig_fwd_kl=1.999441 log_optim_num=-20.136916 optim_num=1.797416e-09 test={'autumn': 81.69323804288071, 'rock': 80.16267510167194, 'overall': 80.85317460317461}
[TRIAL 6][Epoch 17] val_acc=0.864112 ig_fwd_kl=1.999376 log_optim_num=-20.139815 optim_num=1.792213e-09 test={'autumn': 82.2979659153381, 'rock': 80.97605061003163, 'overall': 81.57242063492063}
[TRIAL 6][Epoch 18] val_acc=0.865704 ig_fwd_kl=1.999126 log_optim_num=-20.135475 optim_num=1.800007e-09 test={'autumn': 81.80318856514569, 'rock': 80.02711251694532, 'overall': 80.82837301587301}
[TRIAL 6][Epoch 19] val_acc=0.867296 ig_fwd_kl=1.999168 log_optim_num=-20.134060 optim_num=1.802556e-09 test={'autumn': 81.96811434854315, 'rock': 80.11748757342973, 'overall': 80.95238095238095}
[TRIAL 6][Epoch 20] val_acc=0.867068 ig_fwd_kl=1.999280 log_optim_num=-20.135438 optim_num=1.800075e-09 test={'autumn': 81.69323804288071, 'rock': 79.43967464979666, 'overall': 80.4563492063492}
[TRIAL 6][Epoch 21] val_acc=0.865818 ig_fwd_kl=1.999097 log_optim_num=-20.135048 optim_num=1.800776e-09 test={'autumn': 81.63826278174821, 'rock': 79.93673746046092, 'overall': 80.70436507936508}
[TRIAL 6][Epoch 22] val_acc=0.864453 ig_fwd_kl=1.998874 log_optim_num=-20.134402 optim_num=1.801940e-09 test={'autumn': 81.0885101704233, 'rock': 80.11748757342973, 'overall': 80.55555555555556}
[TRIAL 6][Epoch 23] val_acc=0.865135 ig_fwd_kl=1.999194 log_optim_num=-20.136808 optim_num=1.797610e-09 test={'autumn': 81.41836173721825, 'rock': 80.7501129688206, 'overall': 81.0515873015873}
[TRIAL 6][Epoch 24] val_acc=0.864226 ig_fwd_kl=1.999341 log_optim_num=-20.139336 optim_num=1.793071e-09 test={'autumn': 82.07806487080813, 'rock': 80.93086308178943, 'overall': 81.4484126984127}
[TRIAL 6][Epoch 25] val_acc=0.864453 ig_fwd_kl=1.999348 log_optim_num=-20.139136 optim_num=1.793429e-09 test={'autumn': 81.41836173721825, 'rock': 80.29823768639855, 'overall': 80.80357142857143}
[TRIAL 6][Epoch 26] val_acc=0.866614 ig_fwd_kl=1.999310 log_optim_num=-20.136267 optim_num=1.798583e-09 test={'autumn': 82.13304013194063, 'rock': 80.93086308178943, 'overall': 81.47321428571429}
[TRIAL 6][Epoch 27] val_acc=0.865818 ig_fwd_kl=1.999149 log_optim_num=-20.135568 optim_num=1.799841e-09 test={'autumn': 81.63826278174821, 'rock': 80.79530049706281, 'overall': 81.17559523809524}
[TRIAL 6][Epoch 28] val_acc=0.869115 ig_fwd_kl=1.999447 log_optim_num=-20.134747 optim_num=1.801318e-09 test={'autumn': 80.70368334249588, 'rock': 80.79530049706281, 'overall': 80.75396825396825}
[TRIAL 6][Epoch 29] val_acc=0.867410 ig_fwd_kl=1.999512 log_optim_num=-20.137361 optim_num=1.796616e-09 test={'autumn': 82.02308960967565, 'rock': 80.29823768639855, 'overall': 81.07638888888889}

[TRIAL 7] start params={'base_lr': 0.0023344489676130568, 'classifier_lr': 0.02385217666087124, 'lr2_mult': 0.04657798137280556, 'attention_epoch': 16, 'kl_lambda_start': 1.5716066815832643, 'kl_increment': 0.0} (elapsed=15.86h)
[TRIAL 7][Epoch 0] val_acc=0.720719 ig_fwd_kl=2.002597 log_optim_num=-20.353479 optim_num=1.447427e-09 test={'autumn': 67.72952171522815, 'rock': 68.18798011748757, 'overall': 67.9811507936508}
[TRIAL 7][Epoch 1] val_acc=0.779281 ig_fwd_kl=2.003322 log_optim_num=-20.282605 optim_num=1.553734e-09 test={'autumn': 74.38152831225948, 'rock': 69.90510619069137, 'overall': 71.92460317460318}
[TRIAL 7][Epoch 2] val_acc=0.791904 ig_fwd_kl=2.005076 log_optim_num=-20.284079 optim_num=1.551446e-09 test={'autumn': 73.55689939527213, 'rock': 70.94441934026209, 'overall': 72.12301587301587}
[TRIAL 7][Epoch 3] val_acc=0.800205 ig_fwd_kl=2.004589 log_optim_num=-20.268775 optim_num=1.575371e-09 test={'autumn': 74.49147883452446, 'rock': 71.71260732037958, 'overall': 72.96626984126983}
[TRIAL 7][Epoch 4] val_acc=0.800318 ig_fwd_kl=2.005483 log_optim_num=-20.277580 optim_num=1.561561e-09 test={'autumn': 74.65640461792194, 'rock': 71.8933574333484, 'overall': 73.13988095238095}
[TRIAL 7][Epoch 5] val_acc=0.812713 ig_fwd_kl=2.006107 log_optim_num=-20.268449 optim_num=1.575885e-09 test={'autumn': 74.49147883452446, 'rock': 70.49254405784004, 'overall': 72.29662698412699}
[TRIAL 7][Epoch 6] val_acc=0.809074 ig_fwd_kl=2.004618 log_optim_num=-20.258044 optim_num=1.592368e-09 test={'autumn': 73.22704782847718, 'rock': 73.15860822413013, 'overall': 73.18948412698413}
[TRIAL 7][Epoch 7] val_acc=0.814987 ig_fwd_kl=2.005348 log_optim_num=-20.258066 optim_num=1.592333e-09 test={'autumn': 74.49147883452446, 'rock': 72.43560777225485, 'overall': 73.36309523809524}
[TRIAL 7][Epoch 8] val_acc=0.814078 ig_fwd_kl=2.006021 log_optim_num=-20.265910 optim_num=1.579891e-09 test={'autumn': 74.38152831225948, 'rock': 73.61048350655219, 'overall': 73.95833333333333}
[TRIAL 7][Epoch 9] val_acc=0.833750 ig_fwd_kl=2.006575 log_optim_num=-20.247568 optim_num=1.609137e-09 test={'autumn': 75.4260582737768, 'rock': 74.78535924084953, 'overall': 75.07440476190476}
[TRIAL 7][Epoch 10] val_acc=0.831817 ig_fwd_kl=2.005606 log_optim_num=-20.240205 optim_num=1.621029e-09 test={'autumn': 77.24024189114898, 'rock': 75.59873474920921, 'overall': 76.33928571428571}
[TRIAL 7][Epoch 11] val_acc=0.833068 ig_fwd_kl=2.006616 log_optim_num=-20.248795 optim_num=1.607163e-09 test={'autumn': 76.25068719076415, 'rock': 74.46904654315409, 'overall': 75.27281746031746}
[TRIAL 7][Epoch 12] val_acc=0.842847 ig_fwd_kl=2.006452 log_optim_num=-20.235492 optim_num=1.628687e-09 test={'autumn': 76.52556349642661, 'rock': 74.10754631721645, 'overall': 75.1984126984127}
[TRIAL 7][Epoch 13] val_acc=0.842620 ig_fwd_kl=2.007192 log_optim_num=-20.243156 optim_num=1.616253e-09 test={'autumn': 77.07531610775152, 'rock': 75.82467239042025, 'overall': 76.38888888888889}
[TRIAL 7][Epoch 14] val_acc=0.845235 ig_fwd_kl=2.006593 log_optim_num=-20.234074 optim_num=1.630998e-09 test={'autumn': 76.96536558548654, 'rock': 75.28242205151378, 'overall': 76.04166666666667}
[TRIAL 7][Epoch 15] val_acc=0.849215 ig_fwd_kl=2.007252 log_optim_num=-20.235963 optim_num=1.627920e-09 test={'autumn': 79.10940076965366, 'rock': 77.08992318120198, 'overall': 78.00099206349206}
[TRIAL 7][Epoch 16] val_acc=0.852741 ig_fwd_kl=2.004474 log_optim_num=-20.204038 optim_num=1.680729e-09 test={'autumn': 79.05442550852116, 'rock': 76.77361048350656, 'overall': 77.80257936507937}
[TRIAL 7][Epoch 17] val_acc=0.854560 ig_fwd_kl=2.004108 log_optim_num=-20.198244 optim_num=1.690496e-09 test={'autumn': 80.09895547003848, 'rock': 76.2765476728423, 'overall': 78.00099206349206}
[TRIAL 7][Epoch 18] val_acc=0.857062 ig_fwd_kl=2.003844 log_optim_num=-20.192686 optim_num=1.699918e-09 test={'autumn': 79.54920285871358, 'rock': 76.95436059647537, 'overall': 78.125}
[TRIAL 7][Epoch 19] val_acc=0.856152 ig_fwd_kl=2.003691 log_optim_num=-20.192219 optim_num=1.700712e-09 test={'autumn': 79.16437603078614, 'rock': 76.81879801174875, 'overall': 77.87698412698413}
[TRIAL 7][Epoch 20] val_acc=0.857971 ig_fwd_kl=2.003672 log_optim_num=-20.189901 optim_num=1.704658e-09 test={'autumn': 79.3842770753161, 'rock': 77.49661093538184, 'overall': 78.34821428571429}
[TRIAL 7][Epoch 21] val_acc=0.859450 ig_fwd_kl=2.003649 log_optim_num=-20.187954 optim_num=1.707981e-09 test={'autumn': 79.21935129191864, 'rock': 77.85811116131947, 'overall': 78.47222222222223}
[TRIAL 7][Epoch 22] val_acc=0.858995 ig_fwd_kl=2.003433 log_optim_num=-20.186327 optim_num=1.710762e-09 test={'autumn': 79.93402968664101, 'rock': 76.95436059647537, 'overall': 78.29861111111111}
[TRIAL 7][Epoch 23] val_acc=0.859108 ig_fwd_kl=2.003748 log_optim_num=-20.189343 optim_num=1.705610e-09 test={'autumn': 79.3842770753161, 'rock': 76.99954812471758, 'overall': 78.07539682539682}
[TRIAL 7][Epoch 24] val_acc=0.861042 ig_fwd_kl=2.003385 log_optim_num=-20.183466 optim_num=1.715664e-09 test={'autumn': 79.54920285871358, 'rock': 77.40623587889742, 'overall': 78.37301587301587}
[TRIAL 7][Epoch 25] val_acc=0.860132 ig_fwd_kl=2.003240 log_optim_num=-20.183067 optim_num=1.716348e-09 test={'autumn': 79.76910390324355, 'rock': 77.40623587889742, 'overall': 78.47222222222223}
[TRIAL 7][Epoch 26] val_acc=0.861496 ig_fwd_kl=2.003122 log_optim_num=-20.180305 optim_num=1.721095e-09 test={'autumn': 79.82407916437603, 'rock': 77.67736104835066, 'overall': 78.64583333333333}
[TRIAL 7][Epoch 27] val_acc=0.860132 ig_fwd_kl=2.002988 log_optim_num=-20.180554 optim_num=1.720666e-09 test={'autumn': 79.98900494777351, 'rock': 77.49661093538184, 'overall': 78.62103174603175}
[TRIAL 7][Epoch 28] val_acc=0.860587 ig_fwd_kl=2.003026 log_optim_num=-20.180404 optim_num=1.720925e-09 test={'autumn': 79.98900494777351, 'rock': 77.85811116131947, 'overall': 78.81944444444444}
[TRIAL 7][Epoch 29] val_acc=0.860246 ig_fwd_kl=2.002914 log_optim_num=-20.179680 optim_num=1.722172e-09 test={'autumn': 79.49422759758109, 'rock': 77.67736104835066, 'overall': 78.49702380952381}

[TRIAL 8] start params={'base_lr': 3.2706679800450356e-05, 'classifier_lr': 0.02883632724127818, 'lr2_mult': 0.08193803520498749, 'attention_epoch': 28, 'kl_lambda_start': 1.6717027496491597, 'kl_increment': 0.0} (elapsed=18.13h)
[TRIAL 8][Epoch 0] val_acc=0.783261 ig_fwd_kl=2.005307 log_optim_num=-20.297355 optim_num=1.530985e-09 test={'autumn': 76.96536558548654, 'rock': 75.10167193854497, 'overall': 75.94246031746032}
[TRIAL 8][Epoch 1] val_acc=0.778713 ig_fwd_kl=2.003814 log_optim_num=-20.288252 optim_num=1.544984e-09 test={'autumn': 75.92083562396921, 'rock': 72.43560777225485, 'overall': 74.0079365079365}
[TRIAL 8][Epoch 2] val_acc=0.790653 ig_fwd_kl=2.003519 log_optim_num=-20.270083 optim_num=1.573312e-09 test={'autumn': 76.69048927982408, 'rock': 75.32760957975599, 'overall': 75.94246031746032}
[TRIAL 8][Epoch 3] val_acc=0.814078 ig_fwd_kl=2.003038 log_optim_num=-20.236077 optim_num=1.627734e-09 test={'autumn': 76.6355140186916, 'rock': 76.63804789877993, 'overall': 76.63690476190476}
[TRIAL 8][Epoch 4] val_acc=0.812258 ig_fwd_kl=2.003272 log_optim_num=-20.240657 optim_num=1.620296e-09 test={'autumn': 78.2847718526663, 'rock': 77.81292363307728, 'overall': 78.02579365079364}
[TRIAL 8][Epoch 5] val_acc=0.812827 ig_fwd_kl=2.003490 log_optim_num=-20.242133 optim_num=1.617906e-09 test={'autumn': 76.85541506322156, 'rock': 76.99954812471758, 'overall': 76.93452380952381}
[TRIAL 8][Epoch 6] val_acc=0.824085 ig_fwd_kl=2.003054 log_optim_num=-20.224026 optim_num=1.647469e-09 test={'autumn': 77.02034084661902, 'rock': 76.2765476728423, 'overall': 76.61210317460318}
[TRIAL 8][Epoch 7] val_acc=0.830680 ig_fwd_kl=2.003097 log_optim_num=-20.216480 optim_num=1.659948e-09 test={'autumn': 78.3397471137988, 'rock': 77.08992318120198, 'overall': 77.65376984126983}
[TRIAL 8][Epoch 8] val_acc=0.833068 ig_fwd_kl=2.003573 log_optim_num=-20.218369 optim_num=1.656815e-09 test={'autumn': 79.87905442550853, 'rock': 77.67736104835066, 'overall': 78.67063492063492}
[TRIAL 8][Epoch 9] val_acc=0.845690 ig_fwd_kl=2.003499 log_optim_num=-20.202590 optim_num=1.683165e-09 test={'autumn': 78.8894997251237, 'rock': 78.58111161319476, 'overall': 78.7202380952381}
[TRIAL 8][Epoch 10] val_acc=0.837844 ig_fwd_kl=2.003533 log_optim_num=-20.212253 optim_num=1.666979e-09 test={'autumn': 78.06487080813633, 'rock': 77.31586082241301, 'overall': 77.65376984126983}
[TRIAL 8][Epoch 11] val_acc=0.843075 ig_fwd_kl=2.003664 log_optim_num=-20.207342 optim_num=1.675185e-09 test={'autumn': 79.27432655305113, 'rock': 77.90329868956168, 'overall': 78.52182539682539}
[TRIAL 8][Epoch 12] val_acc=0.839095 ig_fwd_kl=2.003352 log_optim_num=-20.208947 optim_num=1.672498e-09 test={'autumn': 79.21935129191864, 'rock': 78.53592408495255, 'overall': 78.84424603174604}
[TRIAL 8][Epoch 13] val_acc=0.851262 ig_fwd_kl=2.003677 log_optim_num=-20.197805 optim_num=1.691239e-09 test={'autumn': 79.27432655305113, 'rock': 77.81292363307728, 'overall': 78.47222222222223}
[TRIAL 8][Epoch 14] val_acc=0.843757 ig_fwd_kl=2.003854 log_optim_num=-20.208435 optim_num=1.673355e-09 test={'autumn': 79.3842770753161, 'rock': 77.08992318120198, 'overall': 78.125}
[TRIAL 8][Epoch 15] val_acc=0.851262 ig_fwd_kl=2.003975 log_optim_num=-20.200787 optim_num=1.686203e-09 test={'autumn': 79.98900494777351, 'rock': 77.72254857659286, 'overall': 78.74503968253968}
[TRIAL 8][Epoch 16] val_acc=0.855015 ig_fwd_kl=2.004113 log_optim_num=-20.197770 optim_num=1.691297e-09 test={'autumn': 80.26388125343595, 'rock': 78.71667419792138, 'overall': 79.41468253968254}
[TRIAL 8][Epoch 17] val_acc=0.854674 ig_fwd_kl=2.004447 log_optim_num=-20.201501 optim_num=1.684999e-09 test={'autumn': 79.60417811984607, 'rock': 78.89742431089019, 'overall': 79.21626984126983}
[TRIAL 8][Epoch 18] val_acc=0.852627 ig_fwd_kl=2.003844 log_optim_num=-20.197877 optim_num=1.691116e-09 test={'autumn': 79.3842770753161, 'rock': 78.58111161319476, 'overall': 78.94345238095238}
[TRIAL 8][Epoch 19] val_acc=0.853537 ig_fwd_kl=2.004238 log_optim_num=-20.200749 optim_num=1.686267e-09 test={'autumn': 79.60417811984607, 'rock': 78.76186172616357, 'overall': 79.14186507936508}
[TRIAL 8][Epoch 20] val_acc=0.856720 ig_fwd_kl=2.003875 log_optim_num=-20.193396 optim_num=1.698711e-09 test={'autumn': 79.4392523364486, 'rock': 79.03298689561682, 'overall': 79.21626984126983}
[TRIAL 8][Epoch 21] val_acc=0.858199 ig_fwd_kl=2.004215 log_optim_num=-20.195067 optim_num=1.695874e-09 test={'autumn': 79.93402968664101, 'rock': 78.71667419792138, 'overall': 79.26587301587301}
[TRIAL 8][Epoch 22] val_acc=0.857289 ig_fwd_kl=2.003889 log_optim_num=-20.192873 optim_num=1.699600e-09 test={'autumn': 79.71412864211105, 'rock': 79.12336195210122, 'overall': 79.38988095238095}
[TRIAL 8][Epoch 23] val_acc=0.857289 ig_fwd_kl=2.004143 log_optim_num=-20.195407 optim_num=1.695298e-09 test={'autumn': 80.70368334249588, 'rock': 79.53004970628106, 'overall': 80.05952380952381}
[TRIAL 8][Epoch 24] val_acc=0.857971 ig_fwd_kl=2.004267 log_optim_num=-20.195851 optim_num=1.694546e-09 test={'autumn': 80.31885651456844, 'rock': 79.48486217803887, 'overall': 79.86111111111111}
[TRIAL 8][Epoch 25] val_acc=0.861951 ig_fwd_kl=2.003799 log_optim_num=-20.186551 optim_num=1.710378e-09 test={'autumn': 80.70368334249588, 'rock': 79.53004970628106, 'overall': 80.05952380952381}
[TRIAL 8][Epoch 26] val_acc=0.858085 ig_fwd_kl=2.004242 log_optim_num=-20.195470 optim_num=1.695192e-09 test={'autumn': 80.09895547003848, 'rock': 79.57523723452327, 'overall': 79.81150793650794}
[TRIAL 8][Epoch 27] val_acc=0.860700 ig_fwd_kl=2.004166 log_optim_num=-20.191673 optim_num=1.701641e-09 test={'autumn': 80.5387575590984, 'rock': 79.39448712155445, 'overall': 79.91071428571429}
[TRIAL 8][Epoch 28] val_acc=0.856948 ig_fwd_kl=2.004071 log_optim_num=-20.195092 optim_num=1.695833e-09 test={'autumn': 80.5937328202309, 'rock': 79.62042476276548, 'overall': 80.05952380952381}
[TRIAL 8][Epoch 29] val_acc=0.858881 ig_fwd_kl=2.002879 log_optim_num=-20.180917 optim_num=1.720042e-09 test={'autumn': 80.70368334249588, 'rock': 79.12336195210122, 'overall': 79.83630952380952}

[TRIAL 9] start params={'base_lr': 0.00027521719156201003, 'classifier_lr': 0.0018812243781599414, 'lr2_mult': 0.10368022057691184, 'attention_epoch': 23, 'kl_lambda_start': 0.6880643186411968, 'kl_increment': 0.0} (elapsed=20.39h)
[TRIAL 9][Epoch 0] val_acc=0.819195 ig_fwd_kl=2.002850 log_optim_num=-20.227933 optim_num=1.641044e-09 test={'autumn': 79.98900494777351, 'rock': 79.21373700858562, 'overall': 79.56349206349206}
[TRIAL 9][Epoch 1] val_acc=0.836593 ig_fwd_kl=2.002823 log_optim_num=-20.206646 optim_num=1.676351e-09 test={'autumn': 81.1434854315558, 'rock': 80.88567555354722, 'overall': 81.00198412698413}
[TRIAL 9][Epoch 2] val_acc=0.848419 ig_fwd_kl=2.003901 log_optim_num=-20.203389 optim_num=1.681821e-09 test={'autumn': 82.57284222100056, 'rock': 81.87980117487574, 'overall': 82.19246031746032}
[TRIAL 9][Epoch 3] val_acc=0.855128 ig_fwd_kl=2.003872 log_optim_num=-20.195228 optim_num=1.695602e-09 test={'autumn': 81.69323804288071, 'rock': 80.7501129688206, 'overall': 81.17559523809524}
[TRIAL 9][Epoch 4] val_acc=0.861951 ig_fwd_kl=2.003838 log_optim_num=-20.186932 optim_num=1.709728e-09 test={'autumn': 80.97855964815832, 'rock': 80.7501129688206, 'overall': 80.85317460317461}
[TRIAL 9][Epoch 5] val_acc=0.865249 ig_fwd_kl=2.003361 log_optim_num=-20.178352 optim_num=1.724459e-09 test={'autumn': 81.63826278174821, 'rock': 81.51830094893809, 'overall': 81.57242063492063}
[TRIAL 9][Epoch 6] val_acc=0.868319 ig_fwd_kl=2.004029 log_optim_num=-20.181483 optim_num=1.719068e-09 test={'autumn': 82.79274326553052, 'rock': 82.19611387257117, 'overall': 82.46527777777777}
[TRIAL 9][Epoch 7] val_acc=0.869684 ig_fwd_kl=2.005059 log_optim_num=-20.190213 optim_num=1.704127e-09 test={'autumn': 81.96811434854315, 'rock': 81.02123813827383, 'overall': 81.4484126984127}
[TRIAL 9][Epoch 8] val_acc=0.869343 ig_fwd_kl=2.004275 log_optim_num=-20.182771 optim_num=1.716856e-09 test={'autumn': 82.847718526663, 'rock': 81.78942611839132, 'overall': 82.26686507936508}
[TRIAL 9][Epoch 9] val_acc=0.870366 ig_fwd_kl=2.004643 log_optim_num=-20.185270 optim_num=1.712571e-09 test={'autumn': 82.2979659153381, 'rock': 80.97605061003163, 'overall': 81.57242063492063}
[TRIAL 9][Epoch 10] val_acc=0.873550 ig_fwd_kl=2.004394 log_optim_num=-20.179128 optim_num=1.723122e-09 test={'autumn': 82.68279274326554, 'rock': 81.74423859014912, 'overall': 82.16765873015873}
[TRIAL 9][Epoch 11] val_acc=0.874687 ig_fwd_kl=2.004737 log_optim_num=-20.181261 optim_num=1.719450e-09 test={'autumn': 82.07806487080813, 'rock': 81.60867600542251, 'overall': 81.8204365079365}
[TRIAL 9][Epoch 12] val_acc=0.871731 ig_fwd_kl=2.005140 log_optim_num=-20.188679 optim_num=1.706743e-09 test={'autumn': 81.96811434854315, 'rock': 80.7501129688206, 'overall': 81.29960317460318}
[TRIAL 9][Epoch 13] val_acc=0.872527 ig_fwd_kl=2.004750 log_optim_num=-20.183865 optim_num=1.714978e-09 test={'autumn': 81.85816382627817, 'rock': 81.5634884771803, 'overall': 81.69642857142857}
[TRIAL 9][Epoch 14] val_acc=0.871162 ig_fwd_kl=2.005684 log_optim_num=-20.194769 optim_num=1.696381e-09 test={'autumn': 82.02308960967565, 'rock': 81.69905106190691, 'overall': 81.8452380952381}
[TRIAL 9][Epoch 15] val_acc=0.874915 ig_fwd_kl=2.005355 log_optim_num=-20.187183 optim_num=1.709299e-09 test={'autumn': 81.96811434854315, 'rock': 80.84048802530502, 'overall': 81.34920634920636}
[TRIAL 9][Epoch 16] val_acc=0.873095 ig_fwd_kl=2.005271 log_optim_num=-20.188425 optim_num=1.707176e-09 test={'autumn': 82.2979659153381, 'rock': 81.83461364663353, 'overall': 82.0436507936508}
[TRIAL 9][Epoch 17] val_acc=0.875938 ig_fwd_kl=2.005704 log_optim_num=-20.189501 optim_num=1.705340e-09 test={'autumn': 82.79274326553052, 'rock': 81.38273836421148, 'overall': 82.0188492063492}
[TRIAL 9][Epoch 18] val_acc=0.874346 ig_fwd_kl=2.005896 log_optim_num=-20.193237 optim_num=1.698982e-09 test={'autumn': 82.24299065420561, 'rock': 81.92498870311793, 'overall': 82.06845238095238}
[TRIAL 9][Epoch 19] val_acc=0.875483 ig_fwd_kl=2.005696 log_optim_num=-20.189935 optim_num=1.704601e-09 test={'autumn': 81.91313908741067, 'rock': 81.5634884771803, 'overall': 81.72123015873017}
[TRIAL 9][Epoch 20] val_acc=0.875142 ig_fwd_kl=2.005402 log_optim_num=-20.187393 optim_num=1.708939e-09 test={'autumn': 82.3529411764706, 'rock': 81.97017623136014, 'overall': 82.14285714285714}
[TRIAL 9][Epoch 21] val_acc=0.872640 ig_fwd_kl=2.005792 log_optim_num=-20.194155 optim_num=1.697422e-09 test={'autumn': 82.51786695986806, 'rock': 81.29236330772707, 'overall': 81.8452380952381}
[TRIAL 9] budget reached after epoch 21 (elapsed=22.06h).


"""
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paste ALL your log lines for trial_27, trial_34, trial_36 (and any others) here.


BETA_MIN = 1
BETA_MAX = 50

pattern = re.compile(
    r"\[(?:trial_(?P<trial_u>\d+)|trial\s+(?P<trial_s>\d+))\]"
    r"\[Epoch\s+(?P<epoch>\d+)\]\s+val_acc=(?P<val_acc>[0-9.]+).*?"
    r"ig_fwd_kl=(?P<ig_fwd_kl>[0-9.]+).*?"
    r"test=\{.*?'overall':\s*(?P<test_overall>[0-9.]+)\}",
    re.IGNORECASE
)

rows = []
for m in pattern.finditer(LOG_TEXT):
    trial = int(m.group("trial_u") or m.group("trial_s"))
    epoch = int(m.group("epoch"))
    val_acc = float(m.group("val_acc"))
    ig_fwd_kl = float(m.group("ig_fwd_kl"))
    test_overall = float(m.group("test_overall"))
    rows.append((trial, epoch, val_acc, ig_fwd_kl, test_overall))

df = pd.DataFrame(
    rows,
    columns=["trial", "epoch", "val_acc", "ig_fwd_kl", "test_overall"]
)

if df.empty:
    raise ValueError(
        "No epochs parsed. Check LOG_TEXT and that test dict has single quotes like 'overall'."
    )

# Parse diagnostics: helps confirm we consumed all pasted epoch lines.
epoch_like_lines = [
    ln
    for ln in LOG_TEXT.splitlines()
    if (
        ("[trial_" in ln.lower() and "[epoch" in ln.lower())
        or re.search(r"\[trial\s+\d+\]\[epoch\s+\d+\]", ln, re.IGNORECASE)
    )
]
print(f"Parsed epoch rows: {len(df)} / epoch-like lines found: {len(epoch_like_lines)}")
for trial_id, grp in df.groupby("trial"):
    epochs = sorted(grp["epoch"].tolist())
    print(
        f"trial_{int(trial_id)}: {len(epochs)} epochs "
        f"(min={epochs[0]}, max={epochs[-1]})"
    )

# Correlations pooled over all (trial, epoch) points
r_val = float(np.corrcoef(df["val_acc"].to_numpy(), df["test_overall"].to_numpy())[0, 1])
beta_results = []
for beta in range(BETA_MIN, BETA_MAX + 1):
    optim_num = df["val_acc"].to_numpy() * np.exp(-beta * df["ig_fwd_kl"].to_numpy())
    if np.std(optim_num) == 0 or np.std(df["test_overall"].to_numpy()) == 0:
        r_optim = float("nan")
    else:
        r_optim = float(np.corrcoef(optim_num, df["test_overall"].to_numpy())[0, 1])
    beta_results.append((beta, r_optim))

beta_df = pd.DataFrame(beta_results, columns=["beta", "r_optim"])
best_row = beta_df.loc[beta_df["r_optim"].idxmax()]
best_beta = int(best_row["beta"])
best_r = float(best_row["r_optim"])

df["optim_num"] = df["val_acc"] * np.exp(-best_beta * df["ig_fwd_kl"])

print(f"Pooled Pearson r(val_acc, test_overall) = {r_val:.6f}")
print(f"Best beta in [{BETA_MIN}, {BETA_MAX}] by pooled Pearson:")
print(f"  beta={best_beta} -> r(optim_num, test_overall)={best_r:.6f}")
print("\nTop beta values:")
print(beta_df.sort_values("r_optim", ascending=False).head(15).to_string(index=False))

# ---- Plot 1: val_acc vs test_overall ----
plt.figure(figsize=(7, 5))
plt.scatter(df["val_acc"], df["test_overall"], alpha=0.8)
plt.xlabel("val_acc")
plt.ylabel("test_overall (%)")
plt.title(f"val_acc vs test_overall | Pearson r={r_val:.4f}")
plt.tight_layout()
plt.show()

# ---- Plot 2: optim_num vs test_overall ----
plt.figure(figsize=(7, 5))
plt.scatter(df["optim_num"], df["test_overall"], alpha=0.8)

# optim_num gets tiny at beta=10, so log x-scale helps
plt.xscale("log")

plt.xlabel(f"optim_num = val_acc * exp(-{best_beta:g} * ig_fwd_kl)  (log x-scale)")
plt.ylabel("test_overall (%)")
plt.title(f"optim_num vs test_overall | Pearson r={best_r:.4f} | beta={best_beta}")
plt.tight_layout()
plt.show()
