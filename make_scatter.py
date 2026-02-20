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


"""
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paste ALL your log lines for trial_27, trial_34, trial_36 (and any others) here.


BETA = 10.0

pattern = re.compile(
    r"\[trial_(\d+)\]\[Epoch\s+(\d+)\]\s+val_acc=([0-9.]+).*?"
    r"ig_fwd_kl=([0-9.]+).*?"
    r"test=\{.*?'overall':\s*([0-9.]+)\}",
    re.IGNORECASE
)

rows = []
for m in pattern.finditer(LOG_TEXT):
    trial = int(m.group(1))
    epoch = int(m.group(2))
    val_acc = float(m.group(3))
    ig_fwd_kl = float(m.group(4))
    test_overall = float(m.group(5))

    optim_num = val_acc * math.exp(-BETA * ig_fwd_kl)
    rows.append((trial, epoch, val_acc, ig_fwd_kl, test_overall, optim_num))

df = pd.DataFrame(
    rows,
    columns=["trial", "epoch", "val_acc", "ig_fwd_kl", "test_overall", "optim_num"]
)

if df.empty:
    raise ValueError(
        "No epochs parsed. Check LOG_TEXT and that test dict has single quotes like 'overall'."
    )

# Correlations pooled over all (trial, epoch) points
r_val = float(np.corrcoef(df["val_acc"].to_numpy(), df["test_overall"].to_numpy())[0, 1])
r_optim = float(np.corrcoef(df["optim_num"].to_numpy(), df["test_overall"].to_numpy())[0, 1])

print(f"Pooled Pearson r(val_acc, test_overall)  = {r_val:.6f}")
print(f"Pooled Pearson r(optim_num, test_overall) = {r_optim:.6f}  (beta={BETA:g})")

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

plt.xlabel(f"optim_num = val_acc * exp(-{BETA:g} * ig_fwd_kl)  (log x-scale)")
plt.ylabel("test_overall (%)")
plt.title(f"optim_num vs test_overall | Pearson r={r_optim:.4f}")
plt.tight_layout()
plt.show()