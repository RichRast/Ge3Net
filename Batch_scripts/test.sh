# !/bin/bash
source ini.sh
srun -p gpu --gpus 2 python3 trainer.py --data.params $USER_PATH/src/main/experiments/exp_B \
--data.geno_type humans --model.working_dir $OUT_PATH/humans/training/Model_B_exp_id_1_data_id_6_pca/models_dir/ --data.labels_dir $OUT_PATH/humans/labels/data_id_6_pca --log.dir $OUT_PATH/humans/training/Model_B_exp_id_1_data_id_6_pca