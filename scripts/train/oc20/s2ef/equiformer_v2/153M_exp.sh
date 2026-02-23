python main_oc20.py \
    --mode train \
    --config-yml 'oc20/configs/s2ef/2M/equiformer_v2/153M_exp_Oabs.yml' \
    --run-dir 'models' \
    --print-every 200 \
    --amp \
    --checkpoint 'save_models/eq2_153M_ec4_allmd.pt'