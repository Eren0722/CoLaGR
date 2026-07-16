export PYTHONPATH=.:./src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

DATASET=Office  # Office, Grocery, Beauty, Yelp
method_name=T5_APAO_Pointwise  # T5_APAO_Pointwise, Llama_APAO_Pointwise
SEED=42
beam_size=20
test_bsz=512

beta_values=(0.05 0.1 0.2 0.3 0.4)
eta_values=(5e-6 1e-5 3e-5 5e-5 1e-4 5e-4)

for beta in "${beta_values[@]}"; do
    for eta in "${eta_values[@]}"; do

        export WANDB_PROJECT="${DATASET}_${method_name}"
        export WANDB_RUN_NAME="${method_name}_seed${SEED}_beta${beta}_eta${eta}"
        OUTPUT_DIR=./ckpt/${DATASET}_${method_name}/${WANDB_RUN_NAME}/
        RESULTS_FILE=./results/${DATASET}_${method_name}/$WANDB_RUN_NAME/

        python src/train/train_apao.py \
            --output_dir $OUTPUT_DIR \
            --seed $SEED \
            --dataset $DATASET \
            --wandb_run_name $WANDB_RUN_NAME \
            --method_name $method_name \
            --beta $beta \
            --eta_adapt $eta

        python src/test/test.py \
            --ckpt_path $OUTPUT_DIR \
            --seed $SEED \
            --dataset $DATASET \
            --method_name $method_name \
            --results_file $RESULTS_FILE \
            --test_batch_size $test_bsz \
            --num_beams $beam_size \
            --is_constrained_beam_search
    done
done

# To reproduce results

DATASET_values=(Office Grocery Beauty Yelp Office Grocery Beauty Yelp)
method_name_values=(T5_APAO_Pointwise T5_APAO_Pointwise T5_APAO_Pointwise T5_APAO_Pointwise Llama_APAO_Pointwise Llama_APAO_Pointwise Llama_APAO_Pointwise Llama_APAO_Pointwise)
SEED=42
beam_size=20
test_bsz=512

beta_values=(0.3 0.1 0.1 0.05 0.4 0.4 0.3 0.4)
eta_values=(1e-4 5e-5 1e-5 1e-5 5e-4 5e-5 1e-4 1e-5)

for i in "${!DATASET_values[@]}"; do
    DATASET=${DATASET_values[i]}
    method_name=${method_name_values[i]}
    beta=${beta_values[i]}
    eta=${eta_values[i]}

    export WANDB_PROJECT="${DATASET}_${method_name}"
    export WANDB_RUN_NAME="${method_name}_seed${SEED}_beta${beta}_eta${eta}"
    OUTPUT_DIR=./ckpt/${DATASET}_${method_name}/${WANDB_RUN_NAME}/
    RESULTS_FILE=./results/${DATASET}_${method_name}/$WANDB_RUN_NAME/

    python src/train/train_apao.py \
        --output_dir $OUTPUT_DIR \
        --seed $SEED \
        --dataset $DATASET \
        --wandb_run_name $WANDB_RUN_NAME \
        --method_name $method_name \
        --beta $beta \
        --eta_adapt $eta

    python src/test/test.py \
        --ckpt_path $OUTPUT_DIR \
        --seed $SEED \
        --dataset $DATASET \
        --method_name $method_name \
        --results_file $RESULTS_FILE \
        --test_batch_size $test_bsz \
        --num_beams $beam_size \
        --is_constrained_beam_search
done