export PYTHONPATH=.:./src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

DATASET=Office     # Office, Grocery, Beauty, Yelp
method_name=T5_CE    # T5_CE, Llama_CE
SEED=42
beam_size=20
test_bsz=512

export WANDB_PROJECT="${DATASET}_${method_name}"
export WANDB_RUN_NAME="${method_name}_seed${SEED}"
OUTPUT_DIR=./ckpt/${DATASET}_${method_name}/${WANDB_RUN_NAME}/
RESULTS_FILE=./results/${DATASET}_${method_name}/$WANDB_RUN_NAME/

python src/train/train_ce.py \
    --output_dir $OUTPUT_DIR \
    --seed $SEED \
    --dataset $DATASET \
    --wandb_run_name $WANDB_RUN_NAME \
    --method_name $method_name

python src/test/test.py \
    --ckpt_path $OUTPUT_DIR \
    --seed $SEED \
    --dataset $DATASET \
    --method_name $method_name \
    --results_file $RESULTS_FILE \
    --test_batch_size $test_bsz \
    --num_beams $beam_size \
    --is_constrained_beam_search


# To reproduce results

DATASET_values=(Office Grocery Beauty Yelp Office Grocery Beauty Yelp)
method_name_values=(T5_CE T5_CE T5_CE T5_CE Llama_CE Llama_CE Llama_CE Llama_CE)
SEED=42
beam_size=20
test_bsz=512

for i in "${!DATASET_values[@]}"; do
    DATASET=${DATASET_values[i]}
    method_name=${method_name_values[i]}

    export WANDB_PROJECT="${DATASET}_${method_name}"
    export WANDB_RUN_NAME="${method_name}_seed${SEED}"
    OUTPUT_DIR=./ckpt/${DATASET}_${method_name}/${WANDB_RUN_NAME}
    RESULTS_FILE=./results/${DATASET}_${method_name}/$WANDB_RUN_NAME/

    python src/train/train_ce.py \
        --output_dir $OUTPUT_DIR \
        --seed $SEED \
        --dataset $DATASET \
        --wandb_run_name $WANDB_RUN_NAME \
        --method_name $method_name

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