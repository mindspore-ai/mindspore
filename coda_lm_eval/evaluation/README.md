# CODA-LM Evaluation

## Data Preparation

1. First of all, follow the instructions on [CODA-LM submission](https://coda-dataset.github.io/w-coda2024/track1/index.html#submission) to run inference with your LVLM and save your results as requested. Your results should be saved in three `jsonl` files for three tasks separately.
2. If you are using a subset of CODA-LM validation set (e.g., CODA-LM Mini set) for evaluation, you need to copy the corresponding JSON files in a separate `$ROOT_TO_GT` directory like `./CODA_LM/Mini`. Directly setting `$ROOT_TO_GT=./CODA-LM/Val`, but not running inference on the whole validation set will result in a running error. 
3. Now the data organization will be like:

```
├── CODA_LM
│   ├── Train
│   ├── Val
│   ├── Test
│   ├── ROOT_TO_GT
│   │   │── test_*.json
├── ROOT_TO_RESULTS
│   │── general_perception_answer.jsonl
│   │── region_perception_answer.jsonl
│   │── driving_suggestion_answer.jsonl 
```



## Instructions

0. Install additional dependencies for evaluation via pip.

   ```bash
   pip install OpenAI
   ```

1. Insert the ground truth `label name` of each corner case object collected in regional perception.

   ```bash
   # Results will be saved in $ROOT_TO_RESULTS/region_perception_answer_w_label.jsonl
   python convert2eval.py --reference_path ./CODA-LM/$ROOT_TO_GT --prediction_path $ROOT_TO_RESULTS/region_perception_answer.jsonl
   ```

2. Run evaluation for each task separately. By default, we prompt `gpt-4o-2024-05-13` for evaluation.

   ```bash
   # General perception
   python stage1_eval_batch.py --reference_path ./CODA-LM/$ROOT_TO_GT --prediction_path $ROOT_TO_RESULTS/general_perception_answer.jsonl --save_path eval/general_perception_answer --model_name gpt-4o-2024-05-13 --api_key $OPENAI_KEY
   
   # Driving suggestion
   python stage2_eval_batch.py --reference_path ./CODA-LM/$ROOT_TO_GT --prediction_path $ROOT_TO_RESULTS/driving_suggestion_answer.jsonl --save_path eval/driving_suggestion_answer --model_name gpt-4o-2024-05-13 --api_key $OPENAI_KEY
   
   # Regional perception
   python stage3_eval_batch.py --reference_path ./CODA-LM/$ROOT_TO_GT --prediction_path $ROOT_TO_RESULTS/region_perception_answer_w_label.jsonl --save_path eval/region_perception_answer_w_label --model_name gpt-4o-2024-05-13 --api_key $OPENAI_KEY
   ```
