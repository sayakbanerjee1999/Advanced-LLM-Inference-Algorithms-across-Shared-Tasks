# Configuration 1
# Change model path between ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B"] to switch between 0.6B and 4B model
# Configuration 2 has all temperature as 0.1, 0.1, 0.1 (change accordingly)
python self_refine.py \
  --dataset graph \
  --model Qwen/Qwen3-0.6B \
  --num_examples 100 \
  --batch_size 32 \
  --draft_temperature 0.75 \
  --feedback_temperature 0.4 \
  --refine_temperature 0.1 \
  --output_dir ./results

python self_refine.py \
  --dataset mmlu_med \
  --model Qwen/Qwen3-0.6B \
  --num_examples 100 \
  --batch_size 32 \
  --draft_temperature 0.75 \
  --feedback_temperature 0.4 \
  --refine_temperature 0.1 \
  --output_dir ./xyz

# Configuration with 4B model
python self_refine.py \
  --dataset graph \
  --model Qwen/Qwen3-4B \
  --num_examples 100 \
  --batch_size 32 \
  --draft_temperature 0.75 \
  --feedback_temperature 0.4 \
  --refine_temperature 0.1 \
  --output_dir ./results

python self_refine.py \
  --dataset mmlu_med \
  --model Qwen/Qwen3-4B \
  --num_examples 100 \
  --batch_size 32 \
  --draft_temperature 0.75 \
  --feedback_temperature 0.4 \
  --refine_temperature 0.1 \
  --output_dir ./results