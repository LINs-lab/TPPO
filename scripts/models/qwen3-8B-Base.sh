MODEL_ARGS=(
   --swiglu
   --num-layers 36
   --hidden-size 4096
   --ffn-hidden-size 12288
   --num-attention-heads 32
   --max-position-embeddings 32768
   --seq-length 32768
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base "${MODEL_ARGS_ROTARY_BASE:-1000000}"
   --vocab-size 151936
   --kv-channels 128
   --qk-layernorm
   --untie-embeddings-and-output-weights
)
