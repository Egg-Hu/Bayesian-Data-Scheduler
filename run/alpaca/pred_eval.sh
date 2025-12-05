export OPENAI_API_KEY=placeholder
alpaca_eval --model_outputs './run/scripts/ckpt_nscc/llama2_scalar_softmax_entropy0_alpaca_p0.1_1000_1000_100_noprior_noleakage_5000/alpaca_steps/all.json' --annotators_config 'alpaca_eval_gpt4_turbo_fn'
