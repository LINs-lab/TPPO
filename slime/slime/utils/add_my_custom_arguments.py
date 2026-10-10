def add_my_custom_arguments(parser):
    parser.add_argument("--critic-ref-load", type=str, default=None,
                        help="The reference checkpoint for critic model (Megatron format). Used when critic-load is invalid.")

    parser.add_argument(
        "--critic-hf-checkpoint",
        type=str,
        default=None,
        help="The checkpoint for critic model's HF config. If not set, will use the same as the critic model.")
    parser.add_argument("--use-asyppo", action="store_true", default=False,
                        help="Whether to use AsyPPO,critic model must have the same tokenizer as the actor model.")

    parser.add_argument("--use-decoupled-gae", action="store_true", default=False,
                        help="Use separate actor/critic GAE lambdas; otherwise preserve --lambd.")
    parser.add_argument("--ppo-token-level-loss", action="store_true", default=False,
                        help="Average PPO policy gradient loss over all valid tokens in an optimizer step; leave critic reduction unchanged.")
    parser.add_argument("--lambd-actor", type=float,
                        default=0.95, help="PPO GAE lambd for actor")

    parser.add_argument("--lambd-critic", type=float,
                        default=0.95, help="PPO GAE lambd for critic")

    parser.add_argument(
        "--log-position-value-stats",
        action="store_true",
        default=False,
        help="Whether to log the position-value stats.",
    )

    parser.add_argument(
        "--max-log-positions",
        type=int,
        default=500,
        help="The maximum number of positions to log the value stats.",
    )
    parser.add_argument("--use-adaptive-lambda", action="store_true",
                        default=False, help="Use actor lambda=clamp(1-1/(alpha*response_length), 0, 1); requires --use-decoupled-gae.")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="The alpha for adaptive lambda for PPO GAE.")
    parser.add_argument(
        "--rm-timeout-seconds",
        "--rm-timeout",
        dest="rm_timeout_seconds",
        type=float,
        default=600.0,
        help="Per-sample RM timeout in seconds. Set <=0 to disable.",
    )
    parser.add_argument("--train-use-xverify", action="store_true",
                        default=False, help="Whether to use XVerify for training.")
    parser.add_argument("--eval-use-xverify", action="store_true",
                        default=False, help="Whether to use XVerify for evaluation.")
    parser.add_argument("--eval-group", action="store_true",
                        default=False, help="Whether to evaluate the model in group.")
    parser.add_argument("--eval-log-dir", type=str, default=".",
                        help="The directory to save the evaluation results.")
    parser.add_argument("--eval-first", action="store_true",
                        default=False, help="Whether to only evaluate the model.")
    parser.add_argument(
        "--skip-eval-during-critic-warmup",
        action="store_true",
        default=False,
        help="Skip periodic evaluation while rollout_id is below num_critic_only_steps.",
    )
    parser.add_argument(
        "--analysis-export-only",
        action="store_true",
        default=False,
        help=(
            "Run rollout/logprob/value/advantage postprocess for analysis export only, "
            "then skip optimizer updates and checkpoint saves."
        ),
    )
    parser.add_argument(
        "--analysis-full-gradient",
        action="store_true",
        default=False,
        help=(
            "Compute exact untied output-layer and full-model policy-gradient diagnostics "
            "during analysis export. This runs real backward passes without taking optimizer steps."
        ),
    )
    parser.add_argument(
        "--analysis-clipped-gradient",
        action="store_true",
        default=False,
        help="Add PPO-clipped critic/oracle full-model gradient statistics at MC anchors and all response tokens.",
    )
    parser.add_argument(
        "--analysis-clipped-gradient-use-tis",
        action="store_true",
        default=False,
        help="Use vanilla TIS weights in clipped-gradient diagnostics; fail if genuine rollout/old log-probs are absent.",
    )
    parser.add_argument(
        "--analysis-old-actor-checkpoint",
        type=str,
        default=None,
        help="Optional old-policy checkpoint used to recompute token log-probs for PPO clipping during replay.",
    )
    parser.add_argument(
        "--analysis-full-gradient-samples",
        type=int,
        default=512,
        help=(
            "Maximum local responses used by exact full-model gradient analysis. "
            "The 8B Polaris default covers one complete 64-prompt x 8-response rollout."
        ),
    )
    parser.add_argument(
        "--analysis-full-gradient-microbatch-size",
        type=int,
        default=256,
        help=(
            "Responses per statistical gradient batch. This does not control GPU memory; "
            "the existing dynamic microbatch scheduler still enforces max-tokens-per-gpu."
        ),
    )
    parser.add_argument(
        "--analysis-per-anchor-gradient",
        action="store_true",
        default=False,
        help=(
            "Compute one exact full-model score-gradient backward for every valid MC anchor. "
            "No aggregate gradient is used as a per-anchor result."
        ),
    )
    parser.add_argument(
        "--analysis-per-token-gradient",
        action="store_true",
        default=False,
        help=(
            "Compute exact full-model score-gradient norms for selected token positions. "
            "Use --analysis-per-anchor-gradient for the sparse MC-anchor analysis."
        ),
    )
    parser.add_argument(
        "--save-latest-best-ckpts",
        action="store_true",
        default=False,
        help=(
            "Track the latest and best checkpoints. With --best-save, the best checkpoint "
            "is selected after every evaluation and stored separately."
        ),
    )
    parser.add_argument(
        "--best-save",
        type=str,
        default=None,
        help=(
            "Separate directory for checkpoints selected as best after evaluation. "
            "When set, these checkpoints are not written into the regular iter_* directory."
        ),
    )
    parser.add_argument(
        "--critic-best-save",
        type=str,
        default=None,
        help="Separate directory for best critic checkpoints selected after evaluation.",
    )
    parser.add_argument(
        "--save-only-best",
        action="store_true",
        default=False,
        help="Disable regular iter_* checkpoint saves and save only the current best checkpoint.",
    )
    parser.add_argument(
        "--save-all-ckpts",
        action="store_true",
        default=False,
        help=(
            "Keep every checkpoint written at a save point while also tracking "
            "latest and best checkpoints."
        ),
    )
    parser.add_argument(
        "--save-best-metric",
        type=str,
        default="eval/overall",
        help="Metric key used by --save-latest-best-ckpts. Higher is better.",
    )
    parser.add_argument(
        "--save-best-window-ends",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Rollout-count window ends for --save-latest-best-ckpts. "
            "For example, 500 1000 keeps one best checkpoint in [0, 500) "
            "and one best checkpoint in [500, 1000), in addition to latest/global best."
        ),
    )
    parser.add_argument("--log-position-advantage-stats", action="store_true",
                        default=False, help="Whether to log the position-advantage stats.")
    parser.add_argument("--max-log-positions-advantage", type=int, default=500,
                        help="The maximum number of positions to log the advantage stats.")
    parser.add_argument("--use-positive-nll-loss", action="store_true",
                        default=False, help="Whether to use positive NLL loss.")
    parser.add_argument("--positive-nll-coef", type=float,
                        default=0.1, help="The coefficient for positive NLL loss.")
    parser.add_argument("--positive-reward-threshold", type=float,
                        default=0.0, help="Select positive NLL responses when the raw verifier reward is strictly above this threshold.")

    parser.add_argument("--use-critic2", action="store_true", default=False,
                        help="Whether to use a second critic model.")
    parser.add_argument("--critic2-num-nodes", type=int, default=None,
                        help="Number of nodes for the second critic model.")
    parser.add_argument("--critic2-num-gpus-per-node", type=int, default=None,
                        help="Number of GPUs per node for the second critic model.")
    parser.add_argument("--critic2-lr", type=float, default=None,
                        help="Learning rate for the second critic model.")
    parser.add_argument("--critic2-hf-checkpoint", type=str, default=None,
                        help="The checkpoint for critic2 model's HF config.")
    parser.add_argument("--critic2-ref-load", type=str, default=None,
                        help="The reference checkpoint for critic2 model (Megatron format).")
    parser.add_argument("--critic2-save", type=str, default=None,
                        help="The save path for critic2 model.")
    parser.add_argument("--critic2-lr-warmup-iters", type=int, default=None,
                        help="Learning rate warmup iterations for critic2.")
    parser.add_argument("--critic2-load", type=str, default=None,
                        help="The load path for critic2 model.")


    parser.add_argument("--use-asytrain-critic", action="store_true", default=False,
                        help="Whether to use asynchronous critic training where different critics train on different samples.")
    parser.add_argument(
        '--use-advantage-diff-mask',
        action='store_true',
        help='Enable advantage masking based on critic value agreement (mask samples with the lowest standard deviation)'
    )
    parser.add_argument(
        '--advantage-diff-mask-k',
        type=float,
        default=0.1,
        help='Percentage of samples to mask (with lowest std/highest agreement). Default: 0.1 (10%)'
    )
    parser.add_argument(
        '--use-entropy-value-divergence-filter',
        action='store_true',
        help='Enable entropy value divergence filter.'
    )
    parser.add_argument(
        '--entropy-divergence-filter-h',
        type=float,
        default=0.2,
        help='Percentage of samples to filter (with highest entropy value divergence). Default: 0.1 (10%)'
    )

    parser.add_argument(
        '--use-bce-value-loss',
        action='store_true',
        help=(
            'Train the scalar critic head with sigmoid + BCE against the 0/1 return, '
            'WITHOUT the TPPO cumulative-sum reparametrization. Each response '
            'token`s head output is treated directly as a value logit: value = sigmoid(logit), '
            'loss = BCE(logit, return). Mutually exclusive with --critic-learn-tppo '
            'and --critic-value-loss-type=hl_gauss.'
        ),
    )
    parser.add_argument(
        '--bce-value-logit-clip',
        type=float,
        default=0.0,
        help=(
            'PPO-style clipping range in value-logit space for --use-bce-value-loss. '
            'Set <= 0 to disable (plain BCE). Mirrors --critic-tppo-logit-clip '
            'but for the non-delta sigmoid+BCE critic.'
        ),
    )

    parser.add_argument(
        "--critic-value-loss-type",
        type=str,
        choices=["mse", "hl_gauss"],
        default="mse",
        help="Critic value loss type. `mse` keeps PPO clipped-MSE, `hl_gauss` uses distributional CE.",
    )
    parser.add_argument(
        "--critic-output-layer-init-mode",
        type=str,
        choices=["auto_reinit", "legacy_v024"],
        default="auto_reinit",
        help=(
            "Critic output/value head initialization mode. `auto_reinit` uses the current "
            "post-load missing/shape-mismatch guard; `legacy_v024` matches THUDM/slime v0.2.4 "
            "by keeping the constructor normal(0, 0.02) head when checkpoint loading skips it."
        ),
    )
    parser.add_argument(
        "--hlg-num-bins",
        type=int,
        default=101,
        help="Number of bins for HL-Gauss critic head when --critic-value-loss-type=hl_gauss.",
    )
    parser.add_argument(
        "--hlg-v-min",
        type=float,
        default=0.0,
        help="Minimum value support for HL-Gauss targets.",
    )
    parser.add_argument(
        "--hlg-v-max",
        type=float,
        default=1.0,
        help="Maximum value support for HL-Gauss targets.",
    )
    parser.add_argument(
        "--hlg-sigma-ratio",
        type=float,
        default=0.75,
        help="HL-Gauss sigma / delta ratio, delta=(v_max-v_min)/num_bins.",
    )
    parser.add_argument(
        "--hlg-target-sigma-mode",
        type=str,
        choices=["none", "position", "position_group",
                 "position_group_conflict", "mc_sem"],
        default="none",
        help=(
            "Optional token-wise shaping for HL-Gauss target sigma. `none` keeps the fixed "
            "--hlg-sigma-ratio behavior; `mc_sem` uses online MC std/sqrt(n) with the "
            "base sigma as floor; `position` narrows targets later in the response; "
            "`position_group` also widens uncertain prompt groups; `position_group_conflict` "
            "also widens reward/group-conflict tokens."
        ),
    )
    parser.add_argument(
        "--hlg-target-sigma-min-scale",
        type=float,
        default=0.5,
        help="Minimum target sigma as a scale of the base --hlg-sigma-ratio sigma when sigma shaping is enabled.",
    )
    parser.add_argument(
        "--hlg-target-sigma-max-scale",
        type=float,
        default=2.0,
        help="Maximum target sigma as a scale of the base --hlg-sigma-ratio sigma when sigma shaping is enabled.",
    )
    parser.add_argument(
        "--hlg-target-sigma-position-gamma",
        type=float,
        default=1.0,
        help="Exponent for position-based HL-Gauss target sigma decay.",
    )
    parser.add_argument(
        "--hlg-target-sigma-group-coef",
        type=float,
        default=0.5,
        help="Coefficient for widening target sigma by sqrt(G * (1 - G)) in group sigma modes.",
    )
    parser.add_argument(
        "--hlg-target-sigma-conflict-coef",
        type=float,
        default=0.5,
        help="Coefficient for widening target sigma by abs(return - group_success_p) in conflict sigma mode.",
    )
    parser.add_argument(
        "--hlg-clip-aux-coef",
        type=float,
        default=1.0,
        help="Auxiliary weight for expectation-space PPO-style value-clip loss when using HL-Gauss.",
    )
    parser.add_argument(
        "--hlg-clip-aux-target",
        type=str,
        choices=["returns", "target_mean"],
        default="returns",
        help=(
            "Scalar target for the HL-Gauss expectation-space value-clip auxiliary loss. "
            "`returns` preserves the old raw-return auxiliary; `target_mean` aligns E[Z] "
            "to the final HL-Gauss target distribution mean."
        ),
    )
    parser.add_argument(
        "--hlg-zero-init-head",
        action="store_true",
        default=True,
        help="Zero-initialize critic head for HL-Gauss so initial logits are close to uniform.",
    )
    parser.add_argument(
        "--disable-hlg-zero-init-head",
        action="store_false",
        dest="hlg_zero_init_head",
        help="Disable zero initialization for HL-Gauss critic head.",
    )
    parser.add_argument(
        "--hlg-target-mode",
        type=str,
        choices=["single", "group_bernoulli_mixture", "loo_position_beta"],
        default="single",
        help=(
            "HL-Gauss target construction. `single` keeps the old scalar-return Gaussian target; "
            "`group_bernoulli_mixture` mixes single-return and prompt-group Bernoulli targets; "
            "`loo_position_beta` projects a leave-one-out group prior plus position-dependent "
            "response evidence Beta belief onto HLG bins."
        ),
    )
    parser.add_argument(
        "--hlg-group-target-weight",
        type=float,
        default=0.0,
        help="Mixture weight for prompt-group Bernoulli target when --hlg-target-mode=group_bernoulli_mixture.",
    )
    parser.add_argument(
        "--hlg-group-target-alpha",
        type=float,
        default=0.5,
        help="Beta prior smoothing alpha for prompt-group success probability.",
    )
    parser.add_argument(
        "--hlg-beta-target-prior-alpha",
        type=float,
        default=0.5,
        help="Symmetric Beta prior alpha for --hlg-target-mode=loo_position_beta.",
    )
    parser.add_argument(
        "--hlg-beta-target-group-evidence-mult",
        type=float,
        default=1.0,
        help="Multiplier c for leave-one-out group success/failure counts in loo_position_beta targets.",
    )
    parser.add_argument(
        "--hlg-beta-target-response-evidence",
        type=float,
        default=1.0,
        help="Maximum current-response evidence added at the final token in loo_position_beta targets.",
    )
    parser.add_argument(
        "--hlg-beta-target-position-gamma",
        type=float,
        default=1.0,
        help="Exponent gamma for progress^gamma response-evidence schedule in loo_position_beta targets.",
    )
    parser.add_argument(
        "--hlg-advantage-mode",
        type=str,
        choices=["none", "risk_baseline", "uncertainty_weighted"],
        default="none",
        help="Optional uncertainty-aware advantage adjustment for HL-Gauss PPO.",
    )
    parser.add_argument(
        "--hlg-adv-beta",
        type=float,
        default=0.0,
        help=(
            "Uncertainty coefficient for HL-Gauss advantage modes. "
            "For risk_baseline, subtract beta * value_std. For uncertainty_weighted, "
            "multiply advantages by 1 / (1 + beta * normalized_value_std) with scale preservation."
        ),
    )
    parser.add_argument(
        "--hlg-save-debug-stats",
        action="store_true",
        help="Store extra HL-Gauss distribution statistics in rollout_data/debug dumps.",
    )
    parser.add_argument(
        "--hlg-debug-distribution-samples",
        type=int,
        default=0,
        help="Number of local samples to save full HL-Gauss value distributions for each rollout. 0 disables it.",
    )
    parser.add_argument(
        "--hlg-debug-distribution-dir",
        type=str,
        default=None,
        help="Directory for sampled HL-Gauss value distribution debug files. Defaults under --dump-details when enabled.",
    )
    parser.add_argument(
        "--hlg-debug-distribution-max-tokens",
        type=int,
        default=0,
        help="Maximum response tokens saved per sampled distribution. 0 saves all response tokens.",
    )

    # parser.add_argument(
    # "--log-probs-chunk-size",
    # type=int,
    # default=-1,
    # help="Chunk size for log_probs/entropy. -1 disables. Use 8192 for long sequences.",
    # )

    parser.add_argument(
        "--critic-token-loss",
        action="store_true",
        help="Enable token loss for critic model.",
    )
    parser.add_argument(
        "--dense-critic-only-warmup",
        action="store_true",
        help=(
            "During rollout_id < --num-critic-only-steps, train critic value loss on the original "
            "dense return masks instead of applying critic-token-loss or MC-label masks. After the "
            "critic-only warmup, configured sparse/MC critic masks are used normally."
        ),
    )
    parser.add_argument(
        "--critic-token",
        type=int,
        default=None,
        help="Token id used by critic-token-loss. Example: set to tokenizer id of '\\n' to compute critic loss only on newline token positions in response.",
    )
    parser.add_argument(
        "--critic-token-ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional list version of --critic-token, e.g. '--critic-token-ids 198 1699'. Combined with --critic-token.",
    )
    parser.add_argument(
        "--critic-token-interval",
        type=int,
        default=None,
        help="If set (>0), compute critic loss every N response tokens (e.g. 256). Takes precedence over --critic-token/--critic-token-ids.",
    )
    parser.add_argument(
        "--critic-token-ratios",
        type=float,
        nargs="+",
        default=None,
        help=(
            "If set, compute critic loss at these response-relative ratios over valid tokens "
            "(e.g. '--critic-token-ratios 0.3 0.6 0.9'). Takes precedence over "
            "--critic-token-interval and --critic-token/--critic-token-ids."
        ),
    )
    parser.add_argument(
        "--critic-extra-tail-ratio",
        type=float,
        default=None,
        help=(
            "Optional extra response-relative valid-token ratio for critic value loss, enabled only "
            "when the valid response length is at least --critic-extra-tail-min-response-len. "
            "This is useful for supervising near-tail value on unusually long responses without "
            "changing the default sparse critic positions."
        ),
    )
    parser.add_argument(
        "--critic-extra-tail-min-response-len",
        type=int,
        default=0,
        help=(
            "Minimum valid response length required to add --critic-extra-tail-ratio to the critic "
            "value-loss mask. Ignored when --critic-extra-tail-ratio is unset."
        ),
    )
    parser.add_argument(
        "--critic-label-source",
        type=str,
        choices=["returns", "mc_mean"],
        default="returns",
        help=(
            "Value target source for critic training. `returns` keeps the PPO/GAE target; "
            "`mc_mean` uses online MC rollout means stored in mc_value_labels."
        ),
    )
    parser.add_argument(
        "--critic-mc-token-ratios",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Response-relative valid-token ratios where online MC value labels are computed. "
            "If unset, --critic-token-ratios is reused."
        ),
    )
    parser.add_argument(
        "--critic-mc-supervise-coef",
        type=float,
        default=0.0,
        help=(
            "Additional critic MSE coefficient on sparse online-MC value labels. "
            "This keeps the normal value target from --critic-label-source and adds "
            "coef * MSE(V(anchor), mc_mean(anchor)) at --critic-mc-token-ratios."
        ),
    )
    parser.add_argument(
        "--critic-mc-min-samples",
        type=int,
        default=4,
        help="Minimum number of continuation samples per MC value-label anchor.",
    )
    parser.add_argument(
        "--critic-mc-max-samples",
        type=int,
        default=24,
        help="Maximum number of continuation samples per MC value-label anchor.",
    )
    parser.add_argument(
        "--critic-mc-batch-size",
        type=int,
        default=4,
        help="Number of continuation samples added per MC value-label round.",
    )
    parser.add_argument(
        "--critic-mc-anchor-concurrency",
        type=int,
        default=1,
        help=(
            "Maximum number of response-token MC anchors sampled concurrently. "
            "Default 1 preserves the original serial anchor scheduling."
        ),
    )
    parser.add_argument(
        "--critic-mc-anchor-log-interval",
        type=int,
        default=0,
        help="Log online MC value-label progress every N completed anchors. Set 0 to disable progress logs.",
    )
    parser.add_argument(
        "--critic-mc-sample-timeout-seconds",
        type=float,
        default=900.0,
        help=(
            "Timeout for one online MC continuation sample, including generation and per-sample RM when "
            "--group-rm is off. Set <=0 to disable."
        ),
    )
    parser.add_argument(
        "--critic-mc-rm-timeout-seconds",
        type=float,
        default=300.0,
        help="Timeout for the online MC batched RM call used when --group-rm is on. Set <=0 to disable.",
    )
    parser.add_argument(
        "--critic-mc-anchor-timeout-seconds",
        type=float,
        default=1800.0,
        help="Timeout for one online MC anchor across all adaptive sampling rounds. Set <=0 to disable.",
    )
    parser.add_argument(
        "--critic-mc-ci-halfwidth-target",
        type=float,
        default=0.10,
        help="Stop MC sampling once the CI half-width is no larger than this value.",
    )
    parser.add_argument(
        "--critic-mc-z-value",
        type=float,
        default=1.96,
        help="Z value used to compute MC value-label confidence intervals.",
    )
    parser.add_argument(
        "--critic-mask-ratio",
        type=float,
        default=0.1,
        help="The ratio of tokens to mask for critic model.",
    )
    parser.add_argument(
        "--policy-token-value-broadcast",
        action="store_true",
        default=False,
        help="Enable policy-side value broadcasting based on matched token positions (for advantage/return computation).",
    )
    parser.add_argument(
        "--policy-token",
        type=int,
        default=None,
        help="Token id used by policy-token-value-broadcast. Example: tokenizer id of '\\n'.",
    )
    parser.add_argument(
        "--policy-token-ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional list version of --policy-token, e.g. '--policy-token-ids 198 1699'. Combined with --policy-token.",
    )
    parser.add_argument(
        "--policy-token-interval",
        type=int,
        default=None,
        help="If set (>0), broadcast policy values every N response tokens (e.g. 256). Takes precedence over --policy-token/--policy-token-ids.",
    )
    parser.add_argument(
        "--policy-token-ratios",
        type=float,
        nargs="+",
        default=None,
        help=(
            "If set, broadcast policy values from response-relative anchor ratios over valid tokens "
            "(e.g. '--policy-token-ratios 0.3 0.6 0.9'). Takes precedence over "
            "--policy-token-interval and --policy-token/--policy-token-ids."
        ),
    )
    parser.add_argument(
        "--critic-pretrain-reweight",
        action="store_true",
        default=False,
        help="Enable trajectory-level loss reweighting for critic-pretrain data using metadata.progress_percent.",
    )
    parser.add_argument(
        "--critic-pretrain-reweight-min",
        type=float,
        default=0.2,
        help="Minimum sample weight for early-progress trajectories when reweighting is enabled.",
    )
    parser.add_argument(
        "--critic-pretrain-reweight-power",
        type=float,
        default=2.0,
        help="Power for progress->weight mapping: w = w_min + (1-w_min)*(progress^power).",
    )
    parser.add_argument(
        "--critic-pretrain-reward-error-power",
        type=float,
        default=1.0,
        help="Exponent used by critic-pretrain reward: r = 1 - |pred_p - target_p|^power. Set 1 for L1, 2 for Brier-like.",
    )
    return parser
