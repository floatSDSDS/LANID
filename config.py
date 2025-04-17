from argparse import ArgumentParser


def get_parser():
    parser = ArgumentParser()

    parser.add_argument("--data_dir", default='data', type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--save_results_path", type=str, default='outputs',
                        help="The path to save results.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="The path or name for the pre-trained bert model.")
    parser.add_argument("--tokenizer", default="bert-base-uncased", type=str,
                        help="The path or name for the tokenizer")
    parser.add_argument("--feat_dim", default=768, type=int,
                        help="Bert feature dimension.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Warmup proportion for optimizer.")
    parser.add_argument("--save_model_path", default=None, type=str,
                        help="Path to save model checkpoints. Set to None if not save.")
    parser.add_argument("--dataset", default=None, type=str, required=True,
                        help="Name of dataset.")
    parser.add_argument("--known_cls_ratio", default=0.75, type=float, required=True,
                        help="The ratio of known classes.")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed.")
    parser.add_argument("--method", type=str, default='dbscanGPT',
                        help="The name of method.")
    parser.add_argument("--labeled_ratio", default=0.1, type=float,
                        help="The ratio of labeled samples.")
    parser.add_argument("--rtr_prob", default=0.25, type=float,
                        help="Probability for random token replacement")
    parser.add_argument("--pretrain_batch_size", default=64, type=int,
                        help="Batch size for pre-training")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--wait_patient", default=20, type=int,
                        help="Patient steps for Early Stop in pretraining.")
    parser.add_argument("--num_pretrain_epochs", default=100, type=float,
                        help="The pre-training epochs.")
    parser.add_argument("--num_train_epochs", default=34, type=float,
                        help="The training epochs.")
    parser.add_argument("--lr_pre", default=5e-5, type=float,
                        help="The learning rate for pre-training.")
    parser.add_argument("--lr", default=1e-5, type=float,
                        help="The learning rate for training.")
    parser.add_argument("--temp", default=0.07, type=float,
                        help="Temperature for contrastive loss")
    parser.add_argument("--view_strategy", default="rtr", type=str,
                        help="Choose from rtr|shuffle|none")
    parser.add_argument("--update_per_epoch", default=5, type=int,
                        help="Update pseudo labels after certain amount of epochs")
    parser.add_argument("--report_pretrain", action="store_true",
                        help="Enable reporting performance right after pretrain")
    parser.add_argument("--topk", default=50, type=int,
                        help="Select topk nearest neighbors")
    parser.add_argument("--grad_clip", default=1, type=float,
                        help="Value for gradient clipping.")

    # new params
    parser.add_argument("--path_prompt", default='prompt/prompts.txt', type=str,
                        help="path that store promopts for each dataset.")
    parser.add_argument("--file_result", default='results_0623.csv', type=str,
                        help="path that store promopts for each dataset.")

    # dbscan
    # parser.add_argument("--eps", default=3, type=float, help="initial eps for dbscan.")
    parser.add_argument("--minpts", default=4, type=int, help="initial minpts for dbscan.")
    parser.add_argument("--dbscan_q", default=0.25, type=float, help="quantile of dbscan points eps")

    # gpt ref:fxnlprchatzoo
    parser.add_argument("--gpt_model", default='gpt-3.5-turbo', type=str,
                        help="gpt model as implemented in fxnlprchatzoo.")
    parser.add_argument("--gpt_temp", default=0.0, type=float,
                        help="gpt temperature.")
    parser.add_argument("--gpt_p", default=1.0, type=float,
                        help="gpt p.")
    parser.add_argument("--gpt_max_tk", default=256, type=int,
                        help="gpt max_new_tokens.")

    # sampling
    parser.add_argument("--sampling", default='near', type=str,
                        help="strategy for input sampling [near, dbscan, both]")
    parser.add_argument("--sample_k", default=8, type=int,
                        help="number of topk nearest for each sampled utterance")
    parser.add_argument("--p_outlier", default=0.05, type=float,
                        help="proportion of sampled outliers")
    parser.add_argument("--p_core", default=0.05, type=float,
                        help="proportion of sampled cores for each cluster")
    parser.add_argument("--k_pos", default=2, type=int,
                        help="number of sampled positives for each sampled utterance")
    parser.add_argument("--k_pos_db", default=1, type=int,
                        help="number of sampled positives for each sampled utterance")
    parser.add_argument("--k_neg", default=8, type=int,
                        help="number of sampled negative for each sampled utterance")

    # training
    parser.add_argument("--update_step_cl", default=5, type=int,
                        help="dbscan update step.")
    parser.add_argument("--update_step_gpt", default=2, type=int,
                        help="dbscan update step.")
    parser.add_argument("--gpt_epoch", default=10, type=int,
                        help="gpt training epoch.")
    parser.add_argument("--c", default=1, type=float,
                        help="dbscan_gpt loss strength coefficient.")
    parser.add_argument("--batch_gpt", default=32, type=int,
                        help="the size of minibatch for training with gpt")
    parser.add_argument("--max_gpt_ep", default=8192, type=int,
                        help="the maximum size of a data epoch for training with gpt")
    parser.add_argument("--temperature", default=0.07, type=float,
                        help="dbscan_gpt contrastive loss temperature.")
    parser.add_argument("--train_task", default='cl', type=str,
                        help="training loss [cl, pair]")

    parser.add_argument("--fast", action="store_true",
                        help="if fast, load pretrained")
    parser.add_argument("--clnn", action="store_true",
                        help="Enable full training process for CLNN")
    parser.add_argument("--use_known", action="store_true",
                        help="if use known data for gpt training")
    return parser
