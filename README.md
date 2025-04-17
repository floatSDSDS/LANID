# LANID: LLM-assisted New Intent Discovery

## Data
We have already included the data folder in our code.

## Requirements
This is the environment we have testified to be working. However, other versions might be working too.

python==3.8
pytorch==1.10.0
transformers==4.15.0
faiss-gpu==1.7.2
numpy
pandas
scikit-learn
openai
sentence_transformers
tqdm==4.65.0
httpx==0.23.3
tiktoken==0.3.3

## Download External Pretrained Models
The external pretraining is conducted with [IntentBert](https://github.com/fanolabs/IntentBert).
You can also download the pretrained checkpoints from following [link](https://drive.google.com/drive/folders/1k4kI5EYEBibId3cEydcjgLOwb5Y4noCT?usp=sharing). And then put them into a folder ``pretrained_models`` in root directory. The link includes following models.

* IntentBert-banking
* IntentBert-mcid
* IntentBert-stackoverflow

## Run
Depending on the selected model, fill in either the 'huggingface_authorize_token' or the 'openai_api_key' in 'LANID.py'.
```
python main.py --data_dir data --dataset banking --bert_model ./pretrained_models/banking  --known_cls_ratio 0.75 --labeled_ratio 0.1 --seed 0 --lr 1e-5 --save_results_path clnn_outputs  --topk 50 --p_core 0.05 --k_pos 2 --batch_gpt 8 --train_task pair --gpt_epoch 10 --update_step_gpt 2  --method pair_semi --use_known --sampling dbscan --file_result banking75.csv --gpt_model gpt-3.5-turbo
```

## Parameters：

```
# DBSCAN
- eps: The initial eps parameter for DBSCAN
- minpts: The initial minpts parameter for DBSCAN
- strategy_dbscan: The parameter update strategy for DBSCAN (todo)

# Data Composition Strategy
- strategy_input: Data sampling strategy, scalar: use a fixed number of outliers / or prop: sample a certain proportion of data for outliers or each cluster
- n_outlier: Number of outliers sampled when strategy_input=scalar
- n_core: Number of samples drawn for each cluster when strategy_input=scalar (sampling with replacement)
- p_outlier: Proportion of outliers sampled when strategy_input=prop
- p_core: Proportion of samples drawn for each cluster when strategy_input=prop
- k_pos: Number of positive examples matched for each sample point. For outliers, match the top k nearest core points. For each core point, match the top k random points from the same cluster. If the number of core points within the cluster is insufficient, fill with the top k nearest other core points.
- b_mini: Number of data pairs sampled each time
```

For more details on the meaning and settings of other parameters, please refer to init_parameter.py.

## Citation
```
@inproceedings{fan-etal-2024-lanid,
    title = "{LANID}: {LLM}-assisted New Intent Discovery",
    author = "Fan, Lu  and
      Pu, Jiashu  and
      Zhang, Rongsheng  and
      Wu, Xiao-Ming",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.883/",
    pages = "10110--10116",
    abstract = "Data annotation is expensive in Task-Oriented Dialogue (TOD) systems. New Intent Discovery (NID) is a task aims to identify novel intents while retaining the ability to recognize known intents. It is essential for expanding the intent base of task-based dialogue systems. Previous works relying on external datasets are hardly extendable. Meanwhile, the effective ones are generally depends on the power of the Large Language Models (LLMs). To address the limitation of model extensibility and take advantages of LLMs for the NID task, we propose LANID, a framework that leverages LLM`s zero-shot capability to enhance the performance of a smaller text encoder on the NID task. LANID employs KNN and DBSCAN algorithms to select appropriate pairs of utterances from the training set. The LLM is then asked to determine the relationships between them. The collected data are then used to construct finetuning task and the small text encoder is optimized with a triplet loss. Our experimental results demonstrate the efficacy of the proposed method on three distinct NID datasets, surpassing all strong baselines in both unsupervised and semi-supervised settings. Our code can be found in https://github.com/floatSDSDS/LANID."
}
```

## Thanks
Some of the code was adapted from:

https://github.com/thuiar/DeepAligned-Clustering
https://github.com/wvangansbeke/Unsupervised-Classification
https://github.com/HobbitLong/SupContrast
https://github.com/fanolabs/NID_ACLARR2022

## Contact
Lu Fan complu.fan@connect.polyu.hk