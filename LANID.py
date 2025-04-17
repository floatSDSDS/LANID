import numpy as np
import pandas as pd
import torch

# try:
#     from fxnlprchatzoo import TextGenViaApi
# except:
#     pass

from HfApi import TextGenViaApi
from utils.tools import *
from utils.dbscan import MyDBSCAN
from utils.contrastive import ContraLoss
from scipy import sparse


class LANID:
    def __init__(self, args, dataset, text_all):
        set_seed(args.seed)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_name = args.dataset
        self.dataset = dataset  # train_semi_dataset
        self.text_all = text_all  # text_semi
        self.prompt = load_prompts_dataset(args.path_prompt, self.data_name)[0]
        self.size = len(text_all)

        # gpt
        self.text_gen_api = TextGenViaApi(
            huggingface_authorize_token='',
            openai_api_key='',
        )

        # prompt settings
        self.path_prompt = args.path_prompt
        self.gpt_model_name = args.gpt_model
        self.gpt_temperature = args.gpt_temp
        self.gpt_p = args.gpt_p
        self.gpt_max_new_tokens = args.gpt_max_tk

        # sampling
        self.strategy = args.sampling
        self.sample_k = args.sample_k
        self.k_pos = args.k_pos
        self.k_neg = args.k_neg  # random negative sampling
        self.p_core = args.p_core
        self.p_outlier = args.p_outlier
        # self.adj = torch.sparse_coo_tensor(size=(len(self.dataset), len(self.dataset)))
        self.adj = None

        # text
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.generator = view_generator(self.tokenizer, args.rtr_prob, args.seed)
        self.view_strategy = args.view_strategy

        # dbscan
        self.eps = 0
        self.minpts = args.minpts
        self.dbscan_q = args.dbscan_q
        self.dbscan = MyDBSCAN(self.eps, self.minpts)

        # training
        self.train_task = args.train_task
        self.batch_gpt = args.batch_gpt
        self.data_gpt = dict(size=0,
                             idx=torch.tensor([], dtype=torch.int64),
                             pos=torch.tensor([], dtype=torch.int64),
                             relation=torch.tensor([], dtype=torch.int64),
                             text_input=[], response=[]
                             )
        if args.known_cls_ratio > 0 and args.use_known:
            ind_known, val_known = self.get_semi_sparse()
            self.data_gpt['idx'] = torch.cat([self.data_gpt['idx'], ind_known[0, :]])
            self.data_gpt['pos'] = torch.cat([self.data_gpt['pos'], ind_known[1, :]])
            self.data_gpt['relation'] = torch.cat([self.data_gpt['relation'], val_known])

        self.max_gpt_ep = args.max_gpt_ep
        self.tr_dl_gpt = None

        # losses
        self.contra_loss = ContraLoss(device=self.device, temperature=args.temperature)
        self.tri_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def fit(self, sampling, indices, feats=None,
            dist=None, topk_dist=None, topk_ind=None, labels_all=None):

        # 1. sample data, construct sampling pairs
        idx, pos = self.sample_manager(sampling, indices, feats, dist, topk_dist, topk_ind)
        # 2. get gpt predictions
        text_input, responses, relation = self.get_response(idx, pos)
        self.evaluate_relation(idx, pos, relation, labels_all)
        # 3. construct dataloader
        self.update_data(idx, pos, text_input, responses, relation)

    def sample_manager(self, strategy, indices, feats=None,
                       dist=None, topk_dist=None, topk_ind=None):
        """return [a, b] positive data pairs"""
        if strategy == "near":
            idx, pos = self._sample_near(indices, topk_ind)
        elif strategy == "dbscan":
            idx, pos = self._sample_dbscan(indices, feats, dist, topk_dist)
        elif strategy == "both":
            idx_near, pos_near = self._sample_near(indices, topk_ind)
            idx_db, pos_db = self._sample_dbscan(indices, feats, dist, topk_dist)
            idx = torch.cat([idx_near, idx_db])
            pos = torch.cat([pos_near, pos_db])
        else:
            raise NotImplementedError(f"sampling strategy {strategy} not implemented!")
        return idx, pos

    def get_response(self, idx, pos):
        text_idx_all = self.text_all[idx]
        text_pos_all = self.text_all[pos]
        text_input = [self.prompt.format(
            sen_A=a, sen_B=b) for a, b in zip(text_idx_all, text_pos_all)
        ]
        responses = self.text_gen_api.llm_gen_concurrent(text_input,
                                                    self.gpt_model_name,
                                                    verbose=1,
                                                    time_out=20,
                                                    temperature=self.gpt_temperature,
                                                    top_p_value=self.gpt_p,
                                                    max_new_tokens=self.gpt_max_new_tokens,
                                                    max_workers=40)
        responses = [res.lower() if res else '' for res in responses]
        relation = torch.tensor([1 if 'yes' in tmp else 0 for tmp in responses])
        return text_input, responses, relation

    def update_data(self, idx, pos, text_input, responses, relation):

        self.data_gpt['idx'] = torch.cat((self.data_gpt['idx'], idx))
        self.data_gpt['pos'] = torch.cat((self.data_gpt['pos'], pos))
        self.data_gpt['relation'] = torch.cat((self.data_gpt['relation'], relation))
        self.data_gpt['text_input'].extend(text_input)
        self.data_gpt['response'].extend(responses)

        self.data_gpt['size'] = len(self.data_gpt['text_input'])

        if self.train_task == "cl":
            ind_eye = torch.cat([
                torch.arange(self.size).unsqueeze(0), torch.arange(self.size).unsqueeze(0)
            ], dim=0)
            val_eye = torch.ones(self.size)
            ind_all = torch.cat([self.data_gpt['idx'].unsqueeze(0), self.data_gpt['pos'].unsqueeze(0)])
            val_all = self.data_gpt['relation']
            self.adj = torch.sparse_coo_tensor(
                torch.cat([ind_eye, ind_all], dim=1),
                torch.cat([val_eye, val_all]),
                [self.size, self.size]
            )

        # sampling data
        ind_select = self.data_gpt['relation'] > 0
        idx_select = self.data_gpt['idx'][ind_select]
        pos_select = self.data_gpt['pos'][ind_select]
        relation_select = self.data_gpt['relation'][ind_select]
        n_sample = idx_select.shape[0]

        if n_sample > self.max_gpt_ep:
            ind_shuffle = torch.randperm(idx_select.shape[0])[:self.max_gpt_ep]
            idx_select = idx_select[ind_shuffle]
            pos_select = pos_select[ind_shuffle]
            relation_select = relation_select[ind_shuffle]
            n_sample = self.max_gpt_ep

        neg = np.random.choice(np.arange(self.size), size=(n_sample, self.k_neg), replace=True)
        neg = torch.tensor(neg)

        label_idx_select = self.dataset[idx_select][3]
        label_pos_select = self.dataset[pos_select][3]
        dataset = TensorDataset(
            idx_select, pos_select, neg,
            label_idx_select, label_pos_select, relation_select
        )
        self.tr_dl_gpt = DataLoader(dataset, batch_size=self.batch_gpt, shuffle=True)

    def get_semi_sparse(self):
        """return index and value for known data"""
        semi_labels = self.dataset.tensors[3]
        semi_idx = self.dataset.tensors[4]
        ind_known = semi_labels > 0
        idx_known = semi_idx[ind_known]
        label_knwon = semi_labels[ind_known]
        ind_truth_lst = []
        for l in label_knwon.unique():
            ind_l = label_knwon == l
            idx_l = idx_known[ind_l]
            idx_l_lst = [torch.cat([
                i.expand(idx_l.shape[0]).unsqueeze(0), idx_l.unsqueeze(0)
            ], dim=0) for i in idx_l]
            ind_truth_lst.append(torch.cat(idx_l_lst, dim=1))
        ind_truth = torch.cat(ind_truth_lst, dim=1)
        ind_keep = ind_truth[:, ind_truth[0, :] != ind_truth[1, :]]
        val_truth = torch.ones(ind_keep.shape[1])
        return ind_keep, val_truth

    def evaluate_relation(self, idx, pos, relation, labels):
        label_idx = labels[idx]
        label_pos = labels[pos]
        match_flag = (label_idx == label_pos).long()
        acc = (relation == match_flag).sum() / pos.shape[0]
        cm = confusion_matrix(relation.flatten(), match_flag.flatten())
        precision = cm[1][1] / sum(cm[1])
        print(f'acc_gpt: {acc:f}, precision_gpt: {precision:f}')
        print(cm)

    def fit_a_epoch(self, model, optimizer, scheduler, criterion=None):
        fun_fitting = getattr(self, f'_fit_a_epoch_{self.train_task}')
        fun_fitting(model, optimizer, scheduler, criterion)

    def _fit_a_epoch_pair(self, model, optimizer, scheduler, criterion=None):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for batch in self.tr_dl_gpt:
            # batch = tuple(t.to(self.device) for t in batch)
            idx, pos, neg = batch[0], batch[1], batch[2]

            idx = idx.expand_as(neg.t()).flatten()
            pos = pos.expand_as(neg.t()).flatten()
            neg = neg.t().flatten()

            idx, pos, neg = idx.to(self.device), pos.to(self.device), neg.to(self.device)

            x_idx = self._get_x(idx, aug='none')
            x_pos = self._get_x(pos, aug='none')
            x_neg = self._get_x(neg, aug='none')

            with torch.set_grad_enabled(True):
                feat_idx = model(x_idx)["features"]
                feat_pos = model(x_pos)["features"]
                feat_neg = model(x_neg)["features"]

                loss = self.tri_loss(feat_idx, feat_pos, feat_neg)
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                nb_tr_examples += idx.shape[0]
                nb_tr_steps += 1

                tr_loss += loss
        loss = tr_loss / nb_tr_steps
        print('gpt loss', loss)

    def _fit_a_epoch_cl(self, model, optimizer, scheduler, criterion):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for batch in self.tr_dl_gpt:
            # batch = tuple(t.to(self.device) for t in batch)
            idx, pos, relation = batch[0], batch[1], batch[5]
            adj, x = self._construct_batch(idx, pos, relation)
            feat1 = model(x)['features']
            feat2 = model(x)['features']
            feat = torch.stack([feat1, feat2], dim=1)
            # loss = criterion(feat, mask=adj)
            loss = self.contra_loss(feat, adj)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            nb_tr_examples += idx.shape[0]
            nb_tr_steps += 1

        loss = tr_loss / nb_tr_steps
        print('gpt loss', loss)

    def _construct_batch(self, idx, pos, relation=None):
        ind_batch = torch.cat([idx, pos])
        ind_unique, position_ind = ind_batch.unique(return_inverse=True)

        # with only current data
        size_batch = relation.shape[0]
        position_anchor, position_pair = position_ind[:size_batch], position_ind[size_batch:]
        size_unique = ind_unique.shape[0]
        adj = torch.sparse_coo_tensor(
            torch.stack((position_anchor, position_pair), dim=0),
            relation, size=(size_unique, size_unique)).to_dense()
        r_self = torch.eye(size_unique, dtype=adj.dtype)
        adj = adj + r_self
        adj = adj.to(self.device)

        # adj_rows = torch.index_select(self.adj, 0, ind_unique)
        # adj = torch.index_select(adj_rows, 1, ind_unique).to_dense()
        # adj = adj.to(self.device)

        x = self._get_x(ind_unique, aug='none')

        return adj, x

    def _sample_near(self, indices, topk_ind):
        """
            1. 抽样p%的数据点, use p_core as p
            2. 为每个点在k近邻中抽样k_pos个pos pair
        """
        n, topk = topk_ind.shape[0], topk_ind.shape[1] - 1
        n_sample = int(n * self.p_core)
        ind_sampled = indices[torch.randperm(n)[:n_sample]]
        nearest_ind_batch = topk_ind[ind_sampled, 1:]

        idx_repeat = []
        pos_sampling = []
        for i in range(self.k_pos):
            idx_sampling = torch.randint(low=0, high=topk, size=ind_sampled.shape)
            pos_sampling.append(nearest_ind_batch[torch.arange(n_sample), idx_sampling].unsqueeze(1))
            idx_repeat.append(ind_sampled)
        idx_repeat = torch.cat(idx_repeat, dim=0)
        pos_sampling = torch.cat(pos_sampling, dim=0).squeeze(1)
        return idx_repeat, pos_sampling

    def _sample_dbscan(self, indices, feats, dist, topk_dist, nearest=False):
        """
            1. update dbscan parameters with topk_dist
            2. dbscan fitting
            3. sample p% data with dbscan fitting results
        """
        dist_minpts = topk_dist[:, self.minpts]
        eps = torch.quantile(dist_minpts, self.dbscan_q)
        self.dbscan.update_param(eps=eps)

        indices_sort = indices.sort()[1]
        feats_sort = feats[indices_sort]

        self.dbscan.fit(feats_sort)
        core_all = torch.tensor([pt['idx'] for pt in self.dbscan.data if pt['type'] == 1])
        others_all = torch.tensor([pt['idx'] for pt in self.dbscan.data if pt['type'] != 1])

        n_sample = int(others_all.shape[0] * self.p_core)
        others_sampled = others_all[torch.randperm(others_all.shape[0])[:n_sample]]
        dist_others_cores = dist[others_sampled, :][:, core_all]

        dist_selected = dist_others_cores.clone().detach().to(self.device)
        dist_sel_topk, ind_sel_topk = torch.topk(dist_selected, k=self.sample_k, largest=False)
        cores_sample_pool = core_all[ind_sel_topk]

        if nearest:
            cores_selected = cores_sample_pool[:, :self.k_pos]
        else:
            cores_selected = torch.cat([
                c[torch.randperm(self.sample_k)][:self.k_pos].unsqueeze(0) for c in cores_sample_pool],
                dim=0)

        idx_repeat = []
        pos_sampling = []
        for i in range(self.k_pos):
            idx_repeat.append(others_sampled)
            pos_sampling.append(cores_selected[:, i])
        idx_repeat = torch.cat(idx_repeat, dim=0)
        pos_sampling = torch.cat(pos_sampling, dim=0)
        return idx_repeat, pos_sampling

    def _get_x(self, ind_batch=None, aug="rtr"):

        inp_batch = self.dataset[ind_batch]
        inp_batch = tuple(t.to(self.device) for t in inp_batch)

        if aug == "rtr":
            x_batch = {
                "input_ids": self.generator.random_token_replace(inp_batch[0].cpu()).to(self.device),
                "attention_mask": inp_batch[1],
                "token_type_ids": inp_batch[2]
            }
        elif aug == "shuffle":
            x_batch = {
                "input_ids": self.generator.shuffle_tokens(inp_batch[0].cpu()).to(self.device),
                "attention_mask": inp_batch[1],
                "token_type_ids": inp_batch[2]
            }
        elif aug == "none":
            x_batch = {
                "input_ids": inp_batch[0],
                "attention_mask": inp_batch[1],
                "token_type_ids": inp_batch[2]
            }
        else:
            raise NotImplementedError(f"View strategy {self.view_strategy} not implemented!")
        return x_batch

