"""
main program for running internal pre-training and CLNN

some functions are modified from
https://github.com/thuiar/DeepAligned-Clustering/blob/main/DeepAligned.py
"""
import torch

from model import CLBert
from config import get_parser
from dataloader import Data
from mtp import InternalPretrainModelManager
from utils.tools import *
from LANID import LANID

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CLNNModelManager:
    """
    The implementation of Contrastive Learning with Nearest Neighbors
    """
    def __init__(self, args, data, pretrained_model=None):
        set_seed(args.seed)
        n_gpu = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = data.num_labels
        self.data = data
        self.model = CLBert(args.bert_model, device=self.device)
        self.evaluation(args, data, save_results=False)

        if not args.fast:
            if n_gpu > 1:
                self.model = nn.DataParallel(self.model)

            if not args.disable_pretrain:
                self.pretrained_model = pretrained_model
                self.load_pretrained_model()

        self.num_train_optimization_steps = int(len(data.train_semi_dataset) / args.train_batch_size) * args.num_train_epochs

        self.optimizer, self.scheduler = self.get_optimizer(args)

        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.generator = view_generator(self.tokenizer, args.rtr_prob, args.seed)

        self.lanid = LANID(args, data.train_semi_dataset, data.text_semi)

        self.train_dataloader = None
        self.label_all = None
        self.dist_all = None
        self.nearest_topk_dist, self.nearest_topk_ind = None, None

    def evaluation(self, args, data, save_results=True, plot_cm=True):
        """final clustering evaluation on test set"""
        # get features
        feats_test, labels = self.get_features_labels(data.test_dataloader, self.model, args)
        feats_test = feats_test.cpu().numpy()

        # k-means clustering
        km = KMeans(n_clusters = self.num_labels).fit(feats_test)

        y_pred = km.labels_
        y_true = labels.cpu().numpy()

        results = clustering_score(y_true, y_pred)
        print('results', results)

        # confusion matrix
        if plot_cm:
            ind, _ = hungray_aligment(y_true, y_pred)
            map_ = {i[0]:i[1] for i in ind}
            y_pred = np.array([map_[idx] for idx in y_pred])

            cm = confusion_matrix(y_true,y_pred)
            # print('confusion matrix',cm)
            self.test_results = results

        # save results
        if save_results:
            self.save_results(args)

    def update_dist_topk(self, data, model, k=50):
        feat_epoch = []
        label_epoch = []
        for batch in tqdm(data.train_semi_dataloader, desc="update_dist_topk"):
            ind_batch, label_batch = batch[4], batch[5]
            label_epoch.append(label_batch)
            inp_batch = tuple(t.to(self.device) for t in batch)
            x_batch = {
                "input_ids": inp_batch[0],
                "attention_mask": inp_batch[1],
                "token_type_ids": inp_batch[2]
            }
            feat_batch = model(x_batch)['features']
            feat_epoch.append(feat_batch.clone().detach().cpu())
        if not label_epoch:
            self.label_all = torch.cat(label_epoch)
        emb_epoch_full = torch.cat(feat_epoch)
        self.dist_all, self.nearest_topk_dist, self.nearest_topk_ind = cdist_memory_efficient(
            emb_epoch_full, bsz=64, device=self.device, topk=k)
        del emb_epoch_full

    def _get_x(self, inp_batch=None, dataset=None, ind_batch=None, aug="none"):
        if inp_batch:
            inp_batch = tuple(t.to(self.device) for t in inp_batch)
            ind_batch = inp_batch[4]
        else:
            inp_batch = dataset[ind_batch]
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
            raise NotImplementedError(f"View strategy {args.view_strategy} not implemented!")
        return ind_batch, x_batch

    def train(self, args, data):

        if isinstance(self.model, nn.DataParallel):
            criterion = self.model.module.loss_cl
        else:
            criterion = self.model.loss_cl

        self.train_dataloader = DataLoader(
            data.train_semi_dataset, batch_size=args.train_batch_size, shuffle=True)
        self.update_dist_topk(data, self.model, args.topk)

        if not args.fast:
            for epoch in trange(int(args.num_train_epochs), desc="Epoch-CLNN"):
                self.model.train()
                ind_epoch_full, emb_epoch = self.fit_a_epoch(criterion)
                # if ((epoch + 0) % args.update_step_cl) == 0:
                #     self.lanid.fit(
                #         args.sampling, ind_epoch_full, feats=emb_epoch, dist=self.dist_all,
                #         topk_dist=self.nearest_topk_dist, topk_ind=self.nearest_topk_ind,
                #         labels_all=self.data.semi_label_real
                #     )
                self.update_dist_topk(data, self.model, args.topk)
                self.evaluation(args, data, save_results=False)

        for epoch in trange(int(args.gpt_epoch), desc="Epoch-LANID"):
            self.model.train()
            if ((epoch + 0) % args.update_step_gpt) == 0:
                ind_epoch_full, emb_epoch = self.iter_a_epoch()
                self.update_dist_topk(data, self.model, args.topk)
                self.lanid.fit(
                    args.sampling, ind_epoch_full, feats=emb_epoch, dist=self.dist_all,
                    topk_dist=self.nearest_topk_dist, topk_ind=self.nearest_topk_ind,
                    labels_all=self.data.semi_label_real
                )
            self.lanid.fit_a_epoch(self.model, self.optimizer, self.scheduler, criterion)
            self.evaluation(args, data, save_results=False)

    def iter_a_epoch(self):
        ind_epoch = []
        feat_epoch = []
        for batch in self.train_dataloader:
            ind_batch = batch[4]
            ind_epoch.append(ind_batch.detach().cpu())
            ind_batch, x_batch = self._get_x(inp_batch=batch, aug=args.view_strategy)
            feat_batch = self.model(x_batch)["features"]
            feat_epoch.append(feat_batch.detach().cpu())
        ind_epoch_full = torch.cat(ind_epoch)
        emb_epoch = torch.cat(feat_epoch)
        return ind_epoch_full, emb_epoch

    def fit_a_epoch(self, criterion):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        ind_epoch = []
        feat_epoch = []
        for batch in self.train_dataloader:
            with torch.set_grad_enabled(True):
                ind_batch = batch[4]
                ind_epoch.append(ind_batch.detach().cpu())

                nearest_ind_batch = self.nearest_topk_ind[ind_batch, 1:]
                adj = get_adjacency(ind_batch, nearest_ind_batch, batch[3])

                idx_sampling = torch.randint(low=0, high=args.topk, size=ind_batch.shape)
                ind_sampling = nearest_ind_batch[torch.arange(ind_batch.shape[0]), idx_sampling]

                ind_batch, x_batch = self._get_x(inp_batch=batch, aug=args.view_strategy)
                ind_nei, x_nei = self._get_x(
                    dataset=data.train_semi_dataset, ind_batch=ind_sampling, aug="rtr")
                # adj = self.construct_adj(ind_batch, ind_nei, batch[3].to(self.device))
                # adj = get_adjacency(ind_batch, ind_nei, batch[3].to(self.device))

                feat_batch = self.model(x_batch)["features"]
                feat_nei = self.model(x_nei)["features"]
                f_pos = torch.stack([feat_batch, feat_nei], dim=1)
                loss = criterion(f_pos, mask=adj, temperature=args.temp)

                tr_loss += loss.item()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                nb_tr_examples += ind_batch.shape[0]
                nb_tr_steps += 1

                # store data for dbscan gpt
                feat_epoch.append(feat_batch.detach().cpu())
                ind_epoch_full = torch.cat(ind_epoch)
                emb_epoch = torch.cat(feat_epoch)
        loss = tr_loss / nb_tr_steps
        print('nid loss', loss)
        return ind_epoch_full, emb_epoch

    def get_optimizer(self, args):
        num_warmup_steps = int(args.warmup_proportion*self.num_train_optimization_steps)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=self.num_train_optimization_steps)
        return optimizer, scheduler

    def load_pretrained_model(self):
        """load the backbone of pretrained model"""
        if isinstance(self.pretrained_model, nn.DataParallel):
            pretrained_dict = self.pretrained_model.module.backbone.state_dict()
        else:
            pretrained_dict = self.pretrained_model.backbone.state_dict()
        if isinstance(self.model, nn.DataParallel):
            self.model.module.backbone.load_state_dict(pretrained_dict, strict=False)
        else:
            self.model.backbone.load_state_dict(pretrained_dict, strict=False)

    def get_features_labels(self, dataloader, model, args):
        model.eval()
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Extracting representation"):
            batch = tuple(t.to(self.device) for t in batch)
            # input_ids, input_mask, segment_ids, label_ids = batch
            input_ids, input_mask, segment_ids, label_ids, ind_all = batch
            X = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
            with torch.no_grad():
                feature = model(X, output_hidden_states=True)["hidden_states"]

            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))

        return total_features, total_labels

    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [args.dataset, args.method, args.known_cls_ratio, args.labeled_ratio, args.topk, args.view_strategy, args.seed]
        names = ['dataset', 'method', 'known_cls_ratio', 'labeled_ratio', 'topk', 'view_strategy', 'seed']
        vars_dict = {k:v for k,v in zip(names, var)}
        results = dict(self.test_results,**vars_dict)
        keys = list(results.keys())
        values = list(results.values())

        file_name = args.file_result
        results_path = os.path.join(args.save_results_path, file_name)

        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori,columns=keys)
            df1.to_csv(results_path,index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results,index=[1])
            try:
                df1 = df1.append(new,ignore_index=True)
            except:
                df1 = pd.concat([df1, new], ignore_index=True)
            df1.to_csv(results_path, index=False)
        data_diagram = pd.read_csv(results_path)

        print('test_results', data_diagram)


if __name__ == '__main__':

    print('Data and Parameters Initialization...')
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    if args.fast:
        data = Data(args)
        manager = CLNNModelManager(args, data)
    else:
        if args.known_cls_ratio == 0:
            args.disable_pretrain = True # disable internal pretrain
        else:
            args.disable_pretrain = False

        if not args.disable_pretrain:
            data = Data(args)
            print('Pre-training begin...')
            manager_p = InternalPretrainModelManager(args, data)
            manager_p.train(args, data)
            print('Pre-training finished!')
            manager = CLNNModelManager(args, data, manager_p.model)  # pass the model to clnn
        else:
            data = Data(args)
            manager = CLNNModelManager(args, data)

        if args.report_pretrain:
            method = args.method
            args.method = 'pretrain'
            manager.evaluation(args, data)  # evaluate when report performance on pretrain
            args.method = method

    print('Training begin...')
    manager.train(args, data)
    print('Training finished!')

    print('Evaluation begin...')
    manager.evaluation(args, data)
    print('Evaluation finished!')

    print('Saving Model ...')
    if args.save_model_path:
        manager.model.save_backbone(args.save_model_path)

    print(f'gpt cost: {manager.gpt4nid.text_gen_api.cost_estimator.cost}')
    print("Finished!")