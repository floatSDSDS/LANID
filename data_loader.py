# from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset
import sentence_transformers.datasets as td
from sentence_transformers.readers import InputExample

from utils.tools import *
from dataloader import Data

max_seq_lengths = {
    'mcid': 21,
    'clinc': 30,
    'stackoverflow': 45,
    'banking': 55
}


class DataSentBert(Data):

    def __init__(self, args):
        super(DataSentBert, self).__init__(args)
        self.semi_dl = None
        self.train_dl = None
        self.eval_dl = None
        self.test_dl = None

    def update_dl(self, args, model):
        self.semi_dl = self.get_semi_dl(args, model)
        self.train_dl = self.get_dl(self.train_labeled_examples, model, args, 'train')
        self.eval_dl = self.get_dl(self.eval_examples, model, args, 'eval')
        self.test_dl = self.get_dl(self.test_examples, model, args, 'test')

    def get_semi_dl(self, args, model):
        examples = self.train_labeled_examples + self.train_unlabeled_examples
        labels = self.semi_label_ids
        example_lst = []
        for i, e in enumerate(examples):
            example_lst.append(InputExample(
                texts=[e.text_a], label=int(labels[i])
            ))
        # semi_dataset = SentencesDataset(example_lst, model)
        semi_dataset = td.SentenceLabelDataset(example_lst)
        semi_dl = DataLoader(semi_dataset, shuffle=True, batch_size=args.train_batch_size)
        return semi_dl

    def get_dl(self, examples, model, args, mode='train'):
        if mode == 'train' or mode == 'eval':
            features = convert_examples_to_features(examples, self.known_label_list)
        elif mode == 'test':
            features = convert_examples_to_features(examples, self.all_label_list)
        else:
            raise NotImplementedError(f"Mode {mode} not found")

        dataset = td.SentencesDataset(features, model)
        if mode == 'train':
            sampler = RandomSampler(dataset)
            dl = DataLoader(dataset, sampler=sampler, batch_size=args.pretrain_batch_size)
        elif mode in ["eval", "test"]:
            sampler = SequentialSampler(dataset)
            dl = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size)
        else:
            raise NotImplementedError(f"Mode {mode} not found")
        return dl


def convert_examples_to_features(examples, label_list):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i
    features = []
    for (ex_index, example) in enumerate(examples):
        txt = [example.text_a]
        label_id = label_map[example.label]
        features.append(InputExample(texts=txt, label=label_id))
    return features

