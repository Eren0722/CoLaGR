from utils.data import SeqRecDataset, NegSampleOnEpochDataset

def load_datasets(args):
    train_data = SeqRecDataset(args, mode="train", sample_num=args.train_sample_num)
    valid_data = SeqRecDataset(args, mode="valid", sample_num=args.eval_sample_num)

    return train_data, valid_data

def load_neg_sample_on_epoch_datasets(args, tokenizer):
    train_data = NegSampleOnEpochDataset(args, mode="train", sample_num=args.train_sample_num, sampling=args.sampling, neg_k=args.neg_k, tokenizer=tokenizer)
    valid_data = NegSampleOnEpochDataset(args, mode="valid", sample_num=args.eval_sample_num, sampling=args.sampling, neg_k=args.neg_k, tokenizer=tokenizer)
    return train_data, valid_data

def load_test_dataset(args):
    test_data = SeqRecDataset(args, mode="test", sample_num=args.sample_num)
    return test_data
