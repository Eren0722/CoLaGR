from utils.utils import *
from utils.dataset import load_datasets
import argparse
from LightCodeTokenizer import LightCodeTokenizer
from typing import List
import os

def save_vocab(tokens: List[str], save_path: str):
    vocab = {tok: idx for idx, tok in enumerate(tokens)}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    return vocab

parser = argparse.ArgumentParser(description='LLMRec')
parser = parse_global_args(parser)
parser = parse_dataset_args(parser)
parser = parse_train_args(parser)
args = parser.parse_args()

train_data, valid_data = load_datasets(args)

new_tokens = train_data.get_new_tokens()
save_dir = os.path.join(args.tokenizer_save_dir, args.dataset)
# make directory if not exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

special_tokens = ["<pad>", "</s>", "<unk>"]


all_tokens = special_tokens + sorted(set(new_tokens))

vocab = save_vocab(all_tokens, save_dir+"/vocab.json")

tokenizer = LightCodeTokenizer(
    vocab_file=f"{save_dir}/vocab.json",
    pad_token="<pad>",
    eos_token="</s>",
    unk_token="<unk>"
)

# save
tokenizer.save_pretrained(save_dir)


# test
# test_text = "<a_174><b_78><c_178><d_225>"
# encoded = tokenizer(test_text, add_special_tokens=True)
# print(encoded['input_ids'])
# print("Tokens:", tokenizer.convert_ids_to_tokens(encoded['input_ids']))
# print("Encoded:", encoded)