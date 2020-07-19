import torch
from torch.utils.data import DataLoader

import gluonnlp
from gluonnlp.data import SentencepieceTokenizer

from tqdm import tqdm
import subprocess
import os
from tensorboardX import SummaryWriter
import re

from absl import app, logging
import argparse

from utils import get_tokenizer, download, tokenizer
from prepare_pytorch_kogpt2 import get_pytorch_kogpt2_model, load_kogpt2_model, load_kogpt2_model_from_checkpoint

# from kogpt2.model.torch_gpt2 import GPT2Config, GPT2LMHeadModel
# from kogpt2.data import Read_Dataset
# from kogpt2.model.sample import sample_sequence


# set required arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200, help="epoch 를 통해서 학습 범위를 조절합니다.")
parser.add_argument('--save_path', type=str, default='./checkpoint/', help="학습 결과를 저장하는 경로입니다.")
parser.add_argument('--load_path', type=str, default='./checkpoint/Alls/KoGPT2_checkpoint_296000.tar', help="학습된 결과를 불러오는 경로입니다.")
parser.add_argument('--samples', type=str, default="samples/", help="생성 결과를 저장할 경로입니다.")
parser.add_argument('--data_file_path', type=str, default='dataset/lyrics_dataset.txt', help="학습할 데이터를 불러오는 경로입니다.")
parser.add_argument('--batch_size', type=int, default=8, help="batch_size 를 지정합니다.")

args = parser.parse_args()



def auto_enter(text):
	text = (text.replace("   ", "\n"))
	text = text.split("\n")

	text = [t.lstrip() for t in text if t != '']
	return "\n\n".join(text)


def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')

    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))

    return gpu_memory_map



def main(_):
    print('Start main method for training KoGPT2_Lyricist')

    ctx = 'cpu' #'cuda'

    # generate summary writer to visualise and record the summary of training
    summary = SummaryWriter()

    # get arguments
    epoch, save_path, load_path, samples, data_file_path, batch_size = args.epoch, args.save_path, args.load_path, args.samples, args.data_file_path, args.batch_size

    # download and load KoGPT2 model and the vocabulary object
    # kogpt2model, vocab_b_obj = get_pytorch_kogpt2_model(ctx=ctx)
    kogpt2model, vocab_b_obj = load_kogpt2_model(ctx=ctx)

    device = torch.device(ctx)
    kogpt2model.to(device)

    # load checkpoints (if exists)
    kogpt2model, count = load_kogpt2_model_from_checkpoint(kogpt2model, load_path, device, ctx=ctx)
    kogpt2model.train()

    model, vocab = kogpt2model, vocab_b_obj

    print('Load model successfully')

    # get tokenizer
    tok_path = get_tokenizer()
    #TODO tok = SentencepieceTokenizer(tok_path,  num_best=0, alpha=0)
    tok = SentencepieceTokenizer(tok_path)

    #TODO dataset

    print('Get dataset successfully')


if __name__ == '__main__':
    app.run(main)
