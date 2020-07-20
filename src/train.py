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

from dataset import KoLyricsDataset
from utils import get_tokenizer, download, tokenizer
from prepare_pytorch_kogpt2 import get_pytorch_kogpt2_model, load_kogpt2_model, load_kogpt2_model_from_checkpoint

# from kogpt2.model.torch_gpt2 import GPT2Config, GPT2LMHeadModel
# from kogpt2.data import Read_Dataset
# from kogpt2.model.sample import sample_sequence


# set required arguments
parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=200, help="The number of epochs.")
parser.add_argument('--batch_size', type=int, default=8, help="The batch size.")
parser.add_argument('--lr', type=float, default=0.01, help="Learning Rate")

parser.add_argument('--save_path', type=str, default='../checkpoint/', help="File path to store checkpoints")
parser.add_argument('--load_path', type=str, default='../checkpoint/Alls/KoGPT2_checkpoint_296000.tar', help="File path to load checkpoints")

parser.add_argument('--samples', type=str, default="samples/", help="File path to store generated texts")
parser.add_argument('--data_file_path', type=str, default='../dataset/lyrics_dataset.txt', help="File path of the dataset")

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

    dataset = KoLyricsDataset(data_file_path, vocab, tok)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    print('Get dataset successfully')

    learning_rate = args.lr
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print('KoGPT-2 Start Transfer Learning')
    avg_loss = (0.0, 0.0)


    # start transfer learning
    for epoch in range(epoch):
        for data in data_loader:
            optimizer.zero_grad()

            data = torch.stack(data) # list of Tensor로 구성되어 있기 때문에 list를 stack을 통해 변환해준다.
            data = data.transpose(1,0)
            data = data.to(ctx)
            
            model = model.to(ctx)

            outputs = model(data, labels=data)
            loss, logits = outputs[:2]
            loss = loss.to(ctx)
            loss.backward()
            avg_loss = (avg_loss[0] * 0.99 + loss, avg_loss[1] * 0.99 + 1.0)
            optimizer.step()

            if count % 10 == 0:
                print('epoch no.{0} train no.{1}  loss = {2:.5f} avg_loss = {3:.5f}' . format(epoch, count, loss, avg_loss[0] / avg_loss[1]))
                summary.add_scalar('loss/avg_loss', avg_loss[0] / avg_loss[1], count)
                summary.add_scalar('loss/loss', loss, count)
            

            
            if (count > 0 and count % 1000 == 0) or (len(data) < batch_size):
                sent = sample_sequence(model.to("cpu"), tok, vocab, sent="사랑", text_size=100, temperature=0.7, top_p=0.8, top_k=40)
                sent = sent.replace("<unused0>", "\n")
                print(sent)

                summary.add_text('Text', sent, count)

                if count > 500000:
                    now = [int(n) for n in os.listdir(samples)]
                    now = max(now)
                    f = open(samples + str(now + 1), 'w', encoding="utf-8")
                    f.write(sent)
                    f.close()
            #########################################

            count += 1


            # check if the program needs to save model
            if (count > 0 and count % 10000 == 0) or (len(data) < batch_size):
                try:
                    torch.save({
                        'epoch': epoch,
                        'train_no': count,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                    }, save_path + 'KoGPT2_checkpoint_' + str(count) + '.tar')
                except:
                    pass




if __name__ == '__main__':
    app.run(main)
