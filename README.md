# KoGPT2_Lyricist

Training a [KoGPT2](https://github.com/SKT-AI/KoGPT2) model with K-Pop lyrics.

## KoGPT2

[KoGPT2](https://github.com/SKT-AI/KoGPT2) is a pretrained model, which is pretraiend by SKT-AI team. It is trained with about 20GB of korean text data.

## Sample data

Basically, this repository uses the <|endoftext|> token to represent the end of th text. You could find some example dataset in [here](./dataset/lyrics_dataset.txt), which contains the lyrics of [IU](https://g.co/kgs/s7yG1b)'s songs. Clearly, the example data does not have enough amount of texts to retrain the gpt-2 model, thus, please make your own dataset before running the training process.

## Credits

Most of the codes in this repository are from [forus-ai's repository](https://github.com/forus-ai/KoGPT2-Lyrics-Generation-FineTuning-Version1). Credits to [forus-ai](https://github.com/forus-ai).

## Reference

- [KoGPT2](https://github.com/SKT-AI/KoGPT2)
- [KoGPT2-Lyrics-Generation-FineTuning-Version1](https://github.com/forus-ai/KoGPT2-Lyrics-Generation-FineTuning-Version1)
