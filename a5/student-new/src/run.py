import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
import random
import argparse
random.seed(0)

import dataset
import model
import trainer
import utils

# 创建解释器
argp = argparse.ArgumentParser()

# 添加参数
    # function 在这里是位置参数，必须按顺序输入
# choices 表明只有这三个参数可选
argp.add_argument('function',
    help="Whether to pretrain, finetune or evaluate a model",
    choices=["pretrain", "finetune", "evaluate"])
argp.add_argument('variant',
    help="Which variant of the model to run ('vanilla' or 'synthesizer')",
    choices=["vanilla", "synthesizer"])
# default 表明在没有出现这个参数时，默认是什么
argp.add_argument('pretrain_corpus_path',
    help="Path of the corpus to pretrain on", default=None)
# --reading_params_path 是可选参数，位置随意，也可没有
argp.add_argument('--reading_params_path',
    help="If specified, path of the model to load before finetuning/evaluation",
    default=None)
argp.add_argument('--writing_params_path',
    help="Path to save the model after pretraining/finetuning", default=None)
argp.add_argument('--finetune_corpus_path',
    help="Path of the corpus to finetune on", default=None)
argp.add_argument('--eval_corpus_path',
    help="Path of the corpus to evaluate on", default=None)
argp.add_argument('--outputs_path', default=None)

# 解析参数
args = argp.parse_args()

# Save the device
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# Keep the block size 128
# Why is the pretraining corpus always required (even if we're not pretraining?)
# It's because we're using it as a hack to always have the same vocabulary
# (that is, the same mapping from character to integer, and we build the 
# vocab from the pretraining corpus.)
block_size = 128
text = open(args.pretrain_corpus_path).read()
pretrain_dataset = dataset.CharCorruptionDataset(text, block_size)

# We don't suggest you change these hyperparameters, as they're known to work.
# use them for both the vanilla and the synthesizer models
mconf = model.GPTConfig(pretrain_dataset.vocab_size, pretrain_dataset.block_size,
    n_layer=4, n_head=8, n_embd=256)

"""
Don't change above here; write your code below
"""

if args.variant == 'vanilla':
    # model 只需要相关的参数就可以构建
    # 同时 model 比较有意思的是，创建的时候 my_model = model(configuration)
    # 调用 forward 的时候 loss = my_model(src, tgt) , model 类被实例化后，调用名字的函数就是调用 forward
    
    model = model.GPT(mconf)  # 等号右侧的 model 指的是 model 这个文件！
    
    # TODO [part c]: Make some model here
elif args.variant == 'synthesizer':
     # TODO [part g]: Make some other model here
    model = model.GPT_Synthesizer(mconf)
# From here on, your code should be identical independent of which
# variant (vanilla or synthesizer) has been chosen.

if args.function == 'pretrain':
    assert args.pretrain_corpus_path is not None
    assert args.writing_params_path is not None
    # TODO [part f]:
    # - Given:
    #     1. A corpus specified in args.pretrain_corpus_path
    #     2. An output path args.writing_params_path for the model parameters
    # - Goals:
    #     1. Pretrain the model on this corpus
    #     2. Save the resulting model in args.writing_params_path
    # - Make sure to use the following hyperparameters for pretraining:
    #     max_epochs=650
    #     batch_size=128
    #     learning_rate=6e-3
    #     lr_decay=True
    #     warmup_tokens=512*20
    #     final_tokens=200*len(pretrain_dataset)*block_size
    #     num_workers=4
    tconf = trainer.TrainerConfig(max_epochs = 650, batch_size = 128, learing_rate = 6e-3, lr_decay = True,
                                    warmup_tokens = 512*20, final_tokens=200*len(pretrain_dataset)*block_size, num_workers = 4)
    
    Trainer = trainer.Trainer(model = model, train_dataset = pretrain_dataset, test_dataset = None, config = tconf)
    Trainer.train()
    torch.save(model.state_dict(), args.writing_params_path)

elif args.function == 'finetune':
    assert args.writing_params_path is not None
    assert args.finetune_corpus_path is not None
    # TODO [part c] [part f]:
    # - Given:
    #     1. A finetuning corpus specified in args.finetune_corpus_path
    #     2. A path args.reading_params_path containing pretrained model
    #         parameters, or None if finetuning without a pretrained model
    #     3. An output path args.writing_params_path for the model parameters
    # - Goals:
    #     1. If args.reading_params_path is specified, load these parameters
    #         into the model
    #     2. Finetune the model on this corpus
    #     3. Save the resulting model in args.writing_params_path
    # - Make sure to use the following hyperparameters:
    #     Hyperparameters for finetuning WITHOUT a pretrained model:
    #         max_epochs=75
    #         batch_size=256
    #         learning_rate=6e-4
    #         lr_decay=True
    #         warmup_tokens=512*20
    #         final_tokens=200*len(pretrain_dataset)*block_size
    #         num_workers=4
    #     Hyperparameters for finetuning WITH a pretrained model:
    #         max_epochs=10
    #         batch_size=256
    #         learning_rate=6e-4
    #         lr_decay=True
    #         warmup_tokens=512*20
    #         final_tokens=200*len(pretrain_dataset)*block_size
    #         num_workers=4
    
    if args.reading_params_path:
        model.load_state_dict(torch.load(args.reading_params_path))
        tconf = trainer.TrainerConfig(max_epochs = 10, batch_size = 256, learing_rate = 6e-4, learing_decay = True, 
                                      warmup_tokens = 512*20, final_tokens=200*len(pretrain_dataset)*block_size, num_workers = 4)
        
    elif args.reading_params_path == None:
        # model 已经构建好了，现在要构建一个 trainer 。
        # model 接收 batched src 和 tgt 并进行 forward ，输出 loss 
        # 而一个 trainer 需要接受 model 和 unbatched src 和 tgt , 以及 batch_size, learn_rate 等 backward 的东西
        # 还有就是 optimizer 也是在 trainer 里面定义的
        # 所以要先对 trainer 进行配置参数，再传入模型和原始数据等
        tconf = trainer.TrainerConfig(max_epochs = 75, batch_size = 256, learing_rate = 6e-4, learing_decay = True, 
                                      warmup_tokens = 512*20, final_tokens=200*len(pretrain_dataset)*block_size, num_workers = 4)
        
    # open 接受文件路径，返回一个文件对象，read 读取文件所有内容并作为字符串返回
    corpus = open(args.finetune_corpus_path).read()
    
    # 其实在 GPT 模型中，我们想训练的 corpus 既是 src 又是 tgt
    # 这里对我们的原始的数据进行处理，使之能够被传入进模型进行训练
    finetune_dataset = dataset.NameDataset(pretrain_dataset, corpus)
    trainer = trainer.Trainer(model = model, train_dataset = finetune_dataset, test_dataset = None, config = tconf)
    trainer.train()
    
    # 以字典类型保存模型参数
    torch.save(model.state_dict(), args.writing_params_path)
    
elif args.function == 'evaluate':
    assert args.outputs_path is not None
    assert args.reading_params_path is not None
    assert args.eval_corpus_path is not None
    model.load_state_dict(torch.load(args.reading_params_path))
    correct = 0
    total = 0
    with open(args.outputs_path, 'w') as fout:
        predictions = []
        for line in tqdm(open(args.eval_corpus_path)):
            x = line.split('\t')[0]
            x = x + '⁇'
            x = torch.tensor([pretrain_dataset.stoi[s] for s in x], dtype=torch.long)[None,...].to(device)
            pred = utils.sample(model, x, 32, sample=False)[0]
            completion = ''.join([pretrain_dataset.itos[int(i)] for i in pred])
            pred = completion.split('⁇')[1]
            predictions.append(pred)
            fout.write(pred + '\n')
        total, correct = utils.evaluate_places(args.eval_corpus_path, predictions)
    if total > 0:
        print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))
    else:
        print('Predictions written to {}; no targets provided'
                .format(args.outputs_path))

