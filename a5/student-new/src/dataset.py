import random
import torch
from torch.utils.data import Dataset
import argparse
import numpy as np

"""
The input-output pairs (x, y) of the NameDataset are of the following form:

  x: Where was Khatchig Mouradian born?⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  x: Where was Jacob Henry Studer born?⁇Columbus⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Columbus⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

感觉 PAD_CHAR 在这里就是通过规范输出格式来引导神经网络按我们想要的格式输出
Using the PAD_CHAR characters in y before the ⁇[place] keeps the trainer from
optimizing the model to predict the question, "Where was...".

这里必须载入 pretraining_dataset 是为了获取它的词汇表的范式，使得我们自己的数据构建出来的东西
和之前 pretrain 时候输入的数据一样，避免了和模型不兼容的问题
Note that the NameDataset should take the pretraining_dataset defined in run.py
as an input. This is to allow the vocab specification of the NameDataset to be
the same as that of the pretraining dataset.



You don't need to implement anything in NameDataset.
"""

class NameDataset(Dataset):
    def __init__(self, pretraining_dataset, data):
        self.MASK_CHAR = u"\u2047" # the doublequestionmark character, for mask
        self.PAD_CHAR = u"\u25A1" # the empty square character, for pad
        # itos 意思是 index to string
        self.itos = pretraining_dataset.itos 
        self.stoi = pretraining_dataset.stoi 
        self.block_size = pretraining_dataset.block_size
        self.data = list(data.encode('utf-8').decode('ascii', errors='ignore').split('\n'))

    def __len__(self):
        # returns the length of the dataset
        return len(self.data) - 1

    # 用索引获取元素，例: my_data[2]
    def __getitem__(self, idx):
        inp, oup = self.data[idx].split('\t')
        x = inp + self.MASK_CHAR + oup + self.MASK_CHAR
        x = x + self.PAD_CHAR*(self.block_size - len(x))
        y = self.PAD_CHAR*(len(inp)-1) + x[len(inp):]
        
        x = x[:-1]
        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)
        return x, y


"""
[part e]

# 这一部分实现的 T5 模型有点像 NMT 的感觉，利用 encoder 充分获取前半部分的语义
# 用 decoder 来按指定的格式对前半部分 masked 进行补全(不定长)
# intuition :  如果能根据上下文补全句子中缺少的部分，那就说明对于语言有很强的理解能力，类比完形填空
# 而且 T5 在 QA 上的表现也会比较好，因为回答问题其实本质上也是一种填空  

Write a class that yields examples of a simplified span corruption objective.
Do not change the signature of the __init__ or __getitem__ functions.

Make sure to implement the full spec for full credit -- we list below the
criteria that must be satisfied for a full implementation.

--------------
Vocabulary Specification(说明)

Your vocabulary is to be accessible via two dictionaries:

  self.stoi: a dictionary from characters in the vocabulary to indices of type
      int
  self.itos: a dictionary from indices of type int to characters in the
      vocabulary

Your vocabulary must have the following form: 

  Identifier 0 must be assigned to the unicode element u"\u25A1".
      This is the empty_square_character.
      Further, let self.PAD_CHAR = u"\u25A1"
  Identifier 1 must be assigned to the unicode element u"\u2047".
      This is the doublequestionmark character, which we'll use
      as a sentinel to represent that text is missing from the input
      Further, let self.MASK_CHAR = u"\u2047"
  Identifiers 2, ..., len(self.itos)-1 should be the sorted list of characters
      that appear in the data argument.

--------------
Masking Specification

The __getitem__ function takes an index and returns a data point (x, y) where
x and y are Long tensors of length self.block_size. x encodes the input
sequence, and y encodes the output sequence.

0. Use the idx argument of __getitem__ to retrieve the element of self.data
at the given index. We'll call the resulting data entry a document.

1. Randomly truncate the document to a length no less than 4 characters,
and no more than int(self.block_size*7/8) characters.

- IMPORTANT: You are free to decide how to perform this random truncation, but
make sure that the length is picked _randomly_ (every possible length from 4
to int(self.block_size*7/8) has a chance of being picked) for full credit.

2. Now, break the (truncated) document into three substrings:
    
    [prefix] [masked_content] [suffix]

  In other words, choose three strings prefix, masked_content and suffix
    such that prefix + masked_content + suffix = [the original document].
  The length of [masked_content] should be random, and 1/4 the length of the
    truncated document on average.

- IMPORTANT: You are free to decide how to perform this operation, but
make sure that the length is picked _randomly_ (has a chance of being more or
less than 1/4 the length of the truncated document) for full credit.

3. Rearrange these substrings into the following form:

    [prefix] MASK_CHAR [suffix] MASK_CHAR [masked_content] [pads]
  
  This resulting string, denoted masked_string, serves as the output example.
  Here MASK_CHAR is the masking character defined in Vocabulary Specification,
    and [pads] is a string of repeated PAD_CHAR characters chosen so that the
    entire string is of length self.block_size.
  Intuitively, the [masked_content], a string, is removed from the document and
    replaced with MASK_CHAR (the masking character defined in Vocabulary
    Specification). After the suffix of the string, the MASK_CHAR is seen again,
    followed by the content that was removed, and the padding characters.

4. We now use masked_string to construct the input and output example pair. To
do so, simply take the input string to be masked_string[:-1], and the output
string to be masked_string[1:]. In other words, for each character, the goal is
to predict the next character in the masked string.

5. Making use of the vocabulary that you defined, encode the resulting input
and output strings as Long tensors and return the resulting data point.

----------------
Here are some examples of input-output pairs (x, y):

  第一个问号表示被 masked ，后面两个问号中间夹的是具体被 masked 部分
  这种委屈求全的方式挺有意思的，明明只有一个 decoder ，但还是想像训练 T5 那样训练
  但说不定通过这种 language modeling 的方式真的可以像 T5 那样理解语义

  x: Khatchig Mouradian. Khatchig Mouradian is a jour⁇and tran⁇nalist, writer ⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: hatchig Mouradian. Khatchig Mouradian is a jour⁇and tran⁇nalist, writer ⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

  x: Jaco⁇enry ⁇b H⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: aco⁇enry ⁇b H⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

  x: John Stephen. Born in Glasgow, Steph⁇lder's apprentice on⁇en became a we⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: ohn Stephen. Born in Glasgow, Steph⁇lder's apprentice on⁇en became a we⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□


"""
class CharCorruptionDataset(Dataset):
    def __init__(self, data, block_size):
        self.MASK_CHAR = u"\u2047" # the doublequestionmark character, for mask
        self.PAD_CHAR = u"\u25A1" # the empty square character, for pad

        chars = list(sorted(list(set(data))))
        assert self.MASK_CHAR not in chars 
        assert self.PAD_CHAR not in chars
        chars.insert(0, self.MASK_CHAR)
        chars.insert(0, self.PAD_CHAR)

        # enumerate 会生成索引和元素组成的元组
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }

        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data.split('\n')

    def __len__(self):
        # returns the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # TODO [part e]: see spec above
        document = self.data[idx]
        
        trunc_len = random.randint(4, int(self.block_size * 7/8))
        document = document[:trunc_len]
        
        mask_len = int(trunc_len * random.gauss(1/4, sigma = 0.1))
        
        # 把 mask_len 控制在一定范围内，防止越界
        mask_len = np.clip(mask_len, 1, trunc_len-2)
        mask_pos = random.randint(1, trunc_len - mask_len)
        
        prefix = document[:mask_pos]
        mask = document[mask_pos : mask_pos + mask_len]
        posfix = document[mask_pos + mask_len : ]
        
        masked_str = prefix + self.MASK_CHAR + posfix + self.MASK_CHAR  + mask + self.MASK_CHAR
        masked_str = masked_str + self.PAD_CHAR * (self.block_size - len(masked_str))
        x = masked_str[ : -1]
        y = masked_str[1 : ]
        x = torch.tensor([self.stoi[i] for i in x], dtype = torch.long)
        y = torch.tensor([self.stoi[i] for i in y], dtype = torch.long)
        return x, y
"""
Code under here is strictly for your debugging purposes; feel free to modify
as desired.
"""
if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('dataset_type', help="Type of dataset to sample from."
            "Options: namedata, charcorruption.",
            choices=["namedata", "charcorruption"])
    args = argp.parse_args()

    if args.dataset_type == 'namedata':
        # Even if it hasn't been implemented, we use it to define the vocab
        corruption_dataset = CharCorruptionDataset(open('wiki.txt').read(), 128) 
        # Make the name dataset
        name_dataset = NameDataset(corruption_dataset,
            open('birth_places_train.tsv').read())
        for _, example in zip(range(4), name_dataset):
            x, y = example
            print('x:', ''.join([name_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([name_dataset.itos[int(c)] for c in y]))
        pass
    elif args.dataset_type == 'charcorruption':
        corruption_dataset = CharCorruptionDataset(open('wiki.txt').read(), 128) 
        for _, example in zip(range(4), corruption_dataset):
            x, y = example
            print('x:', ''.join([corruption_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([corruption_dataset.itos[int(c)] for c in y]))
    else:
        raise ValueError("Unknown dataset type in command line args: {}"
                .format(args.dataset_type))

