import sys
import os 
import wget
import zipfile
import re 
from typing import List,Tuple
from corpus import Vocab
import torch
"""_________________SETTING___________________"""

PROJECT_PATH = os.path.realpath(os.path.dirname(__file__)).replace(
    "src/data",""
)
ROOT_PATH = PROJECT_PATH.replace("examples/translation_problem","")
sys.path.append(ROOT_PATH)

DEFAULT_PUNCTUATIONS = [
    ",",".","!","?",
    "...",";",":",
    "_",'"',"'","(","[",
    "]",")","\\","/"
] 

"""_________________CODE___________________"""

class TranslationDataLoader():
    
    def __init__(
            self,
            batch_size:int,
            num_steps:int=9,
            num_train:int=512,
            num_val:int=128,
            min_freq_token:int=2,
            sep_src_tgt:str ="\t"
        ):
        super(TranslationDataLoader,self).__init__()
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_train = num_train
        self.num_val = num_val
        self.min_freq_token = min_freq_token
        self.sep_src_tgt = sep_src_tgt

        self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(
        self._download())

    def download_file(self,url:str,destination:str):
        path_save = f"{PROJECT_PATH}{destination}"
        wget.download(url,f"{path_save}/pt-eng.zip")
    
    def extract_zip(self,url:str,destination:str):
        with zipfile.ZipFile(url,"r") as zip:
            zip.extractall(destination)


    def remove_non_breaking(self,text:str)->str:
        text = re.sub(r"(?:\\u202f)|(?:\\xa0)"," ",text)
        return text

    def add_space_between_punctuations(self,text:str)->str:
        pattern_left = r"(?<=\S)("+"|".join(
            [
                r"(\{})".format(punctuation) 
                for punctuation in DEFAULT_PUNCTUATIONS
            ]
        )+r")"
        pattern_right = r"("+"|".join(
            [
                r"(\{})".format(punctuation) 
                for punctuation in DEFAULT_PUNCTUATIONS
            ]
        )+r")(?=\S)"
        text = re.sub(pattern_left,lambda m:f" {m.group(1)}",text)
        text = re.sub(pattern_right,lambda m:f"{m.group(1)} ",text)

        return text

    def tokenization_src_tgt(self,text:str,sep:str)->Tuple[List[str],List[str]]:
        try:
            src,tgt = text.split(sep)[:2]
            src_tkns = src.split() + ["<eos>"]  
            tgt_tkns = tgt.split() + ["<eos>"]
        except:
            src_tkns,tgt_tkns = None,None
        
        return src_tkns,tgt_tkns
    

    def preprocess_src_tgt_by_lines(self,text:str,sep:str)->Tuple[List[str],List[str]]:
        text = self.add_space_between_punctuations(text)
        lines_text = text.split("\n")
        
        src,tgt = [], [] 
        for line in lines_text:
            src_tkns,tgt_tkns = self.tokenization_src_tgt(line,sep)

            if (src_tkns is not None) and (tgt_tkns is not None):
                src.append(src_tkns)
                tgt.append(tgt_tkns)
        
        return src,tgt  
    
    
    def _build_array(
            self,sentences:List[str],
            vocab:Vocab,
            is_tgt:bool=False
        )->Tuple[torch.Tensor,Vocab,torch.Tensor]:

        assert (
            isinstance(sentences,list) and 
            all([isinstance(s,str) for s in sentences])
        ), "sentences parameter must to be a list of strings"

        
        sentences_cleansed = [
            seq[:self.num_steps] if len(seq) >= self.num_steps 
            else  seq + ["<pad>"]*(self.num_steps - len(seq))

            for seq in sentences
        ]

        if is_tgt:
            sentences_cleansed = [["<bos>"] + seq for seq in sentences_cleansed]
        if vocab is None:
            tokens = [token for sentence in sentences_cleansed for token in sentence]
            vocab = Vocab(tokens,min_freq=self.min_freq_token)

        array = torch.tensor([vocab(seq) for seq in sentences_cleansed])
        valid_len = (array!=vocab["<pad>"]).type(torch.int32).sum(1)

        
        return array, vocab, valid_len
    
    def _build_arrays(
            self,
            text:str,
            src_vocab:Vocab=None,
            tgt_vocab:Vocab=None
        )->Tuple[
            Tuple[
                torch.Tensor,
                torch.Tensor,
                int,
                torch.Tensor
            ],
            Vocab,
            Vocab
        ]:

        src,tgt = self.preprocess_src_tgt_by_lines(text,self.sep_src_tgt)
        src_array,src_vocab,src_valid_len = self._build_array(src,src_vocab)
        tgt_array,tgt_vocab,_ = self._build_array(tgt,tgt_vocab,is_tgt=True)

        return (
            (
                src_array,
                tgt_array[:,:-1],
                src_valid_len,
                tgt_array[:,1:]
            ),
            src_vocab,
            tgt_vocab
        )
    