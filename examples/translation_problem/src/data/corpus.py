from typing import List,Union
from collections import Counter

class Vocab:
    def __init__(self,tokens:List[str],min_freq:int,reserved_tokens:list = []):
        assert (
                isinstance(tokens,list) and 
                all([isinstance(token,str) for token in tokens])
                ), "Tokens parameter must to be List[str], ['house','hug','kiss',...]"
        
        token_counts = Counter(tokens)

        self.token_freqs = sorted( 
                            token_counts.items(),
                            key= lambda item: item[1],reverse=True
                        )
    
        self.idx_to_token = list(sorted(set(
            ["<unk>"] + reserved_tokens + 
            [token for token,freq in self.token_freqs if freq>=min_freq]
        )))

        self.token_to_idx = {token:idx for idx,token in enumerate(self.idx_to_token)}

    def __len__(self)->int:
        return len(self.idx_to_token)
    
    def __getitem__(self,token:Union[str, list])->int:
        
        assert ( 
            isinstance(token,str) or 
            isinstance(token,list) 
             
        ), "token parameter must to be a string or a list"

        if isinstance(token,str):
            return self.token_to_idx.get(token,self.unk)

        return [self.__getitem__(tk) for tk in token]
    
    def to_token(self,idx:Union[int,list])->Union[str,list]:
        assert (
            isinstance(idx,int) or 
            isinstance(idx,list)
            
        ), "idx parameter must to be a integer or list of integers"

        if isinstance(idx,int):
            return self.idx_to_token[idx]
        
        return [self.to_token(i) for i in idx]
    
    @property
    def unk(self):
        return self.token_to_idx["<unk>"]