from dataclasses import dataclass, replace
from typing import Optional, List, Union, Dict, Any, Iterable
import re

from emoji import UNICODE_EMOJI
# from spacy.tokens import Doc
# from spacy.vocab import Vocab
import pandas as pd
from pandas import DataFrame

from common.utils import get_columns, ColumnNameSpace


def is_emoji(s):
    return s in UNICODE_EMOJI


def is_cn(s):
    for w in s:
        if not '\u4e00' <= w <= '\u9fff':
            return False
    return True


# class Message(Doc):
#     @property
#     def num_emoji(self):
#         res = sum(c in UNICODE_EMOJI for c in self.text)

#     def __init__(self, sender: str, vocab: Vocab, words: Optional[List[str]] = ..., spaces: Optional[List[bool]] = ..., user_data: Optional[Dict[Any, Any]] = ..., tags: Optional[List[str]] = ..., pos: Optional[List[str]] = ..., morphs: Optional[List[str]] = ..., lemmas: Optional[List[str]] = ..., heads: Optional[List[int]] = ..., deps: Optional[List[str]] = ..., sent_starts: Optional[List[Union[bool, None]]] = ..., ents: Optional[List[str]] = ...) -> None:
#         super().__init__(vocab, words=words, spaces=spaces, user_data=user_data, tags=tags, pos=pos,
#                          morphs=morphs, lemmas=lemmas, heads=heads, deps=deps, sent_starts=sent_starts, ents=ents)
#         self.sender = sender


class Conversation:
    def __init__(self, index, df: DataFrame) -> None:
        self.col_obj = get_columns()
        self.index = index
        if isinstance(df,pd.Series):
            df=df.to_frame().T
        self.df = df


    @property
    def length(self):
        return len(self.df)

    @property
    def num_c_msg(self):
        col = self.col_obj
        df = self.df
        try:
            res = df[col.speaker].value_counts()[0]
        except KeyError:
            res=0
        return res

    @property
    def num_s_msg(self):
        return len(self.df)-self.num_c_msg

    @property
    def percent_c_msg(self):
        return self.num_c_msg/len(self.df)


class Session:
    pass
