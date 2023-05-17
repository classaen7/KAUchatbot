from eunjeon import Mecab
import re

class Preprocessor:
    def __init__(self,usrdict=None,stopwords_tag=None,stopwords_str=None,intent_words_list=None):
        self.mecab = Mecab()
        #리스트 형태
        self.usrdict = usrdict
        self.stopwords_tag = stopwords_tag
        self.stopwords_str = stopwords_str
        self.intent_words_list = intent_words_list
        
    #정규표현식 패턴
    def remove_consonants_vowels(self,sentence):
        pattern = re.compile(r'[ㄱ-ㅎㅏ-ㅣ]+')
        re_sentence = re.sub(pattern, '', sentence)
        return re_sentence
        
    def check_word_in_sentence(self,words,sentence):
        if any(b in self.intent_words_list for b in lb):return True
        else:False

    def transform_sentence(self, sentence):
        re_sentence = ""
        sentence = self.remove_consonants_vowels(sentence)
        for n,pos in self.mecab.pos(sentence):
            if all(pos in self.stopwords_tag for pos in pos.split('+')):
                continue
            if pos in self.stopwords_tag:
                continue
#             if self.stopwords_str != None and n in self.stopwords_str:
#                 continue
            re_sentence += n
        
        return re_sentence
    
    def labeling(self,t_sentence):
        for label,words_list in enumerate(self.intent_words):
            if self.check_word_in_sentence(words_list,t_sentence):
                return label
        return len(self.intent_words)
    
    def transform(self,sentence):
        t_sentence = self.transform_sentence(sentence)
        return t_sentence, self.labeling(t_sentence)
                

#d