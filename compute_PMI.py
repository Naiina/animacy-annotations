import torch
import json
import torch.nn as nn
from transformers import  AutoModelForCausalLM, GPT2TokenizerFast, AutoTokenizer
from datasets import Dataset 
import torch.nn.functional as F
from surprisal_train import proj_decoder 
import copy


torch.manual_seed(40) 


#tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
#o =tokenizer(["the cat","the white cat who like babanas"], padding =  True)
#print(o)
#exit()



tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", add_prefix_space=True)
tokenizer.pad_token = tokenizer.eos_token


#decoder1 = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto")

decoder = torch.load('model_5_emb_size.pth',map_location=torch.device('cpu'))
#exit()
#torch.save(model, 'model.pth')
#decoder2 = AutoModelForCausalLM.from_pretrained("path_to_save", device_map="auto")
#exit()
#decoder2 = AutoModelForCausalLM.from_pretrained("checkpoint-30464", device_map="auto")

text = ["The", "red", "cat", "plays", "with","my","hand","."]
#nouns_id = [2,6]
#nouns = [text[k] for k in nouns_id]
#nouns_tok = tokenizer(nouns,is_split_into_words = True).input_ids
#exit()




def replace_anim_by_zero(data):
    data_w_anim = copy.deepcopy(data)
    for elem in data_w_anim:
        l = elem["animacy"].shape
        elem["animacy"] = torch.zeros(l,dtype=torch.int)
        #print(elem)
    return data_w_anim


lang = "en"
k = 2
data = torch.load("datasets/"+lang+"_train_dataset.pth",map_location=torch.device('cpu') )
#data_w = replace_anim_by_zero(data)

iidx = 1

output = decoder(**data[iidx])
output_w = decoder(data[iidx]["input_ids"],torch.zeros(data[1]["animacy"].shape,dtype=torch.int))
#print(output.logits == output_w.logits)
#print(data[iidx]["input_ids"].shape)
#exit()

sentence = tokenizer.decode(data[iidx]["input_ids"],is_split_into_words = True,skip_special_tokens=True)
print(sentence)
print(data[iidx]["input_ids"][:])
print(data[iidx]["labels"][:])
print(data[iidx]["animacy"][:])

for elem in data[iidx]["input_ids"]:
    print(elem, tokenizer.decode(elem))
#exit()

for idx,elem in enumerate(data[iidx]["labels"]):
    
    if elem != -100:

        #output
        #logits = output.logits[idx-1, :] # logits of size seq_len * voc_size
        logits = output.logits[idx-1, :] # logits of size seq_len * voc_size
        probabilities = torch.softmax(logits, dim=-1) #size voc_size
        proba_true_next_word = probabilities[elem]
        word = tokenizer.decode([elem])
        print("proba of "+word+": ",proba_true_next_word.detach())

        #output_w
        logits_w = output_w.logits[idx-1, :] # logits of size seq_len * voc_size
        probabilities_w = torch.softmax(logits_w, dim=-1) #size voc_size
        proba_true_next_word_w = probabilities_w[elem]
        word_w = tokenizer.decode([elem])
        print("proba of "+word+" without anim info: ",proba_true_next_word_w.detach())

        #mots_probable
        topk_proba,topk_idx = torch.topk(probabilities_w, k, dim=-1)
        topk_proba,topk_idx  = topk_proba.detach(),topk_idx.tolist()
        most_probable_next_word = [tokenizer.decode([idx]) for idx in topk_idx]
        print("most next probable word: ",topk_proba,most_probable_next_word)
        print("\n")



#for i,elem in enumerate(nouns_id):
 #   logits = output.logits[:, elem-1, :]
  #  probabilities = torch.softmax(logits, dim=-1)
    # proba_true_next_word = probabilities[0][nouns_tok[i]]
#   print("proba of "+text[elem]+": ",proba_true_next_word)

#    topk_proba,topk_idx = torch.topk(probabilities, k, dim=-1)
#    topk_proba,topk_idx  = topk_proba[0],topk_idx[0].tolist()
#    most_probable_next_word = [tokenizer.decode([idx]) for idx in topk_idx]
#    print("most next probablem word: ",topk_proba,most_probable_next_word)


#probabilities = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
#top_k_probs, top_k_indices = torch.topk(probabilities, top_k, dim=-1)
#top_k_words = [tokenizer.decode([idx]) for idx in top_k_indices[batch_id]]

