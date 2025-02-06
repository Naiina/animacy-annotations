#1) load model -> use stats.py
#2) load UD file
#3) anim_tags = [1,0,0,2,-100,...] with 0, Inanimate, 1 animal 2 human thanks to the model
#4) gram_tags = [0,1,-100]: 0 if subj, 1 if obj, -100 otherwise
#5) POS = [0,-100] 0 is noun, 1 if proper noun 2 if pronoun
#5) reuse functions from animacy_classifier to compute avg position in sentence and animacy of subject and objects
from conllu import parse
from collections import defaultdict
from tqdm import tqdm
#from nltk.corpus import wordnet as wn
import pandas as pd
import argparse
import os
#from datasets import load_dataset
import numpy as np

import math as m
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency as chi2_contingency

sns.set_theme(style="whitegrid", font="Times New Roman")

def find_roots(l,pro,defin = False):
    l_deprel = ("csubj","xcomp","ccomp","acl","parataxis","acl:relcl")
    l_waiting = []
    d_roots = {}
    l_det = {"head":[],"def":[]}
    defini = None
    #d_roots_v = {}
    #"conj"
    if defin:
        for d_word in l:
            if "DET" == d_word["upos"]:
                if type(d_word["feats"]) is dict:
                    if "Definite" in d_word["feats"].keys():
                        defini = d_word["feats"]["Definite"]
                        head = d_word["head"]
                        l_det["head"].append(head)
                        l_det["def"].append(defini)


    for d_word in l:
        if d_word["head"] == 0:
            id = d_word["id"]
            l_tree_roots_idx = [id]
            if "NOUN" == d_word["upos"]:
                anim = d_word["misc"]["ANIMACY"]
            elif pro and "PRON" == d_word["upos"] and type(d_word["feats"]) == dict and "Person" in d_word["feats"] and d_word["feats"]["Person"] in ["1","2"]:
                anim = "P"
            else:
                anim = None
            g = d_word["deprel"]
            if defin:
                if id in l_det["head"]:
                    defini = l_det["def"][l_det["head"].index(id)]
                else:
                    defini = None
            
            voice = get_voice(d_word)
            if d_word["upos"] == "VERB":
                id_v = id
                verb = d_word["form"]
            else:
                id_v=-1
                verb = None
            d_roots[(d_word["form"]+'0',voice,id_v,verb)]=([id],[anim],[g],[],[defini])
            
            #if d_word["upos"] == "VERB":
            #    d_roots_v[d_word["form"]+'0']=([id],[anim],[g])
        else:
            l_waiting.append(d_word)

    #Find the "roots" of the other clauses
    changes = True
    while changes : 
        changes = False
        for i,d_word in enumerate(l_waiting):
            rel = d_word["deprel"]#.split(":")[0]
            if rel == 'conj' and d_word["head"] in l_tree_roots_idx or rel in l_deprel:
                #print(d_word["form"],rel)
                id = d_word["id"]
                l_tree_roots_idx.append(id)
                
                if "NOUN" == d_word["upos"]:
                    anim = d_word["misc"]["ANIMACY"]
                elif pro and "PRON" == d_word["upos"] and type(d_word["feats"]) == dict and "Person" in d_word["feats"] and d_word["feats"]["Person"] in ["1","2"]:
                    anim = "P"
                else:
                    anim = None
                #if deprel:
                if defin:
                    if id in l_det["head"]:
                        defini = l_det["def"][l_det["head"].index(id)]
                    else:
                        defini = None
                voice = get_voice(d_word)
                if d_word["upos"] == "VERB":
                    verb_id = id
                    verb = d_word["form"]
                else:
                    verb_id = -1
                    verb = None
                d_roots[(d_word["form"],voice,verb_id,verb)]=([id],[anim],[rel],[],[defini])
                    #if d_word["upos"] == "VERB":
                    #    d_roots_v[d_word["form"]]=([id],[anim],[rel])
                ##else:
                    #d_roots[d_word["form"]]=([id],[anim])
                    #if d_word["upos"] == "VERB":
                    #    d_roots_v[d_word["form"]]=([id],[anim])
                l_waiting.pop(i)
                changes = True

    return d_roots,l_tree_roots_idx

        


def create_subtrees_lists(l,pro,deprel = False,direct_arg_only = False, defin = True,iobj=False,obl=False,passive=False):
    l_waiting_idx = []
    l_waiting_anim = []
    l_waiting_head = []
    l_waiting_gram = []
    l_waiting_def = []
    l_waiting_is_verb = []
    l_det = {"head":[],"def":[]}
    # get roots of each subtree
    d_subtrees,l_tree_roots_idx = find_roots(l,pro,defin)

    l_gram = ["obj","nsubj"]
    if iobj:
        l_gram.append("iobj")
    if obl:
        l_gram.append("obl")
    if passive:
        l_gram  += ["obl:agent","nsubj:pass"]

    if defin:
        for d_word in l:
            if "DET" == d_word["upos"]:
                if type(d_word["feats"]) is dict:
                    if "Definite" in d_word["feats"].keys():
                        defini = d_word["feats"]["Definite"]
                        head = d_word["head"]
                        l_det["head"].append(head)
                        l_det["def"].append(defini)

    for d_word in l:
        idx = d_word["id"]
        head = d_word["head"]
        upos = d_word["upos"]
        gram = d_word["deprel"]#.split(":")[0]
        if defin:
            if idx in l_det["head"]:
                defini = l_det["def"][l_det["head"].index(idx)]
            else:
                defini = None
        if upos != "PUNCT":
            if  upos == "NOUN" :
                anim = d_word["misc"]["ANIMACY"]
            #if upos == "PROPN":
            #    if d_word["misc"]["NER"] =="PER":
            #        anim = "H"
            #    else:
            #        anim = "N"
                
            elif upos == "PROPN":
                if "NER" in d_word["misc"].keys():
                    anim = d_word["misc"]["NER"]
                    if anim == "PERSON":
                        anim = "PER"
                else:
                    anim = None

            elif pro and "PRON" == upos and type(d_word["feats"]) == dict and "Person" in d_word["feats"] and d_word["feats"]["Person"] in ["1","2"]:
                anim = "P"
            else:
                anim = None
            if idx not in l_tree_roots_idx:
                l_waiting_idx.append(idx)
                l_waiting_anim.append(anim)
                l_waiting_head.append(head)
                l_waiting_gram.append(gram)
                l_waiting_is_verb.append((upos == "VERB",get_voice(d_word),d_word["form"]))
                if defin:
                    l_waiting_def.append(defini)
        

    ii = 0
    max_it = len(l_waiting_idx)
    #print("anim",l_waiting_anim )
    
    while l_waiting_idx!=[]:
        i = l_waiting_idx.pop(0)
        a = l_waiting_anim.pop(0)
        h = l_waiting_head.pop(0)
        g = l_waiting_gram.pop(0)
        is_v,new_voice,main_verb = l_waiting_is_verb.pop(0)
        if defin:
            d = l_waiting_def.pop(0)
        
        found = False
        
        # look up if already in a subtree
        #print(d_subtrees)
        for (root,voice,id_v,verb),sub_tree in d_subtrees.items():
            if type(id_v) is int:
                if id_v<0 and is_v:
                    d_subtrees[(root,new_voice,i,main_verb)] = d_subtrees.pop((root,voice,id_v,verb))
                    break

        for (root,voice,id_v,verb),sub_tree in d_subtrees.items():
            if h in sub_tree[0]: 
                
                sub_tree[0].append(i)
                sub_tree[1].append(a)
                sub_tree[2].append(g)
                if defin:
                    sub_tree[4].append(d)

                #if direct_arg_only:
                    #print(voice)
                    #if voice == "passif":
                #    if g in l_gram and h == id_v:
                #        sub_tree[3].append(sub_tree[0].index(i))  

                    #if voice == "actif":
                    #    if g in l_gram and h == sub_tree[0][0]:
                    #        sub_tree[3].append(sub_tree[0].index(i)) 

                found = True
                ii = 0
                max_it = max_it - 1
                break
        # if not found, put back at the end of the waiting lists
        if not found:
            ii+=1
            l_waiting_idx.append(i)
            l_waiting_anim.append(a)
            l_waiting_head.append(h)
            l_waiting_gram.append(g)
            l_waiting_is_verb.append((is_v,new_voice,main_verb))
            if defin:
                l_waiting_def.append(d)
        #print(l_waiting_idx)
        
        
        if ii > max_it+1 :
            break  
    for d_word in l:
        idx = d_word["id"]
        h = d_word["head"]
        g = d_word["deprel"]#.split(":")[0]
        for (verb,voice,id_v,verb),sub_tree in d_subtrees.items():
            #print(sub_tree[0])
            #print("id_v",id_v)
            #print("h",h,"id_v",id_v,g)
            if g in l_gram and h == id_v and idx in sub_tree[0]:
                sub_tree[3].append(sub_tree[0].index(idx)) 

    return d_subtrees


def find_ner_tags(UD_file,max_len):
    data_UD = open(UD_file,"r", encoding="utf-8")
    s_ner = set()
    dd_data_UD = parse(data_UD.read())

    for i,elem in enumerate(tqdm(dd_data_UD)):
        if max_len >0:
            if i >max_len:
                break
        text = elem.metadata['text']
        #print(text)
        l = list(elem)
        for d_word in l:
            if "PROPN" == d_word["upos"]:
                ner = d_word["misc"]["NER"]
                s_ner.add(ner)
    print(s_ner)




def create_csv(UD_file,max_len):
    
    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())
    l_wordform = []
    l_idx_clause = []
    l_idx_word = []
    l_idx_sentence = []
    l_anim = []
    l_deprel = []
    l_defin = []
    l_num = []
    l_pos = []
    l_rel_pos = []
    l_order = []
    l_anim_other = []
    l_verb_voice = []
    l_nb_dir_arg = []
    l_sent = []
    l_root = []
    l_verb = []

    for i,elem in enumerate(tqdm(dd_data_UD)):
        if max_len >0:
            if i >max_len:
                break
        text = elem.metadata['text']
        #print(text)
        l = list(elem)
        d_subtrees = create_subtrees_lists(l,False,True,True)
        
        for idx_clause, (k,(li,la,lg,l4,ld)) in enumerate(d_subtrees.items()):
            li_sort = li.copy()
            li_sort.sort()
            l_idx_main_arg = [li[i] for i in l4]
            l_idx_main_arg.sort()
            #print(li,li_sort)
            for d_word in l:
                if "NOUN" == d_word["upos"]:
                    id_w = d_word["id"]
                    if id_w in li:
                        l_wordform.append(d_word["form"])

                        l_idx_clause.append(idx_clause)
                        l_idx_word.append(id_w)
                        l_idx_sentence.append(elem.metadata["sent_id"])

                        l_deprel.append(d_word["deprel"])
                        l_anim.append(d_word["misc"]["ANIMACY"])

                        idx_list = li.index(id_w)
                        #print(li,idx_list)
                        other_anim = ''.join([k for i,k in enumerate(la) if k!=None and i!=idx_list])
                        l_anim_other.append(other_anim)
                        l_defin.append(ld[idx_list])

                        if type(d_word["feats"]) == dict and "Number" in d_word["feats"].keys():
                            nb = d_word["feats"]["Number"]
                        else:
                            nb = None
                        l_num.append(nb)
                        pos = li_sort.index(id_w)
                        l_pos.append(pos)
                        l_rel_pos.append(pos/len(li))
                        if id_w in l_idx_main_arg:
                            order = l_idx_main_arg.index(id_w)
                            l_nb_dir_arg.append(len(l4)) 
                        else:
                            order = None
                            l_nb_dir_arg.append(0) 
                        l_order.append(order)
                        #print(l_idx_main_arg)
                        #l_dir_arg.append(idx_list in l4)
                        l_sent.append(text)
                        l_verb_voice.append(k[1])
                        l_root.append(k[0])
                        l_verb.append(k[3])

    d = {"wordform":l_wordform,
    "idx_clause":l_idx_clause,
    "idx_word":l_idx_word,
    "idx_sent":l_idx_sentence,
    "anim":l_anim,
    "deprel":l_deprel,
    "defin":l_defin,
    "num":l_num,
    "pos":l_pos,
    "rel_pos":l_rel_pos,
    "order":l_order,
    "anim_others":l_anim_other,
    "root":l_root,
    "main verb":l_verb,
    "verb_voice":l_verb_voice,
    "nb_dir_arg":l_nb_dir_arg,
    "sent":l_sent}
    #for k,v in d.items():
    #    print(k,v)
    df = pd.DataFrame.from_dict(d)
    #print(UD_file)
    df.to_csv('csv/'+UD_file[19:21]+'.csv', index=False)

def acl_roots(UD_file,max_len,all_acl=False):
    #acl:relcl for languages where this exists
    #acl and pronType:Rel 
    #acl for sl because none other indication exists

    #if "all_acl":  all acl for all langauges 
    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())
    l_anim = []
    d_tot = {"P":0,"H":0,"A":0,"N":0,"PER":0}
    
    for i,elem in enumerate(tqdm(dd_data_UD)):
        if max_len >0:
            if i >max_len:
                break
        text = elem.metadata['text']
        #print(text)
        l = list(elem)
        d_acl_idx_h = {}
        d_aclrel_idx_h = {}
        d_noun_idx_anim = {}
        l_pronoun_rel_h = []
        for d_word in l: 
            h = d_word["head"] 
            if type(d_word["feats"]) == dict and "PronType" in d_word["feats"].keys() and "Rel" in d_word["feats"]["PronType"]:
                l_pronoun_rel_h.append(h)
            if d_word["deprel"] == "acl":
                d_acl_idx_h[d_word["id"]] = h
            if d_word["deprel"] == "acl:relcl":
                d_aclrel_idx_h[d_word["id"]] = h

            if d_word["upos"] == "NOUN":
                anim = d_word["misc"]["ANIMACY"]
                d_noun_idx_anim[d_word["id"]] = anim
                if anim in ["A","N","P","H"]:
                    d_tot[d_word["misc"]["ANIMACY"]]+=1
            elif "PRON" == d_word["upos"] and type(d_word["feats"]) == dict and "Person" in d_word["feats"] and d_word["feats"]["Person"] in ["1","2"]:
                d_noun_idx_anim[d_word["id"]] = "P"
                d_tot["P"]+=1
            elif "PROPN" == d_word["upos"]:
                if "NER" in d_word["misc"]:
                    ner = d_word["misc"]["NER"]
                    if ner in ["PER","PERSON"]:
                        d_noun_idx_anim[d_word["id"]] = "PER"
                        d_tot["PER"]+=1
                #else:
        #print("l_pron",l_pronoun_rel_h)
        #print("acl rel",d_aclrel_idx_h)
        #print("acl", d_acl_idx_h)
        #print("noun",d_noun_idx_anim)
                    
        for h_acl_rel in d_aclrel_idx_h.values():
            if h_acl_rel in d_noun_idx_anim.keys():
                l_anim.append(d_noun_idx_anim[h_acl_rel])
                #print("h_acl_rel",h_acl_rel)
        for h_pro in l_pronoun_rel_h:
            if h_pro in d_acl_idx_h.keys():
                h_acl = d_acl_idx_h[h_pro]
                if h_acl in d_noun_idx_anim.keys():
                    l_anim.append(d_noun_idx_anim[h_acl])
        if "sl_" in UD_file or all_acl or "eu_" in UD_file or all_acl:
            for h_acl in d_acl_idx_h.values():
                if h_acl in d_noun_idx_anim.keys():
                    l_anim.append(d_noun_idx_anim[h_acl])


    return l_anim,d_tot


def proportion_sentences_only_one_anim(UD_file):

    only_one_class = 0
    no_pron = 0

    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())
    for i,elem in enumerate(tqdm(dd_data_UD)):
        text = elem.metadata['text']
        #print(text)
        l = list(elem)
        mem = None
        b_no_diff_anim = True
        b_no_pron = True
        for d_word in l:
            if  d_word["upos"] in ["PROPN","PRON"]:
                b_no_pron = False
            if "NOUN" == d_word["upos"]:
                anim = d_word["misc"]["ANIMACY"]
                if mem == None:
                    mem = anim
                elif mem != anim:
                    b_no_diff_anim = False
                    #break
        if b_no_diff_anim:
            only_one_class+=1
            print(text)
            if b_no_pron:
                print(text)
                no_pron +=1
            
    print(i,only_one_class,no_pron)
    return only_one_class, no_pron

def definitness_and_animacy(UD_file,max_len):

    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())
    idxx = 0
    d_ind = {"N":0,"A":0,"H":0}
    d_def = {"N":0,"A":0,"H":0}
    d_count = {"Ind":d_ind,"Def":d_def}
    for i,elem in enumerate(tqdm(dd_data_UD)):
        idxx +=1
        if max_len >0:
            if idxx > max_len:
                break
        
        text = elem.metadata['text']
        #print(text)
        l = list(elem)
        l_det = []
        d_noun = {}
        #gather all nouns and det of the sentence l
        for d_word in l:
            if "DET" == d_word["upos"]:
                if type(d_word["feats"]) is dict:
                    if "Definite" in d_word["feats"].keys():
                        defin = d_word["feats"]["Definite"]
                        head = d_word["head"]
                        det = (head,defin)
                        l_det.append(det)
                
            if "NOUN" == d_word["upos"]:
                anim = d_word["misc"]["ANIMACY"]
                id = d_word["id"]
                d_noun[id] = anim
        #find for each det the associated noun
        for (head,defin) in l_det:
            if head in d_noun.keys():
                anim = d_noun[head]
                d_count[defin][anim]=d_count[defin][anim]+1
    np_count = np.array([list(d_count[k].values()) for k in d_count.keys()])

    return d_count,np_count


def number_and_animacy(UD_file, max_len):
    data_UD = open(UD_file, "r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())
    d_count = defaultdict(lambda: defaultdict(int))

    for idxx, elem in enumerate(tqdm(dd_data_UD)):
        if max_len >0:
            if idxx > max_len:
                break

        l = list(elem)
        #gather all nouns of the sentence l
        for d_word in l:
            if d_word["upos"] == "NOUN":
                anim = d_word["misc"]["ANIMACY"]
            elif d_word["upos"] == "PRON":
                anim = "P"
            if d_word["upos"] in ["NOUN", "PRON"] and type(d_word["feats"]) == dict and "Number" in d_word["feats"].keys():
                nb = d_word["feats"]["Number"]
                if nb in ["Sing", "Plur"]:
                    if nb and anim:
                        d_count[nb][anim]+=1

    return d_count


def prop_sentences_only_two_enties_diff_anim(UD_file):

    only_one_class = 0
    exactly_two_anim = 0

    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())
    for i,elem in enumerate(tqdm(dd_data_UD)):
        text = elem.metadata['text']
        #print(text)
        l = list(elem)
        sev_anim = filter_diff_anim(l,"several_anim")
        only_two_enties_diff_anim = filter_diff_anim(l,"exactly_two_diff_anim")
        if not sev_anim:
            only_one_class+=1
        if only_two_enties_diff_anim:
            exactly_two_anim+=1
            
    return only_one_class, exactly_two_anim


def filter_diff_anim(l,tag):
    if tag == "all":
        return True
    mem = []
    only_two_enties_diff_anim = False
    sev_anim = False
    for d_word in l:
        if "NOUN" == d_word["upos"]:
            anim = d_word["misc"]["ANIMACY"]
            mem.append(anim)
    if len(mem) == 2 and mem[0] != mem[1]:
        only_two_enties_diff_anim = True  
    if len(set(mem))>1:
        sev_anim = True
    if tag == "exactly_two_diff_anim":
        return only_two_enties_diff_anim
    if tag == "several_anim":
        return sev_anim


def noun_only_position_in_subtree(UD_file,rel,max_len=-1,pro = True,per = True):
    # The cat of my sister plays with a ball
    #      1           2                  3 
    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())
    
    l_tags = ["N","A","H"]
    if pro:
        l_tags.append("P")
    if per:
        l_tags.append("PER")
    d_pos_anim = {"P":[],"H":[],"A":[],"N":[],"PER":[]}
    for i,elem in enumerate(tqdm(dd_data_UD)):
        if max_len >0:
            if i > max_len:
                break
        
        text = elem.metadata['text']
        #print(text)
        l = list(elem)

        d_subtrees = create_subtrees_lists(l,pro,False)
        #print(d_subtrees)
        pass
        for k,(li,la,lg,l4,ld) in d_subtrees.items():
            zipped = list(zip(li,la))
            z_sorted = sorted(zipped, key = lambda x: x[0])
            pos = 0
            dd_pos_anim = {"PER":[],"P":[],"H":[],"A":[],"N":[]}
            for ind, (idx,anim) in enumerate(z_sorted):
                if anim in l_tags:
                    dd_pos_anim[anim].append(pos)
                    pos+=1
            if rel:
                if pos > 0:
                    for k in dd_pos_anim.keys():
                        dd_pos_anim[k] = [elem/pos for elem in dd_pos_anim[k]]
            for k in d_pos_anim.keys():
                d_pos_anim[k] = d_pos_anim[k] + dd_pos_anim[k]
    for k in d_pos_anim.keys():
        if len(d_pos_anim[k])>0:
            d_pos_anim[k] = sum(d_pos_anim[k])/len(d_pos_anim[k])
        else:
            d_pos_anim[k] = None

    return (d_pos_anim)


def noun_only_position_in_subtree_ner(UD_file,rel,max_len=-1,pro = True):
    # The cat of my sister plays with a ball
    #      1           2                  3 
    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())

    d_pos_anim = {"PER":[],"ORG":[],"LOC":[],"MISC":[],"O":[]}
    for i,elem in enumerate(tqdm(dd_data_UD)):
        if max_len >0:
            if i > max_len:
                break
        
        text = elem.metadata['text']
        #print(text)
        l = list(elem)

        d_subtrees = create_subtrees_lists(l,pro,False)
        #print(d_subtrees)
        
        
        for k,(li,la,lg,l4,ld) in d_subtrees.items():
            subtree_len = len(li)
            zipped = list(zip(li,la))
            z_sorted = sorted(zipped, key = lambda x: x[0])
            #print(z_sorted)
            #pass
            order = 0
            dd_pos_anim = {"PER":[],"ORG":[],"LOC":[],"MISC":[],"O":[]}
            for ind, (idx,anim) in enumerate(z_sorted):
                #if anim != None and :
                if anim in ["PER","O","MISC","LOC","ORG"]:
                        
                    dd_pos_anim[anim].append(order)
                    order+=1
            if rel:
                if order > 0:
                    for k in dd_pos_anim.keys():
                        dd_pos_anim[k] = [elem/order for elem in dd_pos_anim[k]]
            for k in d_pos_anim.keys():
                d_pos_anim[k] = d_pos_anim[k] + dd_pos_anim[k]
    for k in d_pos_anim.keys():
        if len(d_pos_anim[k])>0:
            d_pos_anim[k] = sum(d_pos_anim[k])/len(d_pos_anim[k])
        else:
            d_pos_anim[k] = None

    return (d_pos_anim)



def position_in_subtree(UD_file,rel,which_clauses,max_len=-1,tag = "all",pro = True, defin = True):

    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())
    idxx = 0
    d_pos_anim = {"N":[],"A":[],"H":[],"P":[]}
    d_pos_anim_defin = {"Def_N":[],"Def_A":[],"Def_H":[],"Def_P":[],"Ind_N":[],"Ind_A":[],"Ind_H":[],"Ind_P":[]}
    d_pos_defin = {"Def":[],"Ind":[]}
    for i,elem in enumerate(tqdm(dd_data_UD)):
        idxx +=1
        if max_len >0:
            if idxx > max_len:
                break
        
        text = elem.metadata['text']
        #print(text)
        l = list(elem)

        d_subtrees = create_subtrees_lists(l,pro,defin)
        #print(d_subtrees)
        
        
        for k,(li,la,lg,l4,ld) in d_subtrees.items():
            of_interest = False
            #print(k,k[0])
            if filter_diff_anim(l,tag): # filter on the number of animacy classes
                if which_clauses == "main" and k[0][-1] == '0':
                    of_interest = True
                if which_clauses == "sub" and k[0][-1] != '0':
                    print("hey")
                    exit() 
                    of_interest = True
                if which_clauses == "all":
                    of_interest = True
            
            if of_interest:
                subtree_len = len(li)
                if defin:
                    zipped = list(zip(li,la,ld))
                else:
                    zipped = list(zip(li,la))
                z_sorted = sorted(zipped, key = lambda x: x[0])
                if defin:
                    for ind, (idx,anim,definintness) in enumerate(z_sorted):
                        if anim != None:
                            if rel :
                                d_pos_anim[anim].append(ind/subtree_len)
                                if definintness!=None:
                                    d_pos_defin[definintness].append(ind/subtree_len)
                                    d_pos_anim_defin[definintness+"_"+anim].append(ind/subtree_len)
                            else:
                                d_pos_anim[anim].append(ind)
                                if definintness!=None:
                                    d_pos_defin[definintness].append(ind)
                                    d_pos_anim_defin[definintness+"_"+anim].append(ind)
                else:
                    for ind, (idx,anim) in enumerate(z_sorted):
                        if anim != None:
                            if rel :
                                d_pos_anim[anim].append(ind/subtree_len)
                            else:
                                d_pos_anim[anim].append(ind)

    return (d_pos_anim,d_pos_defin,d_pos_anim_defin)


def tuple_to_consider_rank(max_len,tag,pro):

    if tag == "exactly_two_diff_anim":
        return [(7515, (1, 0, 1, 0)), (2388, (1, 0, 0, 1)), (884, (1, 1, 0, 0)), (372, (0, 0, 1, 1)), (98, (0, 1, 1, 0)), (45, (0, 1, 0, 1))]
    else:
        files = os.listdir("UD_with_anim_annot")
        d_tup = {}
        for file in files:
            UD_file = "UD_with_anim_annot/"+file
            data_UD = open(UD_file,"r", encoding="utf-8")
            dd_data_UD = parse(data_UD.read())
            idxx = 0
            for i,elem in enumerate(tqdm(dd_data_UD)):
                idxx +=1
                if max_len >0:
                    if idxx > max_len:
                        break
                
                text = elem.metadata['text']
                #print(text)
                l = list(elem)

                d_subtrees = create_subtrees_lists(l,pro)

                for t_elem in d_subtrees.values():
                    elem = t_elem[1]
                    tup = (elem.count("N"),elem.count("A"),elem.count("H"),elem.count("P"))
                    if tag == "several_anim":
                        if tup.count(0)<3:
                            if tup in d_tup.keys():
                                d_tup[tup]+=1
                            else:
                                d_tup[tup]=1
                    else:
                        if tup in d_tup.keys():
                            d_tup[tup]+=1
                        else:
                            d_tup[tup]=1

        l_tup = []
        for k,v in d_tup.items():
            l_tup.append((v,k))
        l_tup.sort(reverse = True)
        #print(l_tup[:10])
        return l_tup[:20]


def permut_rank_in_subtree(UD_file,ll_tup,which_clauses,max_len,tag ,pro = True):

    #ll_tup = [(45687, (1, 0, 0, 0)), (45021, (0, 0, 0, 0)), (30076, (2, 0, 0, 0)), (17014, (3, 0, 0, 0)), (9402, (4, 0, 0, 0)), (8543, (0, 0, 1, 0)), (7515, (1, 0, 1, 0)), (5296, (5, 0, 0, 0)), (5050, (2, 0, 1, 0)), (4516, (0, 0, 0, 1))]
    l_tup_to_consider = [k[1] for k in ll_tup]
    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())
    idxx = 0
    d_permut = {str(tup_to_consider):{} for tup_to_consider in l_tup_to_consider}
    for i,elem in enumerate(tqdm(dd_data_UD)):
        idxx +=1
        if max_len >0:
            if idxx > max_len:
                break
        
        text = elem.metadata['text']
        #print(text)
        l = list(elem)

        d_subtrees = create_subtrees_lists(l,pro)
        for k,t_raw in d_subtrees.items():
            of_interest = False
            if which_clauses == "main" and k[-1] == '0':
                of_interest = True
            if which_clauses == "sub" and k[-1] != '0':
                of_interest = True
            if which_clauses == "all":
                of_interest = True
            
            if of_interest:
                t_of_interest = [(t_raw[0][i],anim) for i,anim in enumerate(t_raw[1]) if anim != None]
                l_permut = [anim for pos,anim in t_of_interest]
                c_permut = tuple(l_permut)
                s_permut = str(c_permut)
                t_of_interest.sort()
                which_tuple = str((l_permut.count("N"),l_permut.count("A"),l_permut.count("H"),l_permut.count("P")))

                if which_tuple in d_permut.keys():
                    if s_permut in d_permut[which_tuple].keys():
                        d_permut[which_tuple][s_permut]+=1
                    else:
                        d_permut[which_tuple][s_permut] = 1
    return (d_permut)


def rank_in_subtree_H(which_clauses,max_len=-1,tag = "all",pro = True):
    tup_to_consider = tuple_to_consider_rank(max_len,tag,pro)
    
    l_lang = []
    l_tup = []
    l_rank_H = []
    files = os.listdir("UD_with_anim_annot")
    for file in files:
        UD_file = "UD_with_anim_annot/"+file
        lang = file[:2]
        d_permut = permut_rank_in_subtree(UD_file,tup_to_consider,which_clauses,max_len,tag,pro)
        for tup, dd_perm  in d_permut.items():
            if tup[7] !="0":
                print("tup",tup,"dd_perm",dd_perm)
                pos_H = []
                count = sum(dd_perm.values())
                #print(count)
                str_to_list_tup = [int(elem) for elem in list(tup) if elem not in [",","(",")"," ","'"]]
                nb_entities = sum(str_to_list_tup)
                nb_perm = len(dd_perm)
                print("nb_perm",nb_perm)
                if nb_perm >0:
                    tot = 0
                    for perm, occ in dd_perm.items():
                        str_to_list_perm = [elem for elem in list(perm) if elem not in [",","(",")"," ","'"]]
                        for i,elem in enumerate(str_to_list_perm):
                            if elem == "H":
                                pos_H.append((i+1)*occ)
                                tot+=occ
                    print("pos_H",pos_H)
                    print("tot",tot)
                    if tot>0:
                        l_rank_H.append(sum(pos_H)/(tot*nb_entities))
                        l_tup.append(tup)
                        l_lang.append(lang)
    d = {"Language":l_lang,"Tup":l_tup,"Average H rank":l_rank_H}
    df = pd.DataFrame.from_dict(d)

    return df

def rank_in_subtree_main_arg(UD_file,diff_classes,max_len=-1,tag = "all",pro = False,defin = False,anim_or_gram = "anim",iobj=False,obl=False):
    
    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())
    d_pos = defaultdict(list)
    for i,elem in enumerate(tqdm(dd_data_UD)):

        if max_len >0:
            if i > max_len:
                break
    
        text = elem.metadata['text']
        #print(text)
        l = list(elem)
        d_subtrees = create_subtrees_lists(l,pro,False,True,defin,iobj,obl)
        #print(d_subtrees)
        for l_pos,l_anim,l_gram,ldir_arg,ld in d_subtrees.values():
            num_args = len(ldir_arg)
            #class_counts[num_args] += num_args
            #print(l_anim,ldir_arg)
            l_all = [(l_pos[i],l_anim[i],l_gram[i]) for i in ldir_arg if l_anim[i] !=None ]
            l_anim_only = [l_anim[i] for i in ldir_arg if l_anim[i] !=None ]
            nb_classes = len(set(l_anim_only))
            if (len(l_all)<6 and (iobj or obl)) or len(l_all)==2:
                if diff_classes:
                    if nb_classes>1:
                        #print(text)
                        l_all.sort()
                        #nb = len(l_all)
                        for i,elem in enumerate(l_all):
                            if anim_or_gram == "anim":
                                d_pos[elem[1]].append(i)
                            if anim_or_gram == "gram":
                                d_pos[elem[2]].append(i)
                            if anim_or_gram == "both":
                                if elem[2]=="nsubj":
                                    g = 0
                                else:
                                    g = 1
                                d_pos[elem[1]].append(g)
                else:
                    l_all.sort()
                    #nb = len(l_all)
                    print(l_all)
                    for i,elem in enumerate(l_all):
                        if anim_or_gram == "anim":
                            d_pos[elem[1]].append(i)
                        if anim_or_gram == "gram":
                            d_pos[elem[2]].append(i)
                        if anim_or_gram == "both":
                                if elem[2]=="subj":
                                    g = 0
                                else:
                                    g = 1
                                d_pos[elem[1]].append(g)

    for k,v in d_pos.items():
        print(k,np.mean(v))
    #print(d_pos)
    return d_pos


def rank_in_subtree_entropy(which_clauses,max_len=-1,tag = "all",pro = True):
    tup_to_consider = tuple_to_consider_rank(max_len,tag,pro)
    
    l_lang = []
    l_tup = []
    l_entropy = []
    files = os.listdir("UD_with_anim_annot")
    for file in files:
        UD_file = "UD_with_anim_annot/"+file
        lang = file[:2]
        d_permut = permut_rank_in_subtree(UD_file,tup_to_consider,which_clauses,max_len,tag,pro)
        print(d_permut)
        for tup, dd_perm  in d_permut.items():
            print(dd_perm)
            if len(dd_perm)>1:
                print("hey")
                entropy = 0
                count = sum(dd_perm.values())
                for perm, occ in dd_perm.items():
                    p = occ/count
                    entropy += -p*m.log(p,2)
                l_entropy.append(entropy)
                l_tup.append(str(tup))
                l_lang.append(lang)
    d = {"Language":l_lang,"Tup":l_tup,"Entropy":l_entropy}
    df = pd.DataFrame.from_dict(d)

    return df


def animacy_and_voice(UD_file,max_len):
    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())

    d_voice = {"nsubj_A":{"P":0,"H":0,"A":0,"N":0},"nsubj_S":{"P":0,"H":0,"A":0,"N":0},"nsubj:pass":{"P":0,"H":0,"A":0,"N":0},"obl:agent":{"P":0,"H":0,"A":0,"N":0},"obj":{"P":0,"H":0,"A":0,"N":0}}

    for i,elem in enumerate(tqdm(dd_data_UD)):
        if max_len >0 and i >max_len:
            break
        
        text = elem.metadata['text']
        #print(text)
        l = list(elem)
        l_head_anim_subj = {}
        l_head_anim_obj = {}

        for d_word in l: 
            
            if d_word["upos"] == "NOUN":
                anim = d_word["misc"]["ANIMACY"]
            elif "PRON" == d_word["upos"] and type(d_word["feats"]) == dict and "Person" in d_word["feats"] and d_word["feats"]["Person"] in ["1","2"]:
                anim = "P"
            else:
                anim = None
            if anim in ["A","P","H","N"]:
                #print(d_word["form"],d_word["head"],d_word["deprel"],anim)
                if d_word["deprel"] == "obj":
                    l_head_anim_obj[d_word["head"]] = anim
                elif d_word["deprel"] == "nsubj":
                    l_head_anim_subj[d_word["head"]] = anim
                elif d_word["deprel"] == "nsubj:pass":
                    d_voice["nsubj:pass"][anim]+=1
                elif d_word["deprel"] == "obl:agent":
                    d_voice["obl:agent"][anim]+=1
        #print("obj",l_head_anim_obj)
        #print("subj",l_head_anim_subj)
        for h,a in l_head_anim_subj.items():
            if h in l_head_anim_obj.keys(): #tran verb
                d_voice["obj"][l_head_anim_obj[h]]+=1
                d_voice["nsubj_A"][a]+=1
            else:
                d_voice["nsubj_S"][a]+=1
        #print(d_voice)

    return d_voice
        
        


def print_default_dict(d_voice):
    dd = {}
    #for k,v in d_voice.items()
    d = {kk:dict(vv) for kk,vv in d_voice.items()}
    print(d)     





def plot_proportion_num(ud_repo, max_len, output_repo):
    animacy_classes = ["P", "H", "A", "N"]
    number_classes = ["Plur", "Sing"]
    l_lang = []
    l_prop = []
    l_num = []
    l_anim = []
    files = os.listdir(ud_repo)
    for file in files:
        if file[:2] not in ["ja", "zh", "ko"]:
            UD_file = os.path.join(ud_repo, file)
            d_count = number_and_animacy(UD_file, max_len)

            totals = {ac: sum(d_count[nc][ac] for nc in number_classes) for ac in animacy_classes}

            for ac in animacy_classes:
                for nc in number_classes:
                    l_lang += [file[:2]]
                    l_num += [nc]
                    l_anim += [ac]
                    if totals[ac] > 0:
                        l_prop += [d_count[nc][ac] / totals[ac]]
                    else:
                        l_prop += [0]
    
    d = {"Language": l_lang, "Proportion": l_prop, "Number": l_num, "Animacy": l_anim}
    df = pd.DataFrame.from_dict(d)
    df = df[df.Number == "Plur"]

    sns.lineplot(
    data=df,
    x='Animacy',
    y='Proportion',
    hue='Language',
    ci=None,
    palette='Set2'
    )
    plt.xlabel('Animacy')
    plt.ylabel('Proportion of Plurals')
    plt.xticks(rotation=45)

    plt.savefig(os.path.join(output_repo, "proportion_plot_number.pdf"))
    plt.show()


def plot_proportion_def():
    l_lang = []
    l_prop = []
    l_def = []
    files = os.listdir("UD_with_anim_annot")
    for file in files:
        if file[:2] not in ["ja","zh","ko","sl"]:
            UD_file = "UD_with_anim_annot/"+file
            d_count,np_count = definitness_and_animacy(UD_file,-1)
            d_def = d_count["Def"]
            d_ind = d_count["Ind"]
            a_def = d_def["A"]+d_def["H"]+d_def["N"]
            a_ind = d_ind["A"]+d_ind["H"]+d_ind["N"]
            if a_def>0:
                l_prop += [d_def["H"]/a_def]
            else:
                l_prop+=[0]
            if a_ind >0:
                l_prop += [d_ind["H"]/a_ind]
            else:
                l_prop += [0]
            l_def = l_def +["Def"] +["Ind"]
            l_lang = l_lang + [file[:2]]*2

    d = {"language":l_lang,"proportion":l_prop,"Definitness":l_def}
    df = pd.DataFrame.from_dict(d)
    #print(df)

    

    # Draw a nested barplot by species and sex
    g = sns.catplot(
        data=df, kind="bar",
        x="language", y="proportion", hue="Definitness",
        errorbar="sd", palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("", "proportion")
    #g.fig.suptitle("Proportion of animate entities per definitness")
    plt.savefig("UD_data_anim/proportion_plot_definitness.png")
    plt.show()


def plot_proportion_voice_H(diff_anim,exactly_two):
    l_lang = []
    l_prop = []
    l_voice = []
    files = os.listdir("UD_with_anim_annot")
    for file in files:
        print(file[:2])
        if file[:2] not in ["ja","zh","ko","sl"]:
            UD_file = "UD_with_anim_annot/"+file
            d_voice = animacy_and_voice(UD_file,-1)
            for elem,d in d_voice.items():
                tot = d["A"]+d["H"]+d["N"]
                if tot>0:
                    l_prop += [d["H"]/tot]
                else:
                    l_prop+=[0]
                l_voice = l_voice +[elem] 
                l_lang.append(file[:2])

    d = {"language":l_lang,"proportion of humans":l_prop,"Deprel":l_voice}
    df = pd.DataFrame.from_dict(d)
    #print(df)
    df.iloc[1], df.iloc[2] = df.iloc[2].copy(), df.iloc[1].copy()
    print(df)

    

    # Draw a nested barplot by species and sex
    g = sns.catplot(
        data=df, kind="bar",
        x="language", y="proportion of humans", hue="Deprel",
        errorbar="sd", palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("", "Proportion of human")
    #g.fig.suptitle("Proportion of animate entities per definitness")
    plt.savefig("UD_plots/proportion_plot_voice.png")
    plt.show()


def plot_proportion_voice_subj(diff_anim,exactly_two):
    l_lang = []
    l_prop = []
    l_lab = []
    files = os.listdir("UD_with_anim_annot")
    for file in files:
        print(file[:2])
        UD_file = "UD_with_anim_annot/"+file
        d_voice= animacy_and_voice(UD_file,-1)
        for lab in ["H","N"]:
            tot = d_voice["nsubj:pass"][lab]+d_voice["obj"][lab]
            if tot>0:
                l_prop += [d_voice["nsubj:pass"][lab]/tot]
            else:
                l_prop+=[0]
            l_lab.append(lab)
            l_lang.append(file[:2])
        #print(len(l_lang),len(l_prop),len(l_lab))

    d = {"language":l_lang,"proportion":l_prop,"patient's animacy":l_lab}
    df = pd.DataFrame.from_dict(d)
    #print(df)
    #df.iloc[1], df.iloc[2] = df.iloc[2].copy(), df.iloc[1].copy()
    print(df)

    

    # Draw a nested barplot by species and sex
    g = sns.catplot(
        data=df, kind="bar",
        x="language", y="proportion", hue="patient's animacy",
        errorbar="sd", palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("","proportion of passive sentence")
    #g.fig.suptitle("Proportion of animate entities per definitness")
    plt.savefig("UD_plots/proportion_plot_voice_H.png")
    plt.show()


def plot_proportion_deprel(diff_anim,exactly_two):
    l_lang = []
    l_prop = []
    l_lab = []
    files = os.listdir("UD_with_anim_annot")
    for file in files:
        print(file[:2])
        UD_file = "UD_with_anim_annot/"+file
        d_voice = animacy_and_voice(UD_file,-1)
        for lab in ["H","A","N"]:
            tot = d_voice["nsubj"][lab]+d_voice["nsubj:pass"][lab]+d_voice["obj"][lab]
            if tot>0:
                l_prop += [(d_voice["nsubj"][lab]+d_voice["nsubj:pass"][lab])/tot]
            else:
                l_prop+=[0]
            l_lab.append(lab)
            l_lang.append(file[:2])
        #print(len(l_lang),len(l_prop),len(l_lab))

    d = {"language":l_lang,"proportion":l_prop,"Subject's animacy":l_lab}
    df = pd.DataFrame.from_dict(d)
    #print(df)
    #df.iloc[1], df.iloc[2] = df.iloc[2].copy(), df.iloc[1].copy()
    print(df)

    
    g = sns.catplot(
        data=df, kind="bar",
        x="language", y="proportion", hue="Subject's animacy",
        errorbar="sd", palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("","Proportion of subjects")
    #g.fig.suptitle("Proportion of animate entities per definitness")
    plt.savefig("UD_plots/proportion_plot_deprel.png")
    plt.show()



def plot_pos_subtree_diamonds(rel,which_clauses,size,tag,pro,defin):
    l_lang = []
    l_anim = []
    l_pos = []
    files = os.listdir("UD_with_anim_annot")
    for file in files:

        UD_file = "UD_with_anim_annot/"+file
        d_pos_anim,d_pos_defin,d_pos_anim_defin = position_in_subtree(UD_file,rel,which_clauses,size,tag,pro,defin)
        #print(d_pos)

        ll_anim = []
        for k in d_pos_anim.keys():
            ll_anim = ll_anim + [k]*len(d_pos_anim[k])
            l_pos = l_pos + d_pos_anim[k]
        l_anim = l_anim + ll_anim
        l_lang = l_lang + [file[:2]]*len(ll_anim)
        
        #print(len(l_anim),len(l_lang),len(l_pos))

    d = {"language":l_lang,"position":l_pos,"animacy":l_anim}
    df = pd.DataFrame.from_dict(d)

    # Draw a nested boxplot to show bills by day and time
    sns.set_theme(style="ticks", palette="pastel")
    sns.boxplot(x="language", y="position",
                hue="animacy", palette=["m","r", "g","b"],
                data=df)
    sns.despine(offset=10, trim=True)

    plt.title("Relative position within subtrees per animacy class")
    plt.savefig("UD_plots/position_within_subtrees_plot"+str(rel)+"_"+which_clauses+"_clauses_tag_"+tag+"_pro_"+str(pro)+".png")
    
    plt.show()


def plot_pos_subtree_mean(rel,tag_anim,anim_or_def="anim"):

    if anim_or_def == "anim":
        d = {"language": [],"animacy": [],"mean":[]}
    if anim_or_def == "defin":
        d = {"language": [],"definiteness": [],"mean":[]}
    files = os.listdir("UD_with_anim_annot")
    for file in files:
        
       UD_file = "UD_with_anim_annot/"+file
       d_pos_anim,d_pos_defin,d_pos_anim_defin = position_in_subtree(UD_file,rel,"all",-1,"exactly_two_diff_anim",True,True)
       
       if anim_or_def=="anim":
           for elem,v in d_pos_anim.items():
               d["mean"].append(np.mean(v))
               d["language"].append(file[:2])
               d["animacy"].append(elem)
       if anim_or_def=="defin":
            if file[:2] in ["en","it","es","de","fr","nl"]:

                for elem,v in d_pos_defin.items():
                    d["mean"].append(np.mean(v))
                    d["language"].append(file[:2])
                    d["definiteness"].append(elem)
    df = pd.DataFrame.from_dict(d)
    print(df)
    


    if anim_or_def=="anim":
       g = sns.catplot(
           data=df, kind="bar",
           x="language", y="mean", hue="animacy",
           errorbar=None, palette="dark", alpha=.6, height=6
       )
       g.despine(left=True)
       if rel:
           g.set_axis_labels("", "mean relative position")
           plt.savefig("mean_relative_position"+tag_anim+".png")


       else:
           g.set_axis_labels("", "mean position")
           plt.savefig("mean_position"+tag_anim+".png")
    if anim_or_def=="defin":
       g = sns.catplot(
           data=df, kind="bar",
           x="language", y="mean", hue="definiteness",
           errorbar=None, palette="dark", alpha=.6, height=6
       )
       g.despine(left=True)
       if rel:
           g.set_axis_labels("", "mean relative position")
           plt.savefig("mean_relative_position"+tag_anim+".png")
    g.legend.set_title("")
    plt.show()

def plot_pos_subtree_mean_diff(rel):

    d = {"language": [],"sentence types": [],"mean diff":[]}
    files = os.listdir("UD_with_anim_annot")
    for file in files:
        print(file[:2])
        UD_file = "UD_with_anim_annot/"+file
        #d_pos_anim_two,d_pos_defin,d_pos_anim_defin = position_in_subtree(UD_file,rel,"all",-1,"exactly_two_diff_anim",False,False)
        d_pos_anim_all,d_pos_defin,d_pos_anim_defin = position_in_subtree(UD_file,rel,"all",-1,"all",False,False)
        d_pos_anim_sev,d_pos_defin,d_pos_anim_defin = position_in_subtree(UD_file,rel,"all",-1,"several_anim",False,False)
        #for elem,v in d_pos_anim_two.items():
        d["mean diff"].append(np.mean(d_pos_anim_all["N"])-np.mean(d_pos_anim_all["H"]))
        d["language"].append(file[:2])
        d["sentence types"].append("all entities")
        #d["mean diff"].append(np.mean(d_pos_anim_two["N"])-np.mean(d_pos_anim_two["H"]))
        #d["language"].append(file[:2])
        #d["sentence types"].append("two entities with diff anim labels")
        d["mean diff"].append(np.mean(d_pos_anim_sev["N"])-np.mean(d_pos_anim_sev["H"]))
        d["language"].append(file[:2])
        d["sentence types"].append("several animacy labels")
    df = pd.DataFrame.from_dict(d)
    print(df)
    

    # Draw a nested barplot by species and sex
    g = sns.catplot(
        data=df, kind="bar",
        x="language", y="mean diff", hue="sentence types",
        errorbar=None, palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    if rel:
        g.set_axis_labels("", "N mean pos - H mean pos")
        plt.savefig("mean_relative_position_diff.png")

    else:
        g.set_axis_labels("","N mean pos - H mean pos")
        plt.savefig("mean_position_diff.png")
    g.legend.set_title("")
    plt.show()
    

def plot_pos_subtree_rank(diff_classes,max_len,tag = "all",pro=False,defin=False,anim_or_gram="anim",iobj=False,obl=False):
    l_lang = []
    l_anim = []
    l_pos = []
    files = os.listdir("UD_with_anim_annot")
    for file in files:
        print(file[:2])

        UD_file = "UD_with_anim_annot/"+file
        d_pos= rank_in_subtree_main_arg(UD_file,diff_classes,max_len,tag,pro,defin,anim_or_gram,iobj,obl)
        #print(d_pos)
        ll_anim = []
        for k,v in d_pos.items():
            ll_anim = ll_anim + [k]*len(v)
            l_pos = l_pos + v
        l_anim = l_anim + ll_anim
        l_lang = l_lang + [file[:2]]*len(ll_anim)
        
        #print(len(l_anim),len(l_lang),len(l_pos))

    d = {"language":l_lang,"position":l_pos,"animacy":l_anim}
    df = pd.DataFrame.from_dict(d)
    df.to_csv('results/rank.csv', index=False)

    g = sns.catplot(
    data=df, kind="bar",
    x="language", y="position",
                hue="animacy", palette="dark", alpha=.6, height=6, errorbar=None
    )
    g.despine(left=True)
    g.set_axis_labels("", "Mean rank")
    g.legend.set_title("")

    #plt.title("Relative rank within subtrees per animacy class")
    if diff_classes:
        plt.savefig("UD_plots/rank_within_subtrees_diff_classes_"+anim_or_gram+".png")
    else:
        plt.savefig("UD_plots/rank_within_subtrees_"+anim_or_gram+".png")
    plt.show()


def plot_pos_subtree_anim_def(rel,which_clauses,size,tag,pro,defin):
    sns.set_theme(style="ticks", palette="pastel")
    d = {"language": [],"animacy": [],"mean":[]}
    files = os.listdir("UD_with_anim_annot")
    for file in files:
        print(file[:2])
        if file[:2] in ["en","it","fr","nl","es","de"]:
            
            UD_file = "UD_with_anim_annot/"+file
            d_pos_anim,d_pos_defin,d_pos_anim_defin = position_in_subtree(UD_file,rel,which_clauses,size,tag,pro,defin)
            print(d_pos_anim_defin.keys())
            d["mean"].append(np.mean(d_pos_anim["H"]))
            d["language"].append(file[:2])
            d["animacy"].append("Human")
        #for elem,v in d_pos_defin.items():
            d["mean"].append(np.mean(d_pos_defin["Def"]))
            d["language"].append(file[:2])
            d["animacy"].append("Definite")
        #for elem,v in d_pos_anim_defin.items():
            d["mean"].append(np.mean(d_pos_anim_defin["Def_H"]))
            d["language"].append(file[:2])
            d["animacy"].append("Both")
    df = pd.DataFrame.from_dict(d)
    print(df)
    

    # Draw a nested barplot by species and sex
    g = sns.catplot(
        data=df, kind="bar",
        x="language", y="mean", hue="animacy",
        errorbar=None, palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    if rel:
        g.set_axis_labels("", "mean relative position")
        plt.savefig("mean_relative_position_anim_def.png")

    else:
        g.set_axis_labels("", "mean position")
        plt.savefig("mean_position_anim_def.png")
    g.legend.set_title("")
    plt.show()

    
def print_l_words(l):
    for elem in l:
        print(elem.word,elem.head,elem.gram)
        print(elem.voice)  


def get_voice(d_word):
    voice = None
    #if lang == "fr":
    if "VERB" == d_word["upos"]:
        if d_word["feats"] != None:
            if "Pass" in d_word["feats"].values():
                if any(item in d_word.values() for item in {"root","ccomp","relcl"}):
                    voice = "passif"
                else:
                    voice = None
            else:
                voice = "actif"
            
    return voice

def find_verb_dep(verb,l_subj,l_obj,l_obl):
    l_dep = []
    for elem in l_subj+l_obj+l_obl:
        if elem.head == verb.pos_in_sent+1:
            l_dep.append((elem.pos_in_sent,elem))
    l_dep.sort() 
    return l_dep

def find_det(idx_subj,l_det):
    det = None
    for elem in l_det:
        if elem[0]==idx_subj:
            det = elem[1]
            break

    return det


#annotate_UD_file_passif(UD_file,json_file,-1,False,"Naiina/UD_"+lang+"_anim_pred",lang)

def prop(d):

    a = d["A"]+d["H"]+d["N"]
    if a ==0:
        print("empty dict")
    else:
        print("N:",round(100*d["N"]/a,2)," A:",round(100*d["A"]/a,2)," H:",round(100*d["H"]/a,2))




def plot_rel_pos_all():
    #plot all the subtree positions for all the files in "UD_with_anim_annot"
    l_rel = [True,False]
    l_which_clauses = ["main","all","sub"]
    l_pro = [True,False]
    for pro in l_pro:
        for which_clauses in l_which_clauses:
            for rel in l_rel:
                plot_pos_subtree(rel,which_clauses,-1,"all",pro)
                plot_pos_subtree(rel,which_clauses,-1,"several_anim",pro)
                plot_pos_subtree(rel,which_clauses,-1,"exactly_two_diff_anim",pro)



def compute_k2(feat):
    print(feat)
    files = os.listdir("UD_with_anim_annot")
    all = np.zeros((2,3))
    d = {"lang": [],"khi2": [],"pval":[],"passive proportion":[]}
    tot_subj_a = 0
    tot_subj_p = 0
    for file in files:
        print(file[:2])
        UD_file = "UD_with_anim_annot/"+file
        if feat == "defin":
            a,np_count = definitness_and_animacy(UD_file,-1)
        if feat == "num":
            v,np_count = number_and_animacy(UD_file,-1)
        v,np_count_agent,np_count_patient = animacy_and_voice(UD_file,-1,diff_anim=False,exactly_two=False)
        if feat == "voice_patient":
            np_count = np_count_patient
        if feat == "voice_agent":
            np_count = np_count_agent
        print(np_count)
        if np_count.prod() != 0:
            khi2, pval , ddl , contingent_theorique = chi2_contingency(np_count)
            d["lang"].append(file[:2])
            d["khi2"].append(khi2)
            d["pval"].append(pval)
            if feat in ["voice_agent","voice_patient"]:
                nb_subj_a = sum(v["nsubj"].values())
                nb_subj_p = sum(v["nsubj:pass"].values())
                tot_subj_a+=nb_subj_a
                tot_subj_p+=nb_subj_p
                d["passive proportion"].append(nb_subj_p/(nb_subj_p+nb_subj_a))
        all+=np_count
    df = pd.DataFrame(d)
    nn = df[['pval', 'passive proportion']].to_numpy()
    print(nn)
    khi2, pval , ddl , contingent_theorique = chi2_contingency(all)
    d["lang"].append("all")
    d["khi2"].append(khi2)
    d["pval"].append(pval)
    d["passive proportion"].append(tot_subj_p/(tot_subj_p+tot_subj_a))


    df = pd.DataFrame(d)

    print(df)

    khi2, pval , ddl , contingent_theorique = chi2_contingency(nn)
    print("pval and passive proportion correlation",khi2,pval)
    return df 




def khi2_position():

    files = os.listdir("UD_with_anim_annot")
    all = np.zeros((2,3))
    d = {"lang": [],"A": [],"H":[],"N":[],"P":[]}
    tot_subj_a = 0
    tot_subj_p = 0
    for file in files:
        print(file[:2])
        UD_file = "UD_with_anim_annot/"+file
        d_pos_anim,d_pos_defin,d_pos_anim_defin = position_in_subtree(UD_file,True,"all",-1,"all",True,)
        for elem in d_pos_anim.keys():
            d[elem].append(np.mean(d_pos_anim[elem]))
        d["lang"].append(file[:2])
        #np_count = np.array([list(np.mean(d_pos_anim[k].values())) for k in d_pos_anim.keys()])
    print(d)
    df = pd.DataFrame(d)
    N_H = df[['N',"H"]].to_numpy()
    filtered_df = df[df['P'].notna()]
    N_H_P =  filtered_df[['N',"H","P"]].to_numpy()
    N_P =  filtered_df[['N',"P"]].to_numpy()
    print(N_H)
    khi2, pval , ddl , contingent_theorique = chi2_contingency(N_H)
    print("N H",khi2,pval)
    print(N_H)
    khi2, pval , ddl , contingent_theorique = chi2_contingency(N_H_P)
    print("N H P",khi2,pval)
    print(N_P)
    khi2, pval , ddl , contingent_theorique = chi2_contingency(N_P)
    print("N P",khi2,pval)


def line_plot_pos(UD_folder,rel,pro,per):
    d_all = {}
    for file in os.listdir(UD_folder):
        lang = file[:2]
        print(lang)
        UD_file_ner = UD_folder+file
        d = noun_only_position_in_subtree(UD_file_ner,rel,-1,pro,per)
        print(d)
        d_all[lang] = d
        
    df = pd.DataFrame(d_all)
    df.to_csv("pos_anim.csv")
    df = df.loc[['P', 'H', "A", "N"]]
    print(df)
    df.plot(marker='o', figsize=(8, 5))

    plt.xlabel("Animacy labels")
    plt.ylabel("Average positions")
    plt.legend(title="Languages")
    plt.grid(True)
    plt.savefig("rel_pos.png")
    plt.show()


def line_plot_voice(UD_folder,voice,max_len):
    
    def extract_proportions(d):
        return {gram: anim_dict_count['H'] / sum(anim_dict_count.values()) if sum(anim_dict_count.values())>0 else np.nan for gram,anim_dict_count in d.items() }
    d_heatmap_data={}
    l_lang = []
    for file in os.listdir(UD_folder):
        lang = file[:2]
        l_lang.append(lang)
        print(lang)
        UD_file_ner = UD_folder+file
        d_count = animacy_and_voice(UD_file_ner,max_len)
        print(d_count)
        proportions = extract_proportions(d_count)
        d_heatmap_data[lang] = proportions

    df = pd.DataFrame(d_heatmap_data)


    # Transpose to have languages as the index
    #df = df.T
    if voice == "active":
        df_voice = df.loc[['nsubj_A', 'nsubj_S',"obj"]]
        df_voice.dropna(axis=1, how='all')
    if voice == "passive":
        df_voice = df.drop(columns=['et', 'eu',"sl","ja","zh"])
        df_voice = df_voice.loc[['obl:agent',"nsubj:pass"]]
        df_voice.dropna(axis=1, how='all')
    if voice == "P":
        df_voice = df.drop(columns=['et', 'eu',"sl","ja"])
        df_voice = df_voice.loc[["nsubj:pass","obj"]]
        df_voice.dropna(axis=1, how='all')
    # Plotting the line plot
    df_voice.plot(marker='o', figsize=(8, 5))
    df.to_csv("lineplot_voices.csv", index=False)
    
   
    #plt.title("Values of A, B, and C for Each Language")
    plt.xlabel("deprel")
    plt.ylabel("Proportion of human")
    plt.legend(title="languages")
    plt.grid(True)
    plt.savefig("line_voice_"+voice+".png")
    plt.show()
    

def plot_acl(UD_folder,max_len=-1):
    d = {}
    for file in os.listdir(UD_folder):
        lang = file[:2]
        if lang not in ["eu","sl","ja"]:
            print(lang)
            UD_file_ner = UD_folder+file
        
            l,d_tot = acl_roots(UD_file_ner,max_len)
            d[lang] = {elem: l.count(elem)/d_tot[elem] for elem in ["P","H","A","N"] if d_tot[elem]>0}


    print(d)
    df = pd.DataFrame(d)
    
    #df.loc[["H","A","P", "N"]] = df.loc[["H","A","N", "P"]].values
    df.to_csv("anim_acl_rel_pron.csv")
    #df = df.drop(index=2)
    print(df)

    df.plot(marker='o', figsize=(8, 5))
   
    #plt.title("Values of A, B, and C for Each Language")
    plt.xlabel("Animacy labels")
    plt.ylabel("Proportion of relativized entities")
    plt.legend(title="Languages")
    plt.grid(True)
    plt.savefig("rel_cl")
    plt.show()


def plot_subj_ratio(UD_folder,max_len=-1):
    d_ratio = {}
    for file in os.listdir(UD_folder):
        lang = file[:2]
        print(lang)
        UD_file_ner = UD_folder+file
        d_count  = animacy_and_voice(UD_file_ner,-1)
        print(d_count)
        #exit()
        nb_N = sum([d["N"] for d in d_count.values()])
        nb_H = sum([d["H"] for d in d_count.values()])
        print("nb:",nb_N,nb_H)
        d_ratio[lang] = {"h_patient": (d_count["nsubj:pass"]["H"]/nb_H)/(d_count["nsubj"]["N"]/nb_N+d_count["nsubj:pass"]["H"]/nb_H), "n_patient": (d_count["nsubj:pass"]["N"]/nb_N)/(d_count["nsubj"]["H"]/nb_H+d_count["nsubj:pass"]["N"]/nb_N)}
        
        #s = set(l)
        #print(l,d_tot)
        #d[lang] = {elem: l.count(elem)/d_tot[elem] for elem in ["H","A","N","P","PER"] if d_tot[elem]>0}


    print(d_ratio)
    df = pd.DataFrame(d_ratio)
    
    #df.loc[["H","A","P", "N"]] = df.loc[["H","A","N", "P"]].values
    df.to_csv("anim_voice_ratio")
    print(df)

    df.plot(marker='o', figsize=(8, 5))
   
    #plt.title("Values of A, B, and C for Each Language")
    plt.xlabel("Language")
    plt.ylabel("ratio")
    plt.legend(title="Metrics")
    plt.grid(True)
    plt.show()

def plot_heatmap_voice(UD_folder, max_len):


    def extract_proportions(d):
        return {gram: anim_dict_count['H'] / sum(anim_dict_count.values()) if sum(anim_dict_count.values())>0 else np.nan for gram,anim_dict_count in d.items() }
    d_heatmap_data={}
    l_lang = []
    for file in os.listdir(UD_folder):
        lang = file[:2]
        l_lang.append(lang)
        print(lang)
        UD_file_ner = UD_folder+file
        d_count = animacy_and_voice(UD_file_ner,max_len)
        print(d_count)
        proportions = extract_proportions(d_count)
        d_heatmap_data[lang] = proportions

    df = pd.DataFrame(d_heatmap_data)
    #i, j = 1, 2  # Swap Row2 and Row4
    #rows = df.iloc[[i, j]].copy()
    # Swap values and index names
    #df.iloc[[j, i]] = rows.values
    #df.index.values[[i, j]] = df.index.values[[j, i]]
    print(df)
    df_voice = df.loc[["nsubj:pass","obj"]]
    df.to_csv("heatmap_voice.csv", index=False)
    plt.figure(figsize=(8, 4))
    sns.heatmap(df_voice, annot=True, cmap="Blues")
    plt.title("Proportion of humans")
    plt.show()


PLOT_FUNCTIONS = {
    "number": plot_proportion_num,
}

def get_args():

    parser = argparse.ArgumentParser(description="Animacy plots.")

    # Adding an argument 'type' with choices 'deprel' and 'number'
    parser.add_argument(
        '--type',
        type=str,
        choices=['number'],
        required=True,
    )
    parser.add_argument(
        '--max_len',
        type=int,
        default=-1,
    )
    parser.add_argument(
        '--ud_folder',
        type=str,
        default="UD_with_anim_ner_annot/",
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        default="plots",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    plot_function = PLOT_FUNCTIONS[args.type]
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    plot_function(args.ud_folder, args.max_len, args.output_folder)