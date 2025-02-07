from tqdm import tqdm
from collections import defaultdict
from conllu import parse


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


def create_subtrees_lists(l, pro, defin=True, iobj=True, obl=True, passive=True):
    l_waiting_idx = []
    l_waiting_anim = []
    l_waiting_head = []
    l_waiting_gram = []
    l_waiting_def = []
    l_waiting_is_verb = []
    l_det = {"head":[],"def":[]}
    # get roots of each subtree
    d_subtrees,l_tree_roots_idx = find_roots(l,pro,defin)

    l_gram = ["nsubj", "obj"]
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


def noun_only_position_in_subtree(UD_file, rel=False, max_len=-1, pro=True, per=False):
    # The cat of my sister plays with a ball
    #      1           2                  3 
    data_UD = open(UD_file,"r", encoding="utf-8")
    dd_data_UD = parse(data_UD.read())
    
    l_tags = []
    if pro:
        l_tags.append("P")
    if per:
        l_tags.append("PER")
    l_tags += ["H", "A", "N"]
    d_pos_anim = {ac: [] for ac in l_tags}
    for i, elem in enumerate(tqdm(dd_data_UD)):
        if max_len > 0:
            if i > max_len:
                break
        
        l = list(elem)
        d_subtrees = create_subtrees_lists(l, pro, per)
        for k, (li, la, ld, _, _) in d_subtrees.items():
            zipped = list(zip(li, la, ld))
            z_sorted = sorted(zipped, key = lambda x: x[0])
            pos = 1
            dd_pos_anim = {ac: [] for ac in l_tags}
            for _, anim, deprel in z_sorted:
                # Only main arguments
                if anim in l_tags and deprel in ["nsubj", "obj", "iobj", "obl", "nsubj:pass", "obl:agent"]:
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

def acl_roots(UD_file,max_len, all_acl=False):
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


    return l_anim, d_tot

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