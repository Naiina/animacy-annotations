import os
import csv
import random
import pyconll
from collections import Counter




ud_dir = 'UD_with_anim_ner_annot'
out_dir = 'EXP_ud_sample'
os.makedirs(out_dir, exist_ok=True)

SEED = 42
random.seed(SEED)

# whenev

def get_label_info(sentence):
    debug = False
    #if sentence.id == 'train-s4846':
    #    debug = True

    tokens = []
    labels = []
    indexes = []
    char_spans = []

    text = sentence.text
    text_lower = text.lower()
    text_len = len(text)
    position = 0  

    for i, tok in enumerate(sentence):
        if '.' in str(tok.id) or not tok.misc.get("ANIMACY"): # if . in id, it's an empty node
            continue

        form = tok.form
        form_lower = form.lower()
        form_len = len(form)

        found = False


        while position <= text_len - form_len:
            span = (position, position + form_len)
            window = text_lower[position:position + form_len]


            if window == form_lower:
                start, end = span
                tokens.append(form)
                labels.append(list(tok.misc["ANIMACY"])[0])
                indexes.append(i)
                char_spans.append((start, end))

                extracted = text[start:end]
                if extracted.lower() != form_lower:
                    print(f"[MISMATCH] ID={sentence.id} Token='{form}' vs Text[{start}:{end}]='{extracted}'")

                position = end  # Advance to end of current match
                found = True
                break

            elif text[position:position + form_len] == form:
                start, end = span
                tokens.append(form)
                labels.append(list(tok.misc["ANIMACY"])[0])
                indexes.append(i)
                char_spans.append((start, end))

                if debug:
                    print(f"[FALLBACK RAW MATCH] Matched '{form}' at chars {start}-{end}")

                position = end
                found = True
                break

            else:
                position += 1
            

        if not found:
            print(f"\n[ERROR] Token '{form}' NOT FOUND in sentence ID={sentence.id}")
            print(f"Sentence: {sentence.text}")

    return tokens, labels, indexes, char_spans

target_per_label = {'A': 100, 'H': 100, 'N': 100}

for f in os.listdir(ud_dir):
    if not f.endswith('.conllu'):
        continue

    lang = f.split('_')[0]
    print(f"\nProcessing {lang}")
    ud = pyconll.load_from_file(os.path.join(ud_dir, f))

    sentences = []
    for sentence in ud:
        tokens, labels, indexes, spans = get_label_info(sentence)
        if labels:
            sentences.append({
                'id': sentence.id,
                'text': sentence.text,
                'tokens': tokens,
                'labels': labels,
                'indexes': indexes,
                'spans': spans
            })

    print(f"  Total labeled sentences: {len(sentences)}")

    random.shuffle(sentences)
    selected = []
    label_counter = Counter()
    used_ids = set()

    # Step 1: fill A
    for sent in sentences:
        if 'A' in sent['labels']:
            selected.append(sent)
            label_counter.update(sent['labels'])
            used_ids.add(sent['id'])
            if label_counter['A'] >= target_per_label['A']:
                break
    print(f"  After A pass: {len(selected)} sentences, A count = {label_counter['A']}")

    # Step 2: fill H
    for sent in sentences:
        if sent['id'] in used_ids:
            continue
        if 'H' in sent['labels']:
            selected.append(sent)
            label_counter.update(sent['labels'])
            used_ids.add(sent['id'])
            if label_counter['H'] >= target_per_label['H']:
                break
    print(f"  After H pass: {len(selected)} sentences, H count = {label_counter['H']}")

    # Step 3: fill N if needed
    for sent in sentences:
        if sent['id'] in used_ids:
            continue
        if 'N' in sent['labels']:
            selected.append(sent)
            label_counter.update(sent['labels'])
            used_ids.add(sent['id'])
            if label_counter['N'] >= target_per_label['N']:
                break

    print(f"  After N pass: {len(selected)} sentences, N count = {label_counter['N']}")


    #selected = selected[:max(target_per_label.values()) * len(target_per_label)] 
    print(f"  Final count: {len(selected)}")
    print(f"  Final label distribution (total mentions): {dict(label_counter)}")

    if selected:
        random.shuffle(selected) 
        out_path = os.path.join(out_dir, f"{lang}_sample.csv")
        with open(out_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'sentence_id', 'language', 'text', 'tokens',
                'animacy_labels', 'label_token_indexes', 'label_char_spans'
            ])
            for row in selected:
                writer.writerow([
                    row['id'], lang, row['text'], 
                    '|'.join(row['tokens']),
                    '|'.join(row['labels']),
                    '|'.join(map(str, row['indexes'])),
                    '|'.join([f"{start}-{end}" for (start, end) in row['spans']])
                ])

        print(f"  Written to: {out_path}")