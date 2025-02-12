def get_prompt(l):
    return f"Classify the animacy of '{l['target_word']}' in the context of the following {l['lang']} sentence: '{l['sentence']}'. Use: H for Humans, A for Animate (non-human living organisms, such as plants or animals or ancestral creature), N for Inanimate. Respond with only one letter: H, A, or N.'"
def annotation_prompt(l):
    return f"Classify the animacy of the referent of the word '{l['target_word']}' in the context of the following {l['lang']} sentence: '{l['sentence']}. Respond with only one letter: H (Human, also for collectives inherently composed by human in the context), A (Animate), or N (Inanimate)."

import logging
import jsonlines
def build_data(jl_file, n_samples):
    logging.info(f"Processing file: {jl_file}")
    n_samples = n_samples if n_samples > 0 else float('inf')
    try:
        data = []
        with jsonlines.open(jl_file) as reader:
            labels = ['A', 'N', 'H']
            label_counts = {label: 0 for label in labels}

            for entry in reader:
                label = str(entry['label'])

                if label in label_counts and label_counts[label] < n_samples:
                    data.append({"inputs": get_prompt(entry), "targets": label})
                    label_counts[label] += 1

                if all(label_counts[l] >= n_samples for l in labels):
                    break

        logging.info(f"Label distribution: {label_counts}")
        return data

    except Exception as e:
        logging.error(f"Error processing file {jl_file}: {e}")
        return []

ADJ_MAP_LARGE = {
        'abq': 'abaza',
        'af': 'afrikaans',
        'ajp': 'south levantine arabic',
        'akk': 'akkadian',
        'aln': 'albanian',
        'am': 'amharic',
        'apu': 'apurinã',
        'aqz': 'akuntsu',
        'ar': 'arabic',
        'arr': 'karo (brazil)',
        'be': 'belarusian',
        'bej': 'beja',
        'bg': 'bulgarian',
        'bho': 'bhojpuri',
        'bm': 'bambara',
        'bn': 'bengali',
        'bor': 'borôro',
        'bxr': 'russia buriat',
        'ca': 'catalan',
        'ceb': 'cebuano',
        'ckt': 'chukot',
        'cop': 'coptic',
        'cs': 'czech',
        'cu': 'church slavic',
        'cy': 'welsh',
        'da': 'danish',
        'de': 'german',
        'en': 'english',
        'es': 'spanish',
        'ess': 'esselen',
        'et': 'estonian',
        'eu': 'basque',
        'fa': 'persian',
        'fi': 'finnish',
        'fo': 'faroese',
        'fr': 'french',
        'frm': 'middle french',
        'fro': 'old french',
        'ga': 'irish',
        'gd': 'scottish gaelic',
        'gl': 'galician',
        'gn': 'guarani',
        'got': 'gothic',
        'grc': 'ancient greek',
        'gsw': 'swiss german',
        'gub': 'guajajára',
        'gun': 'mbyá guaraní',
        'gv': 'manx',
        'hbo': 'ancient hebrew',
        'he': 'hebrew',
        'hi': 'hindi',
        'hit': 'hittite',
        'hr': 'croatian',
        'hsb': 'upper sorbian',
        'ht': 'haitian creole',
        'hu': 'hungarian',
        'hy': 'armenian',
        'hyw': 'western armenian',
        'id': 'indonesian',
        'is': 'icelandic',
        'it': 'italian',
        'ja': 'japanese',
        'kfm': 'khunsari',
        'kk': 'kazakh',
        'kmr': 'northern kurdish',
        'koi': 'komi-permyak',
        'ko': 'korean',
        'kpv': 'komi-zyrian',
        'ky': 'kyrgyz',
        'la': 'latin',
        'lij': 'ligurian',
        'lt': 'lithuanian',
        'lv': 'latvian',
        'lzh': 'literary chinese',
        'mdf': 'moksha',
        'mk': 'macedonian',
        'ml': 'malayalam',
        'mr': 'marathi',
        'mt': 'maltese',
        'myu': 'mundurukú',
        'myv': 'erzya',
        'nap': 'neapolitan',
        'nds': 'low german',
        'nhi': 'zacatlán-ahuacatlán-tepetzintla nahuatl',
        'nl': 'dutch',
        'no': 'norwegian',
        'nyq': 'nayini',
        'olo': 'livvi-karelian',
        'orv': 'old east slavic',
        'pcm': 'nigerian pidgin',
        'pl': 'polish',
        'pt': 'portuguese',
        'qaf': 'aramaic',
        'qfn': 'kinnauri',
        'qpm': 'south manchurian',
        'qtd': 'quoted',
        'quc': 'kʼicheʼ',
        'ro': 'romanian',
        'ru': 'russian',
        'sa': 'sanskrit',
        'sah': 'yakut',
        'say': 'saya',
        'si': 'sinhala',
        'sl': 'slovenian',
        'sme': 'northern sami',
        'sms': 'skolt sami',
        'soj': 'soi',
        'sq': 'albanian',
        'sr': 'serbian',
        'swl': 'swedish sign language',
        'ta': 'tamil',
        'te': 'telugu',
        'th': 'thai',
        'tl': 'tagalog',
        'tpn': 'tupinambá',
        'tr': 'turkish',
        'tt': 'tatar',
        'ug': 'uyghur',
        'uk': 'ukrainian',
        'ur': 'urdu',
        'vep': 'veps',
        'vi': 'vietnamese',
        'wo': 'wolof',
        'xcl': 'classical tibetan',
        'xnr': 'kangri',
        'xum': 'umbrian',
        'yo': 'yoruba',
        'yrl': 'nheengatu',
        'yue': 'cantonese',
        'zh': 'chinese'
    }