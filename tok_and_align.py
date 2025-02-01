from datasets import Dataset
from tqdm import tqdm
from utils import get_dataset
from transformers import AutoTokenizer

# Enable or disable debug mode
debug = False  # Set to False to suppress debug prints

def tokenize_and_align_labels_m(dataset, tokenizer,padd_len=325):
    """
    Tokenizes the dataset and aligns token labels while preserving word-to-token mapping.

    Args:
        dataset (list of dict): A dataset where each entry contains:
            - 'tokens': List of words in the sentence.
            - 'anim_tags': List of animacy labels for target words.
            - 'target_indexes': List of indexes of words that need labels.
        tokenizer (AutoTokenizer): Tokenizer from Hugging Face.

    Returns:
        dict: Tokenized dataset with aligned labels.
    """

    # Debug: Show dataset information
    if debug:
        print("\n### DATASET OVERVIEW ###")
        dataset = dataset[:2]
        print("Number of Sentences:", len(dataset))
        print("Sample Entry:", dataset[0] if len(dataset) > 0 else "Empty Dataset")


    # Tokenize all sentences while maintaining word alignment
    tokenized_data = tokenizer(
        [entry["tokens"] for entry in dataset],
        is_split_into_words=True,
        padding='max_length', 
        max_length=padd_len
    )

    # Debug: Show tokenization output
    if debug:
        print("\n### TOKENIZED DATA OVERVIEW ###")
        print("Keys:", tokenized_data.keys())
        print("First Sentence Tokenized:", tokenizer.convert_ids_to_tokens(tokenized_data["input_ids"][0]))

    aligned_labels = []
    aligned_animacy = []

    for sentence_idx, entry in tqdm(enumerate(dataset), total=len(dataset)):
        target_word_positions = entry["target_indexes"]
        word_to_token_mapping = tokenized_data.word_ids(batch_index=sentence_idx)

        # Debug: Print word-token mapping
        if debug:
            print(f"\n### Sentence {sentence_idx + 1} ###")
            print("Original Words:", entry["tokens"])
            print("Tokenized Words:", tokenizer.convert_ids_to_tokens(tokenized_data["input_ids"][sentence_idx]))
            print("Target Word Positions:", target_word_positions)
            print("Word-to-Token Mapping:", word_to_token_mapping)

        token_labels = []
        animacy_labels = []

        for token_idx, token_id in enumerate(tokenized_data["input_ids"][sentence_idx]):
            #print('Token idx: ', token_idx)
            #print('Token id: ', token_id)
            if word_to_token_mapping[token_idx] is not None:
                mapped_word_position = word_to_token_mapping[token_idx]
            else:
                mapped_word_position = None

            #print('Mapped word position: ', mapped_word_position)

            if mapped_word_position in target_word_positions:
                #print('Mapped in target!')
                target_index = target_word_positions.index(mapped_word_position)
                #print('Target index: ', target_index)
                token_labels.append(token_id)
                #print('Anim lab: ', entry["anim_tags"][target_index] + 1)
                animacy_labels.append(entry["anim_tags"][target_index] + 1)
            else:
                token_labels.append(-100)  # Ignored token
                if tokenized_data["attention_mask"][sentence_idx][token_idx] == 0:
                    animacy_labels.append(4)  # Padding token
                else:
                    animacy_labels.append(0)  # Default non-animacy

        aligned_labels.append(token_labels)
        aligned_animacy.append(animacy_labels[1:] + [4])  # Shifted animacy

    tokenized_data["labels"] = aligned_labels
    tokenized_data["animacy"] = aligned_animacy

    # Debug: Print final processed data
    if debug:
        print("\n### FINAL PROCESSED DATA ###")
        print("First Sentence Labels:", aligned_labels[0] if aligned_labels else "No Data")
        print("First Sentence Animacy:", aligned_animacy[0] if aligned_animacy else "No Data")

    return tokenized_data

if __name__ == "__main__":
    # Load dataset and tokenizer
    dataset = get_dataset('it')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Run the function
    processed_dataset = tokenize_and_align_labels_m(dataset['train'], tokenizer)