#!/usr/bin/env python
import os
import json
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from transformers import GPT2LMHeadModel

# Special tokens and vocab config
SPECIAL_TOKENS_LIST = ["<pad>", "<s>", "</s>", "<unk>"]
SPECIAL_TOKENS = {tok: i for i, tok in enumerate(SPECIAL_TOKENS_LIST)}
OFFSET = len(SPECIAL_TOKENS_LIST)
PAD_TOKEN_ID = SPECIAL_TOKENS["<pad>"]
BOS_TOKEN_ID = SPECIAL_TOKENS["<s>"]
EOS_TOKEN_ID = SPECIAL_TOKENS["</s>"]
UNK_TOKEN_ID = SPECIAL_TOKENS["<unk>"]
MAX_SEQ_LEN = 256


class MyModel:
    """
    Character-level GPT-2 model for next-character prediction.
    """

    def __init__(self, model=None, char2id=None, id2char=None):
        self.model = model
        self.char2id = char2id or {}
        self.id2char = id2char or []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def load_training_data(cls):
        # Load multilingual C4 dataset (lazy import for inference)
        from datasets import load_dataset, interleave_datasets

        TOTAL_ROWS = 2_000_000
        ENGLISH_ROWS = 1_000_000
        OTHER_ROWS = 10_000

        LANGUAGES = [
            "af", "am", "ar", "az", "be", "bg", "bn", "ca", "ceb", "co", "cs", "cy", "da", "de",
            "el", "eo", "es", "et", "eu", "fa", "fi", "fil", "fr", "fy", "ga", "gd", "gl", "gu",
            "ha", "haw", "iw", "hi", "hmn", "ht", "hu", "hy", "id", "ig", "is", "it", "ja",
            "jv", "ka", "kk", "km", "kn", "ko", "ku", "ky", "la", "lb", "lo", "lt", "lv", "mg",
            "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "no", "ny", "pa", "pl",
            "ps", "pt", "ro", "ru", "sd", "si", "sk", "sl", "sm", "sn", "so", "sq", "sr", "st",
            "su", "sv", "sw", "ta", "te", "tg", "th", "tr", "uk", "und", "ur", "uz", "vi", "xh",
            "yi", "yo", "zh", "zu"
        ]

        en_train = load_dataset("allenai/c4", "en", split="train", streaming=True).select_columns(["text"])
        other_train_list = [
            load_dataset("allenai/c4", lang, split="train", streaming=True).take(OTHER_ROWS).select_columns(["text"])
            for lang in LANGUAGES
        ]
        train_parts = [en_train.take(ENGLISH_ROWS)] + other_train_list
        train_ds = interleave_datasets(train_parts, seed=42, stopping_strategy="all_exhausted")
        return train_ds

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    @staticmethod
    def _build_char_vocab(dataset, num_samples=400000):
        """Build character vocabulary from dataset."""
        char_set = set()
        for example in dataset.take(num_samples):
            for ch in list(example["text"]):
                char_set.add(ch)
        id2char = sorted(char_set)
        char2id = {ch: i + OFFSET for i, ch in enumerate(id2char)}
        return char2id, id2char

    def run_train(self, data, work_dir):
        # Lazy imports for training only
        from transformers import GPT2Config, Trainer, TrainingArguments

        # Build character vocabulary
        print("Building character vocab...")
        self.char2id, self.id2char = self._build_char_vocab(data)
        vocab_size = OFFSET + len(self.id2char)
        print(f"Vocab size: {vocab_size} (chars: {len(self.id2char)})")

        # Initialize model (larger architecture from notebook)
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=MAX_SEQ_LEN,
            n_embd=768,
            n_layer=12,
            n_head=12,
            bos_token_id=BOS_TOKEN_ID,
            eos_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
        )
        self.model = GPT2LMHeadModel(config).to(self.device)
        print(f"Total Parameters: {self.model.num_parameters():,}")

        # Character-level tokenization
        char2id = self.char2id  # Capture for closure

        def chars_tokenize(examples):
            all_encoded = []
            for text in examples["text"]:
                tokens = list(text)
                encoded = [char2id.get(ch, UNK_TOKEN_ID) for ch in tokens][:MAX_SEQ_LEN]
                all_encoded.append(encoded)
            return {"input_ids": all_encoded}

        tokenized_ds = data.map(chars_tokenize, batched=True, remove_columns=["text"])

        # Custom collator for padding
        def manual_collator(features):
            max_len = max(len(f["input_ids"]) for f in features)
            batch_input_ids = []
            batch_attention_mask = []
            batch_labels = []
            for f in features:
                pad_len = max_len - len(f["input_ids"])
                padded_ids = f["input_ids"] + [PAD_TOKEN_ID] * pad_len
                padded_mask = [1] * len(f["input_ids"]) + [0] * pad_len
                labels = f["input_ids"] + [-100] * pad_len
                batch_input_ids.append(padded_ids)
                batch_attention_mask.append(padded_mask)
                batch_labels.append(labels)
            return {
                "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
                "labels": torch.tensor(batch_labels, dtype=torch.long)
            }

        # Training arguments
        BATCH_SIZE = 32
        TOTAL_ROWS = 2_000_000
        MAX_STEPS = TOTAL_ROWS // BATCH_SIZE

        training_args = TrainingArguments(
            output_dir=os.path.join(work_dir, "checkpoints"),
            per_device_train_batch_size=BATCH_SIZE,
            max_steps=MAX_STEPS,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            warmup_steps=500,
            weight_decay=0.1,
            bf16=torch.cuda.is_available(),
            tf32=torch.cuda.is_available(),
            dataloader_num_workers=1,
            save_total_limit=1,
            logging_steps=100,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_ds,
            data_collator=manual_collator
        )

        trainer.train()

    def _encode_text(self, text):
        """Encode text to character-level token ids."""
        tokens = list(text)
        return [self.char2id.get(ch, UNK_TOKEN_ID) for ch in tokens]

    def _decode_id(self, tok_id):
        """Decode token id to character."""
        if tok_id < OFFSET:
            return SPECIAL_TOKENS_LIST[tok_id]
        idx = tok_id - OFFSET
        if idx < len(self.id2char):
            return self.id2char[idx]
        return " "

    def run_pred(self, data):
        preds = []
        self.model.eval()
        self.model.to(self.device)

        for inp in data:
            try:
                # Skip empty or whitespace-only inputs
                if not inp or not inp.strip():
                    preds.append('eta')
                    continue

                # Encode input text to character-level tokens
                encoded = self._encode_text(inp)[-MAX_SEQ_LEN:]
                if not encoded:
                    preds.append('eta')
                    continue

                input_ids = torch.tensor([encoded], dtype=torch.long).to(self.device)

                with torch.no_grad():
                    logits = self.model(input_ids).logits[:, -1, :]  # Last position
                    probs = torch.softmax(logits, dim=-1)

                # Get top 3 predictions
                top_k_probs, top_k_ids = torch.topk(probs, 3)
                top_guesses = []
                for i in range(3):
                    tok_id = top_k_ids[0][i].item()
                    char = self._decode_id(tok_id)
                    # Replace special tokens, newlines, and carriage returns with space
                    if tok_id < OFFSET or char in ('\n', '\r'):
                        char = " "
                    top_guesses.append(char)

                preds.append(''.join(top_guesses))
            except:
                preds.append('eta')

        return preds

    def save(self, work_dir):
        # Save the trained model
        model_path = os.path.join(work_dir, 'nano-char-gpt-c4-v2')
        self.model.save_pretrained(model_path)

        # Save vocabulary
        vocab_path = os.path.join(work_dir, 'vocab.json')
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump({'char2id': self.char2id, 'id2char': self.id2char}, f, ensure_ascii=False)

    @classmethod
    def load(cls, work_dir):
        # Load vocabulary
        vocab_path = os.path.join(work_dir, 'vocab.json')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        char2id = vocab_data['char2id']
        id2char = vocab_data['id2char']

        # Load the trained model
        model_path = os.path.join(work_dir, 'nano-char-gpt-c4-v2')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        return MyModel(model=model, char2id=char2id, id2char=id2char)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instantiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
