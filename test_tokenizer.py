import argparse
import glob
import os


def main(args):
    # from tokenizers import BertWordPieceTokenizer
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece

    bert_tokenizer = Tokenizer(WordPiece())
    # bert_tokenizer = Tokenizer(MBartTokenizer())

    from tokenizers import normalizers

    from tokenizers.normalizers import Lowercase, NFD, StripAccents

    bert_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])

    from tokenizers.pre_tokenizers import Whitespace

    bert_tokenizer.pre_tokenizer = Whitespace()

    # from tokenizers.processors import TemplateProcessing
    #
    # bert_tokenizer.post_processor = TemplateProcessing(
    #     single="[CLS] $A [SEP]",
    #     pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    #     special_tokens=[
    #         ("[CLS]", 1),
    #         ("[SEP]", 2),
    #     ],
    # )

    from tokenizers.trainers import WordPieceTrainer

    trainer = WordPieceTrainer(
        vocab_size=10000, special_tokens=["[UNK]", "[CLS]", "[PAD]", "[MASK]"]  # "[SEP]", "[PAD]", "[MASK]"]
    )
    files = glob.glob(args.text_raw_files_pattern)
    bert_tokenizer.train(trainer, files)

    model_files = bert_tokenizer.model.save(args.output_dir, "bert-tokenizer-kr")
    bert_tokenizer.model = WordPiece.from_file(*model_files, unk_token="[UNK]")

    os.makedirs(args.output_dir, exist_ok=True)
    bert_tokenizer.save(os.path.join(args.output_dir, "bert-tokenizer-kr.json"))

    # bert_tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")
    # output = bert_tokenizer.encode_batch(
    #     [["Hello, y'all!", "How are you 😁 ?"], ["Hello to you too!", "I'm fine, thank you!"]]
    # )
    # print(output[1].tokens)
    # print(output[1].attention_mask)
    # output[1].type_ids
    # output[1].ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--text_raw_files_pattern', type=str,
                        default='/media/irelin/data_disk/dataset/dacon_summury/abstractive/preprocessed/*.raw')
    parser.add_argument('--output_dir', type=str,
                        default='/media/irelin/data_disk/dataset/dacon_summury/abstractive/tokenizer/')

    main(parser.parse_args())
