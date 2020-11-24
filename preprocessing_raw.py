import argparse
import random
import jsonlines
import os


def main(args):
    for arg in vars(args):
        print(arg, getattr(args, arg))

    random.seed(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    lines_len = 0
    with jsonlines.open(args.train_file) as f:
        docs = []
        for line in f.iter():
            lines_len += 1
            sentences = []
            for sentence in line['article_original']:
                sentences.append(sentence)
            sentences.append(line['abstractive'].replace('\n', ''))
            docs.append(" ".join(sentences).replace('\n', '') + "\n")

    print("lines", lines_len)

    random.shuffle(docs)

    val_test_cnt = round(len(docs) * args.val_test_ratio)
    print(val_test_cnt)

    val_test_docs = docs[:val_test_cnt]
    train_docs = docs[val_test_cnt:]
    val_docs = val_test_docs[:len(val_test_docs) // 2]
    test_docs = val_test_docs[len(val_test_docs) // 2:]

    print("train", len(train_docs))
    print("val", len(val_docs))
    print("test", len(test_docs))

    with open(os.path.join(args.output_dir, "train.raw"), "w+") as f:
        f.writelines(train_docs)
    with open(os.path.join(args.output_dir, "val.raw"), "w+") as f:
        f.writelines(val_docs)
    with open(os.path.join(args.output_dir, "test.raw"), "w+") as f:
        f.writelines(test_docs)

    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', type=str,
                        default='/media/irelin/data_disk/dataset/dacon_summury/abstractive/train.jsonl')
    parser.add_argument('--output_dir', type=str,
                        default='/media/irelin/data_disk/dataset/dacon_summury/abstractive/preprocessed')
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--val_test_ratio', type=float, default=0.01)

    main(parser.parse_args())
