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
        src_docs = []
        tgt_docs = []
        for line in f.iter():
            lines_len += 1
            sentences = []
            for sentence in line['article_original']:
                sentences.append(sentence)
            tgt_docs.append(line['abstractive'].replace('\n', '') + "\n")
            src_docs.append(" ".join(sentences).replace('\n', '') + "\n")

    print("lines", lines_len)

    docs = list(zip(src_docs, tgt_docs))

    random.shuffle(docs)

    val_test_cnt = round(len(docs) * args.val_test_ratio)
    print(val_test_cnt)

    val_test_docs = docs[:val_test_cnt]
    train_docs = docs[val_test_cnt:]
    val_docs = val_test_docs[:len(val_test_docs) // 2]
    test_docs = val_test_docs[len(val_test_docs) // 2:]

    train_source, train_target = zip(*train_docs)
    val_source, val_target = zip(*val_docs)
    test_source, test_target = zip(*test_docs)

    print("train", len(train_source))
    print("val", len(val_source))
    print("test", len(test_source))

    total_source, total_target = zip(*docs)
    with open(os.path.join(args.output_dir, "train.source.total"), "w+") as f:
        f.writelines(total_source)
    with open(os.path.join(args.output_dir, "train.target.total"), "w+") as f:
        f.writelines(total_target)
    with open(os.path.join(args.output_dir, "train.source"), "w+") as f:
        f.writelines(train_source)
    with open(os.path.join(args.output_dir, "train.target"), "w+") as f:
        f.writelines(train_target)
    with open(os.path.join(args.output_dir, "val.source"), "w+") as f:
        f.writelines(val_source)
    with open(os.path.join(args.output_dir, "val.target"), "w+") as f:
        f.writelines(val_target)
    with open(os.path.join(args.output_dir, "test.source"), "w+") as f:
        f.writelines(test_source)
    with open(os.path.join(args.output_dir, "test.target"), "w+") as f:
        f.writelines(test_target)

    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', type=str,
                        default='/media/irelin/data_disk/dataset/dacon_summury/abstractive/train.jsonl')
    parser.add_argument('--output_dir', type=str,
                        default='/media/irelin/data_disk/dataset/dacon_summury/abstractive/preprocessed')
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--val_test_ratio', type=float, default=0.2)

    main(parser.parse_args())
