import os
import gc
import random
import pprint
import argparse
import numpy as np
from model import *
import tensorflow as tf
from utils import split_types
from framework import ZeroShotMNetFramework
from sentence_encoder import BERTSentenceEncoder
from data_loader import get_loader, MNetDataset, TextProcessor, split_dataset
from multimodal_encoder import MultimodalEncoder, SimpleMultimodalEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="../dataset/MET")
    parser.add_argument(
        "--bert_path", type=str, default="../pretrain/cased_L-12_H-768_A-12"
    )
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--vib_latent_size", type=int, default=768)
    parser.add_argument("--vae_latent_size", type=int, default=768)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--zeta", type=float, default=1.0)
    parser.add_argument("--model_path", type=str, default="./weights")
    parser.add_argument("--task", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    gpus = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)

    task = args.task
    batch_size = args.batch_size
    max_seq_len = args.max_seq_len

    model_name = f"model-task{task}-{args.seed}"
    model_path = f"{args.model_path}/{model_name}.h5"

    dataset = MNetDataset(args.data_path)
    textProcessor = TextProcessor(args.bert_path, max_seq_len)

    if task == 1:
        GroupA = ["People", "Site", "Building", "Currency"]
        GroupB = ["Location", "Event", "Book", "Music"]
        GroupC = ["Organization", "Country", "APP", "Movie"]
    elif task == 2:
        GroupC = ["People", "Site", "Building", "Currency"]
        GroupA = ["Location", "Event", "Book", "Music"]
        GroupB = ["Organization", "Country", "APP", "Movie"]
    elif task == 3:
        GroupB = ["People", "Site", "Building", "Currency"]
        GroupC = ["Location", "Event", "Book", "Music"]
        GroupA = ["Organization", "Country", "APP", "Movie"]

    train_types = GroupA
    val_types = GroupB
    test_types = GroupC
    # train_types, val_types, test_types = split_types(dataset)

    print("train:", train_types)
    print("valid:", val_types)
    print("test:", test_types)

    train_dataset, val_dataset, test_dataset = split_dataset(dataset, train_types, val_types, test_types)

    train_dataloader, train_iter = get_loader(
        train_types, val_types + test_types, train_dataset, textProcessor, batch_size, balanced_sampling=True, do_train=True
    )
    val_dataloader, val_iter = get_loader(train_types, val_types, val_dataset, textProcessor, batch_size)
    test_dataloader, test_iter = get_loader(
        train_types, test_types, test_dataset, textProcessor, batch_size=1, shuffle=False
    )

    del dataset
    gc.collect()

    sentence_encoder = BERTSentenceEncoder(args.bert_path)
    multimodal_encoder = MultimodalEncoder(sentence_encoder, vib_latent_size=args.vib_latent_size, vae_latent_size=args.vae_latent_size)
    model = MMProto(multimodal_encoder, use_img=True, eta=args.eta, zeta=args.zeta)

    framework = ZeroShotMNetFramework(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        train_class=train_types,
        val_class=val_types,
        test_class=test_types,
        beta=args.beta,
    )

    if args.mode == "train":
        threshold_params = framework.train(
            model,
            args.lr,
            args.epoch,
            args.patience,
            train_iter,
            val_iter,
            model_path,
        )
    elif args.mode == "test":
        val_metrics, _, threshold_params = framework.eval(
            model, val_iter, model_path=model_path
        )
        test_metrics = framework.eval(
            model, test_iter, do_test=True, threshold_params=threshold_params
        )[0]
        with open(f"{args.model_path}/{model_name}.txt", "w") as fw:
            pprint.pprint(val_metrics, stream=fw)
            pprint.pprint(test_metrics, stream=fw)

if __name__ == "__main__":
    main()