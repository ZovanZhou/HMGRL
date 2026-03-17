import os
import re
import codecs
import json
import hashlib
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from threading import Thread
from vit_keras import vit
from typing import List, Dict
from tensorflow.data import Dataset
from keras_bert.tokenizer import Tokenizer
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
from keras_bert import TOKEN_CLS, TOKEN_MASK, TOKEN_SEP


class Entity(object):
    """
    Named Entity
    """

    def __init__(self, name, type, h_pos, t_pos, url):
        self.__name = name
        self.__type = type
        self.__h_pos = h_pos
        self.__t_pos = t_pos
        self.__url = url

    @property
    def name(self):
        return self.__name

    @property
    def type(self):
        return self.__type

    @property
    def pos(self):
        return [self.__h_pos, self.__t_pos]

    @property
    def url(self):
        return self.__url

    def __str__(self) -> str:
        return str({"name": self.name, "type": self.type, "pos": self.pos})


class MNetSample(Thread):
    """
    Multimodal Named Entity Typing Sample
    """

    def __init__(self, **kwargs) -> None:
        Thread.__init__(self)
        if "sample" not in kwargs:
            self.__sentence = kwargs["sentence"]
            self.__img_url = kwargs["img_url"]
            self.__data_path = kwargs["data_path"]
            self.__topic = kwargs["topic"]
            self.__entities = [
                Entity(name, type, h_pos, t_pos, url)
                for name, type, h_pos, t_pos, url in kwargs["entity"]
            ]
        else:
            sample = kwargs["sample"]
            self.__sentence = sample.sentence
            self.__image = sample.image
            self.__topic = sample.topic
            self.__entities = kwargs["entity"]

    def run(self):
        file_name = self.__parse_img_file_name(self.__img_url)
        self.__image = self.__load_image(f"{self.__data_path}/wikinewsImgs/{file_name}")

    @property
    def sentence(self):
        return self.__sentence

    @property
    def image(self):
        return self.__image

    @property
    def topic(self):
        return self.__topic

    @property
    def entity(self):
        return self.__entities

    def __str__(self) -> str:
        entities = "".join([str(e) for e in self.entity])
        return str({"sentence": self.sentence, "topic": self.topic, "entity": entities})

    def __parse_img_file_name(self, url: str):
        m_img = url.split("/")[-1]
        prefix = hashlib.md5(m_img.encode()).hexdigest()
        suffix = re.sub(
            r"(\S+(?=\.(jpg|JPG|png|PNG|svg|SVG)))|(\S+(?=\.(jpeg|JPEG)))", "", m_img
        )
        m_img = prefix + suffix
        m_img = m_img.replace(".svg", ".png").replace(".SVG", ".png")
        return m_img

    def __load_image(self, fname: str, image_size: int = 384):
        if not os.path.exists(fname):
            fname = f"{self.__data_path}/wikinewsImgs/17_06_4705.jpg"
        try:
            img = image.load_img(fname, target_size=(image_size, image_size))
        except Exception:
            fname = f"{self.__data_path}/wikinewsImgs/17_06_4705.jpg"
            img = image.load_img(fname, target_size=(image_size, image_size))
        x = image.img_to_array(img)
        # x = preprocess_input(x)
        x = vit.preprocess_inputs(x).reshape(image_size, image_size, 3)
        return x


class MNetDataset(object):
    def __init__(self, data_path: str):
        self.__samples = self.__collect_data(data_path)
        self.__dict_type2samples = self.__parse_relation_dict(self.__samples)

    @property
    def type2sampleIdxs(self):
        return self.__dict_type2samples

    @property
    def types(self):
        return list(self.__dict_type2samples.keys())

    def idx2samples(self, idxs: List):
        samples = []
        for i, j in idxs:
            sample = self.__samples[i]
            entities = [sample.entity[j]]
            samples.append(MNetSample(sample=sample, entity=entities))
        return samples

    def __parse_relation_dict(self, samples: List[MNetSample]) -> Dict:
        dict_type2samples = {}
        for i, sample in enumerate(samples):
            for j, entity in enumerate(sample.entity):
                type = entity.type
                if type not in dict_type2samples:
                    dict_type2samples[type] = []
                dict_type2samples[type].append((i,j))
        return dict_type2samples

    def __collect_data(self, data_path: str) -> List[MNetSample]:
        samples = []
        for dtype in ["train", "valid", "test"]:
            with open(f"{data_path}/{dtype}.json", "r") as fr:
                json_data = json.load(fr)
                for sentence, image_url, topic, entities in tqdm(
                    json_data, ascii=True, ncols=80
                ):
                    sample = MNetSample(
                        sentence=sentence,
                        img_url=image_url,
                        topic=topic,
                        entity=entities,
                        data_path=data_path,
                    )
                    sample.start()
                    samples.append(sample)
                for sample in samples:
                    sample.join()
        return samples


def split_dataset(dataset, train_types, val_types, test_types):
    train_dataset = {k:[] for k in train_types}
    val_dataset = {k:[] for k in train_types + val_types}
    test_dataset = {k:[] for k in train_types + test_types}
    type2sampleIdxs = dataset.type2sampleIdxs

    for train_type in train_types:
        train_type_sampleIdxs = type2sampleIdxs[train_type]
        n_seen_train_val_sample = int(len(train_type_sampleIdxs) * 0.7)
        n_seen_test_sample = len(train_type_sampleIdxs) - n_seen_train_val_sample
        n_seen_train_sample = int(n_seen_train_val_sample * 0.9)
        n_seen_val_sample = n_seen_train_val_sample - n_seen_train_sample

        np.random.shuffle(train_type_sampleIdxs)

        seen_train_val_sampleIdxs =  train_type_sampleIdxs[:n_seen_train_val_sample]
        seen_test_sampleIdxs = train_type_sampleIdxs[-n_seen_test_sample:]
        assert len(seen_train_val_sampleIdxs) + len(seen_test_sampleIdxs) == len(train_type_sampleIdxs)

        seen_train_sampleIdxs = seen_train_val_sampleIdxs[:n_seen_train_sample]
        seen_val_sampleIdxs = seen_train_val_sampleIdxs[-n_seen_val_sample:]
        assert len(seen_train_sampleIdxs) + len(seen_val_sampleIdxs) == len(seen_train_val_sampleIdxs)

        train_dataset[train_type] += dataset.idx2samples(seen_train_sampleIdxs)
        val_dataset[train_type] += dataset.idx2samples(seen_val_sampleIdxs)
        test_dataset[train_type] += dataset.idx2samples(seen_test_sampleIdxs)
    
    for val_type in val_types:
        val_dataset[val_type] += dataset.idx2samples(type2sampleIdxs[val_type])
    
    for test_type in test_types:
        test_dataset[test_type] += dataset.idx2samples(type2sampleIdxs[test_type])

    return train_dataset, val_dataset, test_dataset


class TextProcessor(object):
    def __init__(self, bert_path: str, max_length: int = 128):
        self.__max_length = max_length
        self.__tokenizer = self.__load_tokenizer(bert_path)

    def encode(self, sentence: str, max_seq_len: int):
        indices, segments = self.__tokenizer.encode(first=sentence, max_len=max_seq_len)
        return indices, segments

    def tokenize(self, sentence: str, pos: List):
        h_in_idx, t_in_idx = pos
        context1 = sentence[0:h_in_idx]
        entity = sentence[h_in_idx:t_in_idx]
        context2 = sentence[t_in_idx:]

        tokens_context1 = self.__tokenizer.tokenize(context1)[1:-1]
        tokens_entity = (
            ["[unused1]"] + self.__tokenizer.tokenize(entity)[1:-1] + ["[unused2]"]
        )
        tokens_context2 = self.__tokenizer.tokenize(context2)[1:-1]

        tokens = (
            [TOKEN_CLS]
            + tokens_context1
            + tokens_entity
            + tokens_context2
            + [TOKEN_SEP]
        )
        mask_in_index = tokens.index("[unused1]")

        indices = self.__tokenizer._convert_tokens_to_ids(tokens)
        while len(indices) < self.__max_length:
            indices.append(0)
        indices = indices[: self.__max_length]
        segments = np.zeros_like(indices).tolist()

        return (indices, segments, mask_in_index)

    def __load_tokenizer(self, bert_path):
        token_dict = {}
        with codecs.open(f"{bert_path}/vocab.txt", "r", "utf8") as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        return Tokenizer(token_dict)


class ZeroShotMNetDataset(object):
    def __init__(
        self,
        batch_size: int,
        seen_classes: List[str],
        unseen_classes: List[str],
        dataset: Dict,
        textProcessor: TextProcessor,
        shuffle: bool = False,
        balanced_sampling: bool = False,
        do_train: bool = False
    ):
        self.__seen_classes = seen_classes
        self.__unseen_classes = unseen_classes
        if do_train:
            self.__classes = seen_classes
        else:
            self.__classes = seen_classes + unseen_classes
        self.__batch_size = batch_size
        self.__textProcessor = textProcessor
        (
            self.__dataset,
            self.__sample_index,
            self.__total_step,
        ) = self.__generate_data_index(dataset)
        self.__cur_step = 0
        self.__shuffle = shuffle
        self.__balanced_sampling = balanced_sampling

    @property
    def step(self):
        if self.__batch_size == 1:
            return self.__total_step
        else:
            return self.__total_step // self.__batch_size + 1

    def _generate_data(self):
        dict_sample_index = {}
        if self.__balanced_sampling:
            for i, j, k in self.__sample_index:
                if i not in dict_sample_index:
                    dict_sample_index[i] = []
                dict_sample_index[i].append((i, j, k))
        else:
            sample_index = self.__sample_index
            if self.__shuffle:
                np.random.shuffle(sample_index)
        del self.__sample_index
        while True:
            if self.__balanced_sampling:
                target_class_idx = dict_sample_index[
                    self.__cur_step % len(self.__classes)
                ]
                idx = target_class_idx[
                    np.random.choice(np.arange(len(target_class_idx)))
                ]
            else:
                idx = sample_index[self.__cur_step]
            packed_data = self.__getitem(idx)
            self.__cur_step += 1
            if self.__cur_step == self.__total_step:
                self.__cur_step = 0
                if self.__shuffle:
                    np.random.shuffle(sample_index)
            yield packed_data["sentence_indices"], packed_data[
                "sentence_segments"
            ], packed_data[
                "seen_label_indices"
            ], packed_data[
                "seen_label_segments"
            ], packed_data[
                "unseen_label_indices"
            ], packed_data[
                "unseen_label_segments"
            ], packed_data["mask_in_index"], packed_data[
                "label"
            ], packed_data[
                "img"
            ]

    def __generate_data_index(self, original_dataset):
        dataset = {}
        total_step = 0
        sample_index = []
        for i, class_name in enumerate(self.__classes):
            if class_name not in dataset:
                dataset[class_name] = []
            samples = original_dataset[class_name]
            for j, sample in enumerate(samples):
                dataset[class_name].append(sample)
                entities = sample.entity
                for k, entity in enumerate(entities):
                    total_step += 1
                    sample_index.append((i, j, k))
        return dataset, sample_index, total_step

    def __getitem(self, sample_index):
        i, j, k = sample_index
        target_type = self.__classes[i]
        packed_data = {
            "sentence_indices": [],
            "sentence_segments": [],
            "mask_in_index": [],
            "img": [],
            "seen_label_indices": [],
            "seen_label_segments": [],
            "unseen_label_indices": [],
            "unseen_label_segments": [],
            "label": [],
        }
        sample = self.__dataset[target_type][j]
        sentence = sample.sentence
        image = sample.image
        entity = sample.entity[k]
        (
            sentence_indices,
            sentence_segments,
            mask_in_index
        ) = self.__textProcessor.tokenize(sentence, entity.pos)
        packed_data["sentence_indices"].append(sentence_indices)
        packed_data["sentence_segments"].append(sentence_segments)
        packed_data["mask_in_index"].append(mask_in_index)
        packed_data["img"].append(image)
        packed_data["label"].append(self.__classes.index(target_type))

        for class_name in self.__seen_classes:
            label_indices, label_segments = self.__textProcessor.encode(class_name, 5)
            packed_data["seen_label_indices"].append(label_indices)
            packed_data["seen_label_segments"].append(label_segments)
        for class_name in self.__unseen_classes:
            label_indices, label_segments = self.__textProcessor.encode(class_name, 5)
            packed_data["unseen_label_indices"].append(label_indices)
            packed_data["unseen_label_segments"].append(label_segments)
        return packed_data


def get_loader(
    seen_classes,
    unseen_classes,
    dataset,
    textProcessor,
    batch_size: int,
    shuffle: bool = False,
    balanced_sampling: bool = False,
    do_train: bool = False,
):
    zeroShotMNetDataset = ZeroShotMNetDataset(
        batch_size, seen_classes, unseen_classes, dataset, textProcessor, shuffle, balanced_sampling, do_train
    )
    dataloader = Dataset.from_generator(
        zeroShotMNetDataset._generate_data,
        tuple([tf.int64] * 8 + [tf.float32]),
        tuple(
            [tf.TensorShape([None, None])] * 6
            + [tf.TensorShape([None])] * 2 + [tf.TensorShape([None] * 4)]
        ),
    )
    dataloader = dataloader.batch(batch_size)
    return iter(dataloader), zeroShotMNetDataset.step


if __name__ == "__main__":
    dataset = MNetDataset("./dataset")
    print(dataset.types)
    # def load_tokenizer(bert_path):
    #     token_dict = {}
    #     with codecs.open(f"{bert_path}/vocab.txt", "r", "utf8") as reader:
    #         for line in reader:
    #             token = line.strip()
    #             token_dict[token] = len(token_dict)
    #     return Tokenizer(token_dict)

    # tokenizer = load_tokenizer("../POS-LB-SF-ID/wwm_uncased_L-24_H-1024_A-16")

    # def tokenize_sample(tokenizer, sentence, pos1, pos2):
    #     context1 = sentence[0:pos1]
    #     entity = sentence[pos1:pos2]
    #     context2 = sentence[pos2:]

    #     tokens_context1 = tokenizer.tokenize(context1)[1:-1]
    #     tokens_entity = tokenizer.tokenize(entity)[1:-1]
    #     tokens_context2 = tokenizer.tokenize(context2)[1:-1]
    #     print(tokens_context1)
    #     print(tokens_entity)
    #     print(tokens_context2)
    #     tokens = (
    #         ["[CLS]"]
    #         + tokens_context1
    #         + ["[unused0]"]
    #         + tokens_entity
    #         + ["[unused1]"]
    #         + tokens_context2
    #         + ["[SEP]"]
    #     )
    #     print(tokens)
    #     pos_in_index = len(tokens_context1) + 1
    #     print(pos_in_index)
    #     assert pos_in_index == tokens.index("[unused0]")

    #     indices = tokenizer._convert_tokens_to_ids(tokens)
    #     print(indices)

    #     # while len(indices) < self.__max_length:
    #     #     indices.append(0)
    #     # indices = indices[: self.__max_length]
    #     segments = np.zeros_like(indices).tolist()

    # # sentence = "10 Chris Bond, 11 Ryan Scott and teammates during a time out."
    # # pos1, pos2 = 3, 13
    # # sentence = "Paul Schlesselman self-portrait photo from his MySpace page."
    # # pos1, pos2 = 0, 17
    # sentence = "The home page of nytimes.com."
    # pos1, pos2 = 17, 28
    # tokenize_sample(tokenizer, sentence, pos1, pos2)
