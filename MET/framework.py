import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from rich.progress import (
    SpinnerColumn,
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def get_ranking_loss(margin: float = 1.0):
    def _loss(y_true, y_pred):
        def _loss_elem(i):
            scores, pos_label = y_pred[i], tf.cast(y_true[i], dtype="int32")
            pos_score = scores[pos_label]
            loss = 0.0
            for j in tf.range(tf.shape(scores)[0]):
                if j != pos_label:
                    neg_score = scores[j]
                    loss += tf.maximum(0.0, margin - pos_score + neg_score)
            return loss
        return tf.map_fn(_loss_elem, tf.range(tf.shape(y_true)[0]), dtype=K.floatx())
    return _loss


class ZeroShotMNetModel(tf.keras.models.Model):
    def __init__(self, encoder, use_img):
        super(ZeroShotMNetModel, self).__init__()
        self.encoder = encoder
        self.use_img = use_img
        self.loss_func = get_ranking_loss()
        # self.loss_func = tf.losses.sparse_categorical_crossentropy

    def __dist__(self, x, y, dim):
        return tf.reduce_sum(x * y, axis=dim)

    def _batch_dist(self, X, Y):
        if len(tf.shape(X)) == 2:
            X = tf.expand_dims(X, axis=1)
        return self.__dist__(X, Y, 2)

    def unpack_data(self, data):
        return data[:-1] if not self.use_img else data

    def call(self, data, n_class):
        raise NotImplementedError

    def loss(self, logits, label, depth):
        return tf.reduce_mean(self.loss_func(label, logits))

    def accuracy(self, pred, label):
        return tf.reduce_mean(tf.cast(pred == label, tf.float32))

    def metrics(self, pred, label):
        p  = precision_score(label, pred, average="weighted", zero_division=0)
        r  = recall_score(label, pred, average="weighted", zero_division=0)
        f1 = f1_score(label, pred, average="weighted", zero_division=0)
        acc = accuracy_score(label, pred)
        return p, r, f1, acc

class ZeroShotMNetFramework(object):
    def __init__(
        self,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        train_class,
        val_class,
        test_class,
        beta,
    ) -> None:
        self.__train_dataloader = train_dataloader
        self.__val_dataloader = val_dataloader
        self.__test_dataloader = test_dataloader
        self.__train_class = train_class
        self.__val_class = val_class
        self.__test_class = test_class
        self.__train_n_class = len(train_class)
        self.__val_n_class = len(val_class)
        self.__test_n_class = len(test_class)
        self.__beta = beta

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[bold red]{task.fields[info]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

    def __get_data(self, dataloader):
        (
            s_ind,
            s_seg,
            s_l_ind,
            s_l_seg,
            u_l_ind,
            u_l_seg,
            mask_idx,
            label,
            img,
        ) = next(dataloader)
        s_len = s_ind.get_shape().as_list()[-1]
        l_len = s_l_ind.get_shape().as_list()[-1]
        dim_img_feature = img.get_shape().as_list()[-3:]
        data = (
            tf.reshape(s_ind, shape=(-1, s_len)),
            tf.reshape(s_seg, shape=(-1, s_len)),
            tf.reshape(s_l_ind, shape=(-1, l_len)),
            tf.reshape(s_l_seg, shape=(-1, l_len)),
            tf.reshape(u_l_ind, shape=(-1, l_len)),
            tf.reshape(u_l_seg, shape=(-1, l_len)),
            tf.reshape(mask_idx, shape=(-1, 1)),
            tf.reshape(img, shape=(-1, *dim_img_feature)),
        )
        label = tf.reshape(label, shape=(-1,))
        return (data, label)

    def __train_model_with_batch(self, model, optimizer, dataloader, n_seen_class, n_unseen_class):
        train_data, train_label = self.__get_data(dataloader)
        with tf.GradientTape() as tape:
            logits, pred, aux_loss = model(train_data, n_seen_class, n_unseen_class, training=True)
            loss = model.loss(logits, train_label, n_seen_class)
            overall_loss = loss + self.__beta * aux_loss
        grads = tape.gradient(overall_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        acc = model.accuracy(pred, train_label)
        return loss, acc

    def train(
        self,
        model,
        lr: float,
        epoch: int,
        patience: int,
        train_iter: int,
        val_iter: int,
        model_path: str,
    ):
        dataloader = self.__train_dataloader
        n_seen_class = self.__train_n_class
        n_unseen_class = self.__val_n_class + self.__test_n_class
        losses = []
        train_accs = []
        val_loss = 0.0
        f1, acc = 0.0, 0.0
        dict_metrics = {}
        best_gamma = -1.0
        best_threshold_params = ()
        min_val_loss = np.inf
        n_patience = 0
        optimizer = tf.optimizers.Adam(learning_rate=lr)
        for e in range(1, epoch + 1):
            losses.clear()
            train_accs.clear()
            train_tqdm = self.progress.add_task(
                description=f"Training epoch {e}",
                total=train_iter,
                info="train_loss:--.--, train_acc:--.--%, val_loss:--.--",
            )
            self.progress.start()
            for _ in range(train_iter):
                loss, train_acc = self.__train_model_with_batch(
                    model, optimizer, dataloader, n_seen_class, n_unseen_class
                )
                train_accs.append(train_acc)
                losses.append(loss)
                info = "train_loss: {0:2.6f}, train_acc: {1:3.2f}%, val_loss: {2:2.6f}".format(
                    np.mean(losses), 100 * np.mean(train_accs), min_val_loss
                )
                self.progress.advance(train_tqdm, advance=1)
                self.progress.update(train_tqdm, info=info)
            val_metrics, val_loss, threshold_params = self.eval(model, val_iter)
            gamma = threshold_params[0]
            if f1 <= val_metrics["overall"]["f1"] or acc <= val_metrics["overall"]["acc"]:
                f1 = val_metrics["overall"]["f1"]
                acc = val_metrics["overall"]["acc"]
                dict_metrics = val_metrics
                n_patience = 0
                min_val_loss = val_loss
                best_gamma = gamma
                best_threshold_params = threshold_params
                self.progress.log("[bold green] Gamma : " + str(best_gamma))
                self.progress.log("[bold green]Best checkpoint")
                info = "Seen: Acc {0:3.2f} F1 {1:3.2f}; Unseen: Acc {2:3.2f} F1 {3:3.2f}; Overall: Acc {4:3.2f} F1 {5:3.2f}".format(
                    dict_metrics["seen"]["acc"] * 100, dict_metrics["seen"]["f1"] * 100,
                    dict_metrics["unseen"]["acc"] * 100, dict_metrics["unseen"]["f1"] * 100,
                    dict_metrics["overall"]["acc"] * 100, dict_metrics["overall"]["f1"] * 100, 
                )
                self.progress.log("[bold blue] Valid result: " + info)
                model.save_weights(model_path)
            else:
                n_patience += 1
                if n_patience == patience:
                    break
        self.progress.log("[bold red]Finish training " + model_path)
        return best_threshold_params

    def _load_model(self, model, model_path):
        if os.path.exists(model_path):
            optimizer = tf.optimizers.Adam()
            self.__train_model_with_batch(
                model, optimizer, self.__train_dataloader, self.__train_n_class, self.__val_n_class + self.__test_n_class
            )
            model.load_weights(model_path, by_name=True)
        else:
            print(f"The model file [{model_path}] are not found !")

    def __selectgamma(self, logits, labels, n_seen_class, dict_class, model):
        from copy import deepcopy
        best_threshold = 0.0
        max_threshold = np.nanmax(logits[:, :n_seen_class])
        step = 1e-3
        threshold = step
        max_f1, max_acc = 0.0, 0.0
        while threshold < max_threshold:
            tmp_logits = deepcopy(logits)
            tmp_logits[:, :n_seen_class] -= threshold
            preds = np.argmax(tmp_logits, axis=-1)
            seen_preds, seen_labels = [], []
            unseen_preds, unseen_labels = [], []
            for p, l in zip(preds, labels):
                if dict_class[l] in self.__train_class:
                    seen_preds.append(p)
                    seen_labels.append(l)
                else:
                    unseen_preds.append(p)
                    unseen_labels.append(l)
            _, _, seen_f1, seen_acc = model.metrics(seen_preds, seen_labels)
            _, _, unseen_f1, unseen_acc = model.metrics(unseen_preds, unseen_labels)
            overall_f1 = 2 * seen_f1 * unseen_f1 / (seen_f1 + unseen_f1)
            overall_acc = 2 * seen_acc * unseen_acc / (seen_acc + unseen_acc)
            if overall_f1 > max_f1 or overall_acc > max_acc:
                best_threshold = threshold
                max_f1 = overall_f1
                max_acc = overall_acc
            threshold += step
        return best_threshold

    # def __selectgamma(self, logits, labels, n_seen_class, dict_class, model):
    #     best_threshold = 0.0

    #     max_seen_logits = np.nanmax(logits[:, :n_seen_class], axis=-1)
    #     max_unseen_logits = np.nanmax(logits[:, n_seen_class:], axis=-1)
    #     coef_logits = max_seen_logits / max_unseen_logits

    #     max_f1, max_acc = 0.0, 0.0
    #     for threshold in coef_logits:
    #         preds = []
    #         for i, entropy in enumerate(coef_logits):
    #             if entropy > threshold:
    #                 preds.append(np.argmax(logits[i, :n_seen_class]))
    #             else:
    #                 preds.append(n_seen_class + np.argmax(logits[i, n_seen_class:]))
    #         seen_preds, seen_labels = [], []
    #         unseen_preds, unseen_labels = [], []
    #         for p, l in zip(preds, labels):
    #             if dict_class[l] in self.__train_class:
    #                 seen_preds.append(p)
    #                 seen_labels.append(l)
    #             else:
    #                 unseen_preds.append(p)
    #                 unseen_labels.append(l)
    #         _, _, seen_f1, seen_acc = model.metrics(seen_preds, seen_labels)
    #         _, _, unseen_f1, unseen_acc = model.metrics(unseen_preds, unseen_labels)
    #         overall_f1 = 2 * seen_f1 * unseen_f1 / (seen_f1 + unseen_f1)
    #         overall_acc = 2 * seen_acc * unseen_acc / (seen_acc + unseen_acc)
    #         if overall_f1 > max_f1 or overall_acc > max_acc:
    #             best_threshold = threshold
    #             max_f1 = overall_f1
    #             max_acc = overall_acc
    #     return best_threshold

    def eval(self, model, val_iter, do_test: bool = False, model_path: str = "", threshold_params: tuple = ()):
        n_seen_class = self.__train_n_class
        if model_path:
            self._load_model(model, model_path)
        if do_test:
            n_unseen_class = self.__test_n_class
            dict_class = self.__train_class + self.__test_class
            dataloader = self.__test_dataloader
        else:
            n_unseen_class = self.__val_n_class
            dict_class = self.__train_class + self.__val_class
            dataloader = self.__val_dataloader

        eval_tqdm = self.progress.add_task(
            description="Evaluating", total=val_iter, info="val_loss:--.--"
        )
        self.progress.start()

        h_embs = []
        labels, logits = [], []
        for _ in range(val_iter):
            val_data, val_label = self.__get_data(dataloader)
            h_emb, logit = model(val_data, n_seen_class, n_unseen_class)
            h_embs.append(h_emb)
            labels.append(val_label)
            logits.append(logit)
            self.progress.advance(eval_tqdm, advance=1)
        h_embs = tf.concat(h_embs, axis=0)
        labels = tf.concat(labels, axis=0)
        logits = tf.concat(logits, axis=0)
        loss = model.loss(logits, labels, n_seen_class+n_unseen_class)

        logits = logits.numpy()
        labels = labels.numpy()

        if not do_test:
            gamma = self.__selectgamma(logits, labels, n_seen_class, dict_class, model)
            min_t = np.nanmax(logits[:, n_seen_class:])
            max_t = np.nanmax(logits[:, :n_seen_class])
            print(max_t, min_t, gamma)
        else:
            gamma, max_t, min_t = threshold_params

        logits[:, :n_seen_class] -= gamma
        preds = np.argmax(logits, axis=-1)

        if True:
            # results = tf.concat(
            #     [tf.expand_dims(preds, axis=1), tf.expand_dims(labels, axis=1)], axis=-1
            # ).numpy()
            # np.savetxt(f"{model_path}.results.csv", results, delimiter=",")
            pd.to_pickle(
                (preds, labels, h_embs.numpy()),
                # h_embs.numpy(),
                f"results.pkl",
            )

        seen_preds, seen_labels = [], []
        unseen_preds, unseen_labels = [], []
        for p, l in zip(preds, labels):
            if dict_class[l] in self.__train_class:
                seen_preds.append(p)
                seen_labels.append(l)
            else:
                unseen_preds.append(p)
                unseen_labels.append(l)
        seen_p, seen_r, seen_f1, seen_acc = model.metrics(seen_preds, seen_labels)
        unseen_p, unseen_r, unseen_f1, unseen_acc = model.metrics(unseen_preds, unseen_labels)
        overall_f1 = 2 * seen_f1 * unseen_f1 / (seen_f1 + unseen_f1)
        overall_acc = 2 * seen_acc * unseen_acc / (seen_acc + unseen_acc)

        info = "val_loss:{0:2.6f}".format(loss)
        self.progress.update(eval_tqdm, info=info)
        dict_metrics = {
            "seen":{
                "p": seen_p,
                "r": seen_r,
                "f1": seen_f1,
                "acc": seen_acc,
            },
            "unseen":{
                "p": unseen_p,
                "r": unseen_r,
                "f1": unseen_f1,
                "acc": unseen_acc,
            },
            "overall":{
                "f1": overall_f1,
                "acc": overall_acc,
            },
        }
        return dict_metrics, loss, (gamma, max_t, min_t)