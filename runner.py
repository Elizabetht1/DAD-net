
import os
import datetime
import torch
import logging
import numpy as np
import hydra
from omegaconf import DictConfig

from tools.utils import (train_val_split, train_epoch, evaluate_epoch,
                         test_epoch,get_dataset)

from tools.plot import (visualize_dataset, visualize_dataloader,
                        plot_confusion_matrix, vizualize_imputation_batch)
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.attention import attentionMTSC

from models.CNNLSTM_dual_network import CNNLSTM_dn
from models.CNNLSTM_layer_wise import CNNLSTM_lw
from models.simpleCNN import simpleCNN
from models.cnn_cw import CNN_cw

from aeon.classification.interval_based import TimeSeriesForestClassifier


sansserif = {'fontname':'sans-serif'}

def load_model(data_cfg, model_cfg, task, debug):
    if model_cfg.name == "attention":
        return attentionMTSC(
            # gives number of time steps
            series_len=int(data_cfg.window_size * data_cfg.sample_frequency),
            # input idms for this dataset
            input_dim=data_cfg.num_features,
            dataset_name=data_cfg.name,
            learnable_pos_enc=model_cfg.learnable_pos_enc,
            classes=data_cfg.num_classes,
            task=task,
            d_model=model_cfg.d_model,
            dropout=model_cfg.dropout,
            dim_ff=model_cfg.feed_forward_dim,
            num_layers=model_cfg.num_layers,
            is_batch=model_cfg.batch_normalization,
            weights_fp=model_cfg.weights_fp,
            debug=debug)

    elif model_cfg.name == "cnn_lw":
        return CNNLSTM_lw(
            # number of features in the series
            data_cfg.num_features,
            model_cfg.num_filters,
            model_cfg.hidden_size,
            data_cfg.num_classes)

    elif model_cfg.name == "cnn_dn":
        return CNNLSTM_dn(
            # number of features in the series
            data_cfg.num_features,
            # gives number of time steps
            int(data_cfg.window_size * data_cfg.sample_frequency),
            model_cfg.num_filters,
            model_cfg.hidden_size,
            data_cfg.num_classes)

    elif model_cfg.name == "cnn":
        return simpleCNN(
            # number of features in the input series
            data_cfg.num_features,
            # gives number of time steps
            int(data_cfg.window_size * data_cfg.sample_frequency),
            model_cfg.num_filters,
            data_cfg.num_classes)

    elif model_cfg.name == "tsf":
        return TimeSeriesForestClassifier(n_estimators=model_cfg.num_trees)
    
    elif model_cfg.name == "cnn_cw":
        return CNN_cw(input_dim=data_cfg.num_features,
                      classes=data_cfg.num_classes,
                      time_steps=int(data_cfg.window_size * data_cfg.sample_frequency),
                      num_filters1=model_cfg.nfilters1,
                      num_filters2=model_cfg.nfilters2,
                      pool1=model_cfg.pool1,
                      pool2=model_cfg.pool2,
                      dim_ff=model_cfg.dimff
                      )
    
    else:
        raise Exception("unrecognized model.")


def is_supervised(task_name):
    supervised = ["classification"]
    return task_name in supervised


def is_early(mod_name):
    early = ['tsf']
    return mod_name in early


class Runner:
    def __init__(self, data_cfg, model_cfg, task, lr, batch_size, num_epochs,
                 num_workers, output_path, log=True, plots=False, debug=False):

        self.dataset_name = data_cfg.name
        self.model_name = model_cfg.name
        self.task = task

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_workers = num_workers

        self.output_path = output_path
        self.logging = log
        self.plots = plots
        self.debug = debug

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = load_model(
            data_cfg, model_cfg, task, debug)

        if not is_early(self.model_name):
            self.model.to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_torch(self, ds):
        train_set, val_set = train_val_split(ds)
        if self.task == "classification":
            train_loader = DataLoader(
                train_set, batch_size=self.batch_size, shuffle=True,
                num_workers=self.num_workers, drop_last=True)
            val_loader = DataLoader(
                val_set, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, drop_last=True)
        elif self.task == "imputation":
            train_loader = DataLoader(
                train_set, batch_size=self.batch_size, shuffle=True,
                num_workers=self.num_workers, drop_last=True)
            val_loader = DataLoader(
                val_set, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, drop_last=True)
        else:
            raise Exception("unrecognized task.")

        assert len(val_loader) > 0 and len(train_loader) > 0, \
            "batch size too large for dataset length"

        if self.plots and is_supervised(self.task):
            os.makedirs(os.path.join("./plots",
                                     self.dataset_name,
                                     "dataset_summary",
                                     self.model_name), exist_ok=True)
            visualize_dataloader(
                train_loader,
                os.path.join("./plots",
                             self.dataset_name,
                             "dataset_summary",
                             self.model_name,
                             "train_dataloader.png"))
            visualize_dataloader(
                val_loader,
                os.path.join("./plots",
                             self.dataset_name,
                             "dataset_summary",
                             self.model_name,
                             "val_dataloader.png"))

        train_loss = []
        val_loss = []

        for i in range(self.num_epochs):

            train_epoch_loss = train_epoch(
                self.model, self.optimizer, train_loader,
                self.device, task=self.task)
            train_loss.append(train_epoch_loss)

            if len(val_loader) > 0:
                val_epoch_loss, epoch_results = evaluate_epoch(
                    self.model, val_loader, self.device, self.task)
                val_loss.append(val_epoch_loss)

                if self.logging:
                    logging.info(
                        "Epoch %d: average loss - %.2f" % (i, val_epoch_loss))
                print("Epoch %d: average loss - %.2f" % (i, val_epoch_loss))

                if (i + 1) % 10 == 0 and self.plots and \
                        not is_supervised(self.task):
                    save_folder = os.path.join(
                        "./plots", self.dataset_name, "imputation",
                        self.model_name)
                    os.makedirs(save_folder, exist_ok=True)
                    vizualize_imputation_batch(
                        preds=epoch_results['preds'],
                        targets=epoch_results['targets'],
                        masks=epoch_results['masks'],
                        out_fp=os.path.join(
                            save_folder, f"imputation_{i + 1}.png")
                    )

        if self.plots:
            plt.plot(train_loss, color='r', label='train loss')
            plt.plot(val_loss, color='b', label='validation loss')
            plt.legend(loc="upper right")
            plots_dir = os.path.join(
                "./plots", self.dataset_name, "loss_train_valid",
                self.model_name, self.task)
            os.makedirs(plots_dir, exist_ok=True)
            plt.savefig(os.path.join(plots_dir, "loss_graph.png"))
            plt.close()

    def ds2np(self,ds):
        out = [(x,y) for x,y in ds]
        X = np.array([x for x,_ in out])
        Y = np.array([y for _,y in out])
        return X,Y

    def train_early(self,ds):
        trainX,trainY = self.ds2np(ds)
        self.model.fit(trainX,trainY)

    def test_early(self,ds):
        testX,testY = self.ds2np(ds)
        pred_classes = self.model.predict(testX)
        classes, counts = np.unique(pred_classes == testY,return_counts = True)
        acc = {c:count/np.sum(counts) for c,count in zip(classes,counts)}
        if self.plots and is_supervised(self.task):
            # plot confusion matrix (TODO: have not tested yet)
            save_folder = os.path.join(
                "./plots", self.dataset_name, "confusion_matrix",
                self.model_name, self.task)
            os.makedirs(save_folder, exist_ok=True)

            plot_confusion_matrix(pred_classes, testY,
                                  os.path.join(save_folder, "cm.png"))

        final_acc = acc[True]
        if self.logging:
            logging.info("FINAL MODEL ACCURACY: %.2f%%" % (final_acc*100))

        print("FINAL MODEL ACCURACY: %.2f%%" % (final_acc*100))

    def test_torch(self, dataset):

        full_results = {'preds': [], 'targets': []}
        total_acc = 0.0
        total_elems = 0

        if self.task == "classification":
            test_loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers,
                drop_last=True)
        elif self.task == "imputation":
            test_loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, drop_last=True)
        else:
            raise Exception("unrecognized task.")

        if self.plots and is_supervised(self.task):
            os.makedirs(
                os.path.join("./plots", self.dataset_name,
                             "dataset_summary", self.model_name),
                exist_ok=True)
            visualize_dataloader(
                test_loader,
                os.path.join("./plots", self.dataset_name, "dataset_summary",
                             self.model_name, "test_dataloader.png"))

        for _ in range(self.num_epochs):
            batch_acuarcy, batch_elems_count, epoch_results = \
                test_epoch(self.model, test_loader, self.device,
                           self.task, batch_size=self.batch_size,
                           plots=self.plots)
            full_results['preds'].extend(epoch_results['preds'])
            full_results['targets'].extend(epoch_results['targets'])
            total_acc += batch_acuarcy
            total_elems += batch_elems_count

        if self.plots and is_supervised(self.task):
            # plot confusion matrix
            y_preds = full_results['preds']
            y_trues = full_results['targets']

            save_folder = os.path.join(
                "./plots", self.dataset_name, "confusion_matrix",
                self.model_name, self.task)
            os.makedirs(save_folder, exist_ok=True)

            plot_confusion_matrix(y_preds, y_trues,
                                  os.path.join(save_folder, "cm.png"))

        final_acc = total_acc / total_elems
        if self.logging:
            logging.info("FINAL MODEL ACCURACY: %.2f%%" % (final_acc*100))

        print("FINAL MODEL ACCURACY: %.2f%%" % (final_acc*100))

    def train_model(self, dataset):

        if self.logging:
            logging.info("======== TRAINING MODEL ======== ")

        if is_early(self.model_name): 
            self.train_early(dataset)
        else: 
            self.train_torch(dataset)

    def test_model(self, dataset):
        if self.logging:
            logging.info("======== TESTING MODEL ======== ")

        if is_early(self.model_name): 
            self.test_early(dataset)
        else: 
            self.test_torch(dataset)

    def get_model(self):
        return self.model

    def save_model(self):
        if self.logging:
            logging.info("======== SAVING MODEL ======== ")
        os.makedirs(os.path.join(self.output_path, self.dataset_name),
                    exist_ok=True)
        torch.save(
            self.model.state_dict,
            os.path.join(self.output_path, self.dataset_name,
                         f"{self.model_name}_model_weights_{self.task}.pth"))


@hydra.main(config_path="conf", config_name="", version_base="1.3")
def run(cfg: DictConfig):
    if cfg.log:
        def modLog(modName: str, params: dict) -> None:
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            if (logger.hasHandlers()):
                logger.handlers.clear()

            dir_fp = "./logs"
            os.makedirs(dir_fp, exist_ok=True)
            log_fp = os.path.join(dir_fp, f"{modName}.log")
            file_handler = logging.FileHandler(log_fp)
            logger.addHandler(file_handler)
            current_time = datetime.datetime.now()
            current_time.strftime("%D : %I:%M:%S %p")

            logging.info("===========================================")
            logging.info(f"###          MODEL: {modName}        ###")
            logging.info(f"###    START_TIME: {current_time}    ###")

            logging.info(" loading data. ")
            logging.info("###      Parameters.    ###")
            for key, value in params.items():
                logging.info(f"--> {key} : {value}")

        modLog(cfg.model.name, cfg)
        logging.info("======== MODEL CONFIGURATION ======== ")
        logging.info(f"task: {cfg.task}.")
        logging.info(f"device: {cfg.device}.")
        logging.info(f"======== LOADING DATASET: {cfg.dataset.name} ======== ")

    train_dataset, test_dataset = get_dataset(cfg.dataset, task=cfg.task)
    if cfg.plots and is_supervised(cfg.task):
        os.makedirs(
            os.path.join("./plots", cfg.dataset.name, "dataset_summary"),
            exist_ok=True)
        visualize_dataset(train_dataset,
                          os.path.join("./plots", cfg.dataset.name,
                                       "dataset_summary", "train_data"))
        visualize_dataset(test_dataset,
                          os.path.join("./plots", cfg.dataset.name,
                                       "dataset_summary", "test_data"))

    runner = Runner(
        data_cfg=cfg.dataset,
        model_cfg=cfg.model,
        task=cfg.task,
        lr=cfg.hyp.learning_rate,
        batch_size=cfg.hyp.batch_size,
        num_epochs=cfg.hyp.num_epochs,
        output_path=cfg.path.output_path,
        num_workers=cfg.device.num_workers,
        log=cfg.log,
        plots=cfg.plots,
        debug=cfg.debug)

    runner.train_model(train_dataset)
    runner.test_model(test_dataset)
    if cfg.save:
        runner.save_model()


if __name__ == "__main__":
    run()
