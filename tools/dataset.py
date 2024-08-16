import os
import numpy as np
from aeon.datasets import load_from_tsfile
import pandas as pd
from torch.utils.data import Dataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import SimpleExpSmoothing
import matplotlib.pyplot as plt





'''
DATASET PREPROCESSING
'''


class Standardize(object):
    def __init__(self,feature_columns):
        self.feats = feature_columns
    def __call__(self,series):
        scalar = ColumnTransformer(
        transformers =[("standardize",StandardScaler(),self.feats)],
        remainder = "passthrough"
        )
        ts = scalar.fit_transform(series.transpose()).transpose()
        return ts


class SES2(object):
    def __init__(self,alpha):
        self.alpha = alpha
       
    def __call__(self,input):
        series = sample['series']
        channels,timesteps = series.shape
        out = np.zeros((channels,timesteps),dtype="float32")
        for i in range(channels):
            ts = series[i,:]
            dim_out = SimpleExpSmoothing(ts, initialization_method="heuristic").fit(smoothing_level=self.alpha, optimized=False).fittedvalues
            out[i,:] = np.array(dim_out,dtype="float32")
       
        return {"series":out,"target":sample["target"]}



class DFT_Filter(object):
    def __init__(self,frequency,input_dim,plots=True):
        self.dt = frequency ##need to determine timestep
        self.plots = plots
        self.dim = input_dim
    def _detect_peaks(self,sample):
        ts = sample['series']
        _,n = ts.shape
        out = []
        for dim in range(self.dim):
            fhat = np.fft.rfft(ts[dim],n)
            power_spectral_density = (fhat * np.conj(fhat))/ n
            out.append((fhat,power_spectral_density))
            _,ax = plt.subplots()
            if self.plots:
                freq = (1/(self.dt*n)) * np.arange(n)
                L = np.arange(1,n//2,dtype="int")
                ax.plot(freq[L],power_spectral_density[L],color='c')
            plt.show()
        return out
    def __call__(self,sample):
        res = self._detect_peaks(sample)
        filtered = np.zeros(sample['series'].shape)

        for dim in range(self.dim):
            fhat,psd = res[dim]
            filter_lvl = np.mean(psd)*0.001 ##need to look into literature on how to compute this
            indicies = psd > filter_lvl
            #zeroed_psd = psd * indicies
            zeroed_fhat = fhat * indicies
            filtered_ts = np.fft.irfft(zeroed_fhat)
         
            filtered[dim,:] = filtered_ts
            if self.plots:
                plt.subplot(3,1,dim+1)
                plt.plot(filtered_ts,color="c")
        if self.plots:
            plt.show()
        return {"series":np.array(filtered,dtype="float32"),"target": sample["target"]}



class ClassificationDataset(Dataset):
    '''
    CLASSIFICATION DATASET CLASSES
        assumes datasets are saved in csv files with the form
        (features*,class,subj) or (features*,class)
    '''
    def __init__(self, data_fp, num_classes, num_features, sample_frequency,
                 window_size=5, window_overlap=0.5, transform=None):

        assert os.path.exists(data_fp), \
            f"Data file path '{data_fp}' does not exist."

        # ts file needs thte equal lenght hyperparam
        x, y, meta = load_from_tsfile(data_fp, return_meta_data=True)
        x = x.astype('float32')

        x = x.transpose(0, 2, 1)
        # shape of x: [num_samples, num_timesteps, num_features]
        # shape of y: [num_samples]

        self.num_classes = num_classes
        self.num_features = num_features
        self.sample_frequency = sample_frequency
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.transform = transform

        classes = np.unique(y)
        assert len(classes) == num_classes, "Number of classes do not match"
        class_dict = {classes[i]: i for i in range(num_classes)}

        if x.shape[1] > self.window_size * self.sample_frequency:
            self.clips = []
            for i, ts in enumerate(x):
                self.clips.extend(self._clip(ts, np.array(class_dict[y[i]])))
        elif x.shape[1] == self.window_size * self.sample_frequency:
            self.clips = [(ts, np.array(class_dict[lbl]))
                          for ts, lbl in zip(x, y)]
        else:
            raise ValueError(
                "Time series length is less than window size * "
                "sample frequency")

    def _augment_slice(self, input):
        '''
        input: (series_len,dims)
        output: time series with noise added
        '''
        nsteps, dims = input.shape
        # generate noise centered at 0
        noise = np.random.normal(0, 1, (int(nsteps*0.1), dims))
        # lbls = np.full(())
        return pd.concat([slice, noise])

    def _clip(self, series, act):

        # Create clips from the time series data, with each clip
        # being of length window_size * sample_frequency,
        # with overlap of window_overlap * 100 percent. For example:
        # data   -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # freq -> 1 Hz
        # window -> 5
        # window_overlap -> 0.2
        # clips  -> [[0, 1, 2, 3, 4], [4, 5, 6, 7, 8]]

        window = self.window_size * self.sample_frequency
        # number of steps to slide window on each iter
        step = int(window - window*self.window_overlap)

        curr = 0
        series_list = []
        while curr <= len(series) - window:
            s = series[
                curr:curr + window:
            ]  # need to check dimensions
            curr += step
            series_list.append((s, act))
        return series_list

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        ts, lbl = self.clips[idx]

        # shape of series sample: (channels, timesteps)
        ts = ts.transpose()
        if self.transform:
            ts = self.transform(ts)

        return ts, lbl


class ImputationDataset(ClassificationDataset):
    '''
    IMPUTATION DATASET CLASS DEFINITION
        assumes datasets are saved in csv files with the form
        (features*,class,subj) or (features*,class)
    '''
    def __init__(self,
                 data_fp, num_classes, num_features,
                 sample_frequency, window_size=5, window_overlap=0.5,
                 mean_mask_length=3, masking_ratio=0.15, transform=None):

        super().__init__(data_fp, num_classes, num_features, sample_frequency,
                         window_size, window_overlap, transform)

        self.lm = mean_mask_length
        self.r = masking_ratio

    def __getitem__(self, idx):
        ts, _ = self.clips[idx]
        mask = noise_mask(ts, lm=self.lm, r=self.r)

        ts = ts.transpose 
        mask = mask.transpose()

        if self.transform:
            ts = self.transform(ts)

        ts_masked = ts * mask
        target = copy.deepcopy(ts)

        target_masks = ~mask

        return ts, target, target_masks
