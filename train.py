
import numpy as np
import torch
# import matplotlib.pyplot as plt
import torchaudio
import sys, os
sys.path.append('torchaudio-contrib')
import tac
from torch import nn, optim
from time import time
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
from models import *

import argparse

import logging

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--basepath', metavar='DIR',
                    help='path to train manifest csv',
                    default='/export02/data/peterd//Projects/speechsamples/Librispeech_exp/data/')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='./data/identification_train.csv')
parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to test manifest csv', default='./data/identification_test.csv')
parser.add_argument('--lrate', type=float,
                    help='path to test manifest csv', default=1e-3)
parser.add_argument('--n-future-p', type=int,
                    help='path to test manifest csv', default=1)
parser.add_argument('--n-future-w', type=int,
                    help='path to test manifest csv', default=0)
parser.add_argument('--speakers', type=int,
                    help='path to test manifest csv', default=0)
parser.add_argument('--cutoff-temp', type=int,
                    help='path to test manifest csv', default=0)
parser.add_argument('--cutoff-spec', type=int,
                    help='path to test manifest csv', default=0)
parser.add_argument('--rndseed', type=int,
                    help='path to test manifest csv', default=20102018)
parser.add_argument('--n-train', type=int,
                    help='path to test manifest csv', default=25817)
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
args = parser.parse_args()

# basepath = '/export02/data/peterd/Projects/speechsamples/Librispeech_exp/data/'
# lr0 = 1e-3
# cutoff_temp = 0
# cutoff_spec = 2
# cuda = False
# n_future_p = 1
# n_future_w = 0
# speakers = False
# rndseed = 20102018
# n_train = 25817
# n_train = 8640
# ii = 1

basepath = args.basepath
lr0 = args.lrate
n_future_p = args.n_future_p
n_future_w = args.n_future_w
speakers = args.speakers
cutoff_temp = args.cutoff_temp
cutoff_spec = args.cutoff_spec
cuda = args.cuda
rndseed = args.rndseed
n_train = args.n_train

manifest_filepath = './converted_aligned_phones.txt'
librispeech_path = basepath+'/LibriSpeech/train-clean-100/'
n_phones = 41
n_words = 7727
grad_clip = 1.

torch.manual_seed(rndseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(rndseed)

experiment_string = 'lr_{}_nfuturep_{}_nfuturew_{}_speakers_{}_ctemp_{}_cspec_{}_ntrain_{}_{}'.format(
    int(lr0*10000), n_future_p, n_future_w, speakers, cutoff_temp, cutoff_spec, n_train, rndseed)
# ii = 1
# while os.path.isfile('log_'+experiment_string+'.log'):
#     experiment_string = experiment_string[:-1]+str(ii)
#     ii +=1
logging.basicConfig(filename='log_{}.log'.format(experiment_string), filemode='w', format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(args)

# basepath = '/local_raid/data/peterd'
manifest_filepath = './data/converted_aligned_phones.txt'
manifest_filepath_p = './data/alignments_wordcount8_phones.txt'
manifest_filepath_w = './data/alignments_wordcount8_words.txt'

# Sort speaker ids, make index
pids = np.sort([int(pid) for pid in os.listdir(librispeech_path)])
pid_index = torch.LongTensor(np.zeros(pids.max()+1, dtype=int))
for i_pid, pid in enumerate(pids):
    pid_index[pid] = i_pid

# In[83]:


from torch.utils.data import Dataset

def get_librispeech_filepath(librispeech_path, filecode):
    fparts = filecode.split('-')
    return os.path.join(librispeech_path, fparts[0], fparts[1], filecode+'.flac')

def compute_next_items(phones, words):

    phones2 = torch.ones_like(phones)
    phones3 = torch.ones_like(phones)
    words2 = torch.ones_like(words)

    curr_phone = phones[0].item()
    curr_word = words[0].item()
    w_break = 0
    p_break = 0
    for i_p in range(1,len(phones)):
        if not (words[i_p].item()==curr_word):
            words2[range(w_break,i_p)] = words[i_p].item()
            w_break = i_p
            curr_word = words[i_p].item()
        if not (phones[i_p].item()==curr_phone):
            phones2[range(p_break,i_p)] = phones[i_p].item()
            p_break = i_p
            curr_phone = phones[i_p].item()

    curr_phone = phones2[0].item()
    p_break = 0
    for i_p in range(1,len(phones)):
        if not (phones2[i_p].item()==curr_phone):
            phones3[range(p_break,i_p)] = phones2[i_p].item()
            p_break = i_p
            curr_phone = phones2[i_p].item()

    return phones2, phones3, words2

class LibriSpeechDataset(Dataset):
    def __init__(self, manifest_filepath, librispeech_path, chunk=None, index=None, sample_rate=16000,
                 normalize=True, train=True, manifest_filepath_w=None):
        """
        """
        with open(manifest_filepath) as f:
            afiles = f.readlines()
        if manifest_filepath_w is not None:
            with open(manifest_filepath_w) as f:
                afiles_w = f.readlines()
        files = []
        # print(len(afiles))
        for i_file, afile in enumerate(afiles):
            # print(i_file)
            asplit = afile.split(' ')
            filecode = asplit[0]
            speaker_id = int(filecode.split('-')[0])
            phones = torch.LongTensor([int(phone) for phone in asplit[1:]])
            asplit_w = afiles_w[i_file].split(' ')
            words = torch.LongTensor([int(word)-1 for word in asplit_w[1:]])
            # phones2, phones3, words2 = compute_next_items(phones, words)
            audiolen = len(phones)*.01
            if len(phones)<=(chunk*100+1):
                continue
            else:
                files.append([filecode, speaker_id, phones, phones, phones, audiolen, words, words])
        if index is None:
            self.files = files
        else:
            self.files = [files[ii] for ii in index]
        self.size = len(self.files)
        self.librispeech_path = librispeech_path
        self.train = train
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.chunk = chunk
        super(LibriSpeechDataset, self).__init__()

    def parse_audio(self, audio_path, index=None):
        sound, _ = torchaudio.load(audio_path, normalization=self.normalize)
        if index is not None:
            sound = sound[:,index]
        return sound

    def __getitem__(self, index):
        sample = self.files[index]
        filecode, speaker_id, phones, phones2, phones3, audiolen, words, words2 = tuple(sample)
        # filecode, speaker_id, phones, audiolen, words = sample[0], sample[1], sample[2], sample[3], sample[4]
        # index 3 second window
        if self.chunk is not None:
            if self.train:
                offset = np.random.uniform(0, float(audiolen)-self.chunk)
                offset = int(np.floor(offset*100))
            else:
                offset = 0
            index_phones = np.arange(offset, offset+int(self.chunk*100)+1)
            offset = offset*(self.sample_rate/100)
            index = np.arange(offset, offset+self.sample_rate*self.chunk)
        else:
            index = None
        # phones2, phones3, words2 = compute_next_items(phones, words)
        y = self.parse_audio(get_librispeech_filepath(self.librispeech_path, filecode), index=index)
        target = (pid_index[speaker_id], phones[index_phones], phones2[index_phones], phones3[index_phones],
                  words[index_phones], words2[index_phones])
        return y, target

    def __len__(self):
        return self.size

class ModFilter(nn.Module):

    def __init__(self, temporal, spectral):
        super(ModFilter, self).__init__()
        self.temporal = temporal
        self.spectral = spectral

    def forward(self, x):
        spec_fft = torch.rfft(x, 2)
        spec_fft_abs = torch.sqrt(spec_fft[:,:,:,:,0]**2+spec_fft[:,:,:,:,1]**2)
        gainmap = torch.torch.zeros_like(spec_fft[:,:,:,:,0])
        if self.temporal is not 0:
            gainmap[:,0,:,:self.temporal] = 1
            gainmap[:,0,:,self.temporal] = .5
        if self.spectral is not 0:
            gainmap[:,0,:self.spectral,:] = 1
            gainmap[:,0,self.spectral,:] = .5
            gainmap[:,0,-self.spectral:,:] = 1
            gainmap[:,0,-self.spectral,:] = .5
        spec_fft_new = spec_fft*gainmap[:,:,:,:,None].repeat(1,1,1,1,2)
        spec_new = torch.irfft(spec_fft_new, 2)
        return spec_new

class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        np.random.shuffle(ids)
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids)-len(ids)%batch_size, batch_size)]


    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)

def amplitude_to_db(x, ref=1.0, amin=1e-7):
    """
    Note: Given that FP32 is used and its corresponding `amin`,
    we do not implement further numerical stabilization for very small inputs.
    :param x: torch.tensor, the input value
    :param ref:
    :param amin: float
    :return: torch.tensor, same size of x, but decibel-scaled
    """
    x = torch.clamp(x, min=amin)
    return 10.0 * (torch.log10(x) - torch.log10(torch.tensor(ref, device=x.device, requires_grad=False)))

def train(loader):

    model.train()
    avg_losses = [0. for ii in range(model.n_losses)]
    epoch_start = time()
    train_sampler.shuffle(epoch)

    for i, (data) in enumerate(train_loader, start=0):

        data = (data[0].to(device), tuple([d.to(device) for d in data[1]]))
        # get audio signal and multiple targets
        y, targets = data
        # compute mel-spectrogram
        spec = mel(y)
        spec = amplitude_to_db(spec)
        # apply modulation filter
        if is_modfilter:
            spec = modfilter(spec)
        # run through CNN/RNN
        out = model(spec)
        # compute losses
        losses = model.loss(out, targets)
        avg_losses = [al+l.item() for al, l  in zip(avg_losses, losses)]
        # compute gradient wrt summed losses
        optimizer.zero_grad()
        sum(losses).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if (i>0) and (i%20==0):
            logger.info('Batch {} of {}'.format(i, n_batch))
            for i_loss, avg_loss in enumerate(avg_losses):
                logger.info('Average loss {}: {}'.format(i_loss, avg_loss/(i+1)))
            logger.info('Elapsed time: {} seconds'.format(time()-epoch_start))

    return avg_losses, i

def evaluate(loader):
    model.eval()
    avg_losses = [0. for ii in range(model.n_losses)]
    avg_accuracies = [0. for ii in range(model.n_losses)]
    epoch_start = time()
    logger.info('Testing..')
    for i, (data) in enumerate(val_loader, start=0):

        logger.info(i)
        data = (data[0].to(device), tuple([d.to(device) for d in data[1]]))
        y, targets = data
        spec = mel(y)
        spec = amplitude_to_db(spec)
        if is_modfilter:
            spec = modfilter(spec)
        out = model(spec)
        losses = model.loss(out, targets)
        accuracies = model.accuracy(out, targets)
        avg_losses = [al+l.item() for al, l  in zip(avg_losses, losses)]
        avg_accuracies = [aa+a for aa, a  in zip(avg_accuracies, accuracies)]
    return avg_losses, avg_accuracies, i

# ==== Load data ==== #

train_dataset = LibriSpeechDataset(manifest_filepath_p, librispeech_path, 3, index=None,
                                   manifest_filepath_w=manifest_filepath_w)
n_files = len(train_dataset)

rand = np.random.RandomState(20102018)
file_index = np.arange(n_files)
rand.shuffle(file_index)
train_index = file_index[:-640*2]
train_index = file_index[:n_train]
val_index = file_index[-640*2:-640]
test_index = file_index[-640:]

batch_size = 64
train_dataset = LibriSpeechDataset(manifest_filepath_p, librispeech_path, 3, index=train_index,
                                   manifest_filepath_w=manifest_filepath_w)
train_sampler = BucketingSampler(train_dataset, batch_size=batch_size)
train_loader = DataLoader(train_dataset, num_workers=1, batch_sampler=train_sampler)

val_dataset = LibriSpeechDataset(manifest_filepath_p, librispeech_path, 3, index=val_index, train=False,
                                  manifest_filepath_w=manifest_filepath_w)
val_sampler = BucketingSampler(val_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, num_workers=1, batch_sampler=val_sampler)

test_dataset = LibriSpeechDataset(manifest_filepath_p, librispeech_path, 3, index=test_index, train=False,
                                  manifest_filepath_w=manifest_filepath_w)
test_sampler = BucketingSampler(test_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, num_workers=1, batch_sampler=test_sampler)

# ==== Load model ==== #

if speakers==1:
    n_s = len(pids)
elif speakers==0:
    n_s = 0
if n_future_w>0:
    model = SpecNetPW(embed_size=256, n_p=41, n_future_p=n_future_p, n_w=7727, n_future_w=n_future_w, n_s=n_s)
else:
    model = SpecNetP(embed_size=256, n_p=41, n_future_p=n_future_p, n_s=n_s)

logger.info(model)

mel = tac.layers.Melspectrogram(128, train_dataset.sample_rate, hop=int(.01*train_dataset.sample_rate), n_fft=2**10)
modfilter = ModFilter(cutoff_temp, cutoff_spec)
if (cutoff_spec==0) and (cutoff_temp==0):
    is_modfilter = False
else:
    is_modfilter = True
if cuda:
    model.cuda()
    mel = mel.cuda()
    modfilter = modfilter.cuda()
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# ==== Training ==== #

optimizer = optim.Adam(model.parameters(), lr=lr0)
n_batch = int(len(train_dataset)/batch_size)

train_loss, val_loss, val_accuracy = ([], [], [])
best_val_losses = [0. for ii in range(model.n_losses)]
for epoch in range(70):
    logger.info('Epoch {}'.format(epoch))
    [l.append([]) for l in [train_loss, val_loss, val_accuracy]]
    # Training loop
    avg_losses, i = train(train_loader)
    for i_loss, avg_loss in enumerate(avg_losses):
        train_loss[-1].append(avg_loss/(i+1))
    # Validation loop
    avg_losses, avg_accuracies, i = evaluate(val_loader)
    # Check whether to save the current model, or decrease the learning rate
    is_best = False
    for i_loss, (avg_loss, avg_accuracy) in enumerate(zip(avg_losses, avg_accuracies)):
        logger.info('Validation loss {}: {}'.format(i_loss, avg_loss/(i+1)))
        logger.info('Validation accuracy {}: {}'.format(i_loss, avg_accuracy/(i+1)))
        val_loss[-1].append(avg_loss/(i+1))
        val_accuracy[-1].append(avg_accuracy/(i+1))
        if (epoch==0) or (avg_loss/(i+1) < best_val_losses[i_loss]):
            best_val_losses[i_loss] = avg_loss/(i+1)
            is_best = True
    if is_best:
        logger.info('Saving model')
        with open('./state_dict_{}.pkl'.format(experiment_string), 'wb') as f:
            torch.save(model.state_dict(), f)
    else:
        lr0 /= 4
        logger.info('Decreasing learning rate: {}'.format(lr0))
        for g in optimizer.param_groups:
            g['lr'] /= 4

# Test
avg_losses, avg_accuracies, i = evaluate(test_loader)
for i_loss in range(len(avg_losses)):
    avg_losses[i_loss] = avg_losses[i_loss]/(i+1)
    avg_accuracies[i_loss] = avg_accuracies[i_loss]/(i+1)
    logger.info('Test loss {}: {}'.format(i_loss, avg_losses[i_loss]))
    logger.info('Test accuracy {}: {}'.format(i_loss, avg_accuracies[i_loss]))

np.savez('./results_{}.npz'.format(experiment_string),
    train_loss=np.array(train_loss),
    val_loss=np.array(val_loss),
    val_accuracy=np.array(val_accuracy),
    test_loss=np.array(avg_losses),
    test_accuracy=np.array(avg_accuracies))
