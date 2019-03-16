import numpy as np
import torch
import matplotlib.pyplot as plt
import sys, os
sys.path.append('./torchaudio-contrib')
# sys.path.append('audio')
# sys.path.append('audio/torchaudio')
import torchaudio
import tac
from torch import nn, optim
from time import time
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--basepath', metavar='DIR',
        help='path to train manifest csv', default='/meg/meg1/users/peterd/')
parser.add_argument('--train-manifest', metavar='DIR',
        help='path to train manifest csv', default='./data/identification_train.csv')
parser.add_argument('--test-manifest', metavar='DIR',
        help='path to test manifest csv', default='./data/identification_test.csv')
parser.add_argument('--lrate', type=float,
        help='path to test manifest csv', default=1e-3)
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
args = parser.parse_args()

# basepath = '/local_raid/data/peterd'
basepath = args.basepath
manifest_filepath = './converted_aligned_phones.txt'
librispeech_path = basepath+'/LibriSpeech/train-clean-100/'
lr0 = args.lrate

from torch.utils.data import Dataset

def get_librispeech_filepath(librispeech_path, filecode):
    fparts = filecode.split('-')
    return os.path.join(librispeech_path, fparts[0], fparts[1], filecode+'.flac')

class LibriSpeechDataset(Dataset):
    def __init__(self, manifest_filepath, librispeech_path, chunk=None, index=None, sample_rate=16000, normalize=True):
        """
        """
        with open(manifest_filepath) as f:
            afiles = f.readlines()
        files = []
        for afile in afiles:
            asplit = afile.split(' ')
            filecode = asplit[0]
            speaker_id = int(filecode.split('-')[0])
            phones = torch.LongTensor([int(phone) for phone in asplit[1:]])
            audiolen = len(phones)*.01
            if len(phones)<=(chunk*100+1):
                continue
            else:
                files.append([filecode, speaker_id, phones, audiolen])
        if index is None:
            self.files = files
        else:
            self.files = [files[ii] for ii in index]
        self.size = len(self.files)
        self.librispeech_path = librispeech_path
        self.train = True
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
        filecode, speaker_id, phones, audiolen = sample[0], sample[1], sample[2], sample[3]
        # index 3 second window
        if self.chunk is not None:
            offset = np.random.uniform(0, float(audiolen)-self.chunk)
            offset = int(np.floor(offset*100))
            index_phones = np.arange(offset, offset+int(self.chunk*100)+1)
            offset = offset*(self.sample_rate/100)
            index = np.arange(offset, offset+self.sample_rate*self.chunk)
        else:
            index = None
        y = self.parse_audio(get_librispeech_filepath(self.librispeech_path, filecode), index=index)
        target = (speaker_id, phones[index_phones])
        return y, target

    def __len__(self):
        return self.size

class ModFilter(nn.Module):

    def __init__(self, temporal, spectral):
        super(ModFilter, self).__init__()
        self.temporal = temporal
        self.spectral = spectral

    def forward(self, x):
        print(x.shape)
        spec_fft = torch.rfft(x, 2)
        print(spec_fft.shape)
        spec_fft_abs = np.sqrt(spec_fft[:,:,:,:,0]**2+spec_fft[:,:,:,:,1]**2)
        gainmap = torch.zeros(spec_fft.shape[:4])
        if self.temporal is not None:
            gainmap[0,0,:,:self.temporal] = 1
            gainmap[0,0,:,self.temporal] = .5
        if self.spectral is not None:
            gainmap[0,0,:self.spectral,:] = 1
            gainmap[0,0,self.spectral,:] = .5
            gainmap[0,0,-self.spectral:,:] = 1
            gainmap[0,0,-self.spectral,:] = .5
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

class SpecNet(nn.Module):

    def __init__(self, embed_size=256):
        super(SpecNet, self).__init__()
#         self.mel = tac.layers.Melspectrogram(128, 16000, n_fft=2**9)
#         self.lnorm = torch.nn.LayerNorm((1,128,376))
#         self.n_future = 12
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(128,1), stride=1,
                                   bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=(1,3), stride=1,
                                   bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, embed_size, kernel_size=(1,3), stride=1,
                                   bias=False),
            nn.BatchNorm2d(embed_size),
            nn.LeakyReLU()
            )
        self.gru = nn.GRU(input_size=embed_size, hidden_size=embed_size)
        self.lin = nn.Linear(embed_size, 41)

    def forward(self, data):
#         spec = self.lnorm(self.mel(data))
        conv_out = self.convnet(spec)
        gru_in = conv_out[:,:,0,:].transpose(1,2).transpose(0,1)
        gru_out = self.gru(gru_in)[0]
        prediction = self.lin(gru_out).transpose(0,1).transpose(1,2)
        return prediction

train_dataset = LibriSpeechDataset(manifest_filepath, librispeech_path, 3, index=None)
n_files = len(train_dataset)

file_index = np.arange(n_files)
np.random.shuffle(file_index)
test_index = file_index[-640:]
train_index = file_index[:-640]

batch_size = 64
train_dataset = LibriSpeechDataset(manifest_filepath, librispeech_path, 3, index=train_index)
train_sampler = BucketingSampler(train_dataset, batch_size=batch_size)
train_loader = DataLoader(train_dataset, num_workers=1, batch_sampler=train_sampler)
test_dataset = LibriSpeechDataset(manifest_filepath, librispeech_path, 3, index=test_index)
test_sampler = BucketingSampler(test_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, num_workers=1, batch_sampler=test_sampler)

model = SpecNet()

mel = tac.layers.Melspectrogram(128, train_dataset.sample_rate, hop=int(.01*train_dataset.sample_rate), n_fft=2**10)
modfilter = ModFilter(None, 5)
if args.cuda:
    model.cuda()
    mel = mel.cuda()
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

ce_loss = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=lr0)
n_batch = int(len(train_dataset)/batch_size)

train_loss = []
test_loss = []
accuracies = []
for epoch in range(100):
    train_loss.append([])
    model.train()
    avg_loss = 0.
    epoch_start = time()
    train_sampler.shuffle(epoch)
    for i, (data) in enumerate(train_loader, start=0):

        print('Epoch {}, Batch {} of {}'.format(epoch, i, n_batch))
        data = (data[0].to(device), (data[1][0].to(device), data[1][1].to(device)))

        y, (speaker_id, phones) = data

        spec = mel(y)
        spec = tac.scaling.amplitude_to_db(spec)

        if False:
            spec = modfilter(spec)

        out = model(spec)
        loss = ce_loss(out, phones[:,2:-2])
        avg_loss += loss.item()

        # compute gradient
        optimizer.zero_grad()
        loss.backward()

        # if False:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)

        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if True:
            optimizer.step()

        if (i>0) and (i%50==0):
            print('Average loss: {}'.format(avg_loss/i))
            print('Elapsed time: {} seconds'.format(time()-epoch_start))
            train_loss[-1].append(avg_loss/i)

    model.eval()
    avg_loss = 0.
    accuracy = 0.
    epoch_start = time()
    print('Testing..')
    for i, (data) in enumerate(test_loader, start=0):

        print(i)
        data = (data[0].to(device), (data[1][0].to(device), data[1][1].to(device)))
        y, (speaker_id, phones) = data

        spec = mel(y)
        spec = tac.scaling.amplitude_to_db(spec)

        if False:
            spec = modfilter(spec)

        out = model(spec)
        loss = ce_loss(out, phones[:,2:-2])
        accuracy += torch.sum(out.argmax(1) == phones[:,2:-2]).item() / (batch_size*(phones.shape[1]-4))
        avg_loss += loss.item()

    test_loss.append(avg_loss/i)
    accuracies.append(accuracy/i)
    print('Test loss: {}'.format(avg_loss/i))
    print('Test accuracy: {}'.format(accuracy/i))

    if (epoch==0) or (avg_loss/i < best_val_loss):
        best_val_loss = avg_loss/i
        with open('./spec_net_1.pkl', 'wb') as f:
            torch.save(model.state_dict(), f)
