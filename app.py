import eel
import time
import ctypes
import yaml
import os
import math
import sys
import random
from sys import exit
import subprocess
import shutil
import cairo

import cv2

import base64
from datetime import datetime, timezone

import h5py

import torch
from torch.cuda.amp import autocast, GradScaler
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel
import torch.optim as optim

from sklearn.metrics import classification_report

from decord import VideoReader
from decord import cpu, gpu

from cmap import Colormap

import queue

import numpy as np
import pandas as pd

import ctypes

import threading

from classifier_head import classifier

def remove_leading_zeros(num):
    for i in range(0,len(num)):
        if num[i]!='0':
            return int(num[i:])
    return 0

class Actogram:

    def __init__(self, directory, model, behavior, framerate, start, binsize, color, threshold, norm, lightcycle, width=500, height=500):

        self.directory = directory
        self.model = model
        self.behavior = behavior
        self.framerate = framerate
        self.start = start
        self.color = color 
        self.threshold = threshold
        self.norm = norm
        self.lightcycle = lightcycle

        self.width = width
        self.height = height

        self.binsize = binsize

        self.timeseries()

        self.draw()

    def draw(self):

        self.cycles = []
        for c in self.lightcycle:
            if c=='1':
                self.cycles.append(True)
            else:
                self.cycles.append(False)

        clocklab_time = "{:02d}".format(int(self.start)) + ":" + "{:02d}".format(int(60*(self.start - int(self.start))))

        clocklab_file = [self.behavior, '01-jan-2024', clocklab_time, self.binsize/self.framerate/60*4, 0, 0, 0]

        bins = []

        for b in range(0, len(self.totalts), self.binsize):
            bins.append((sum(np.array(self.totalts[b:b+self.binsize]) >= self.threshold), b/self.framerate/3600))
            clocklab_file.append(sum(np.array(self.totalts[b:b+self.binsize]) >= self.threshold))

        df = pd.DataFrame(data=np.array(clocklab_file))

        self.clfile = os.path.join(self.directory, self.model+'-'+self.behavior+'-'+'clocklab.csv')

        df.to_csv(self.clfile, header=False, index=False)

        awdfile = self.clfile.replace('.csv', '.awd')

        if os.path.exists(awdfile):
            os.remove(awdfile)

        os.rename(self.clfile, self.clfile.replace('.csv', '.awd'))

        self.timeseries_data = bins

        if len(bins)<2:
            print('Not enough videos to make an actogram.')
            return

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width, self.height)
        ctx = cairo.Context(surface)

        width = self.width
        height = self.height

        ctx.scale(width, height)

        self.align_draw(ctx)

        self.file = os.path.join(self.directory, self.model+'-'+self.behavior+'-'+'actogram.png')

        surface.write_to_png(self.file)

        frame = cv2.imread(self.file)

        ret, frame = cv2.imencode('.jpg', frame)

        frame = frame.tobytes()

        blob = base64.b64encode(frame)
        blob = blob.decode("utf-8")

        eel.updateActogram(blob)


    def align_draw(self, ctx):

        padding = .01

        actogram_width = (1 - 2*padding)
        actogram_height = (1 - 2*padding)

        cx = padding
        cy = padding

        self.draw_actogram(ctx, cx, cy, actogram_width, actogram_height, padding)

    def draw_actogram(self, ctx, tlx, tly, width, height, padding):

        ctx.set_line_width(.005)

        ctx.rectangle(tlx - padding/4,tly - padding/4,width + padding/2,height + padding/2)
        ctx.set_source_rgba(.1, .1, .1, 1)
        ctx.fill()

        tsdata = self.timeseries_data

        total_days = math.ceil((tsdata[-1][1]+self.start)/24)

        if total_days%2==0:
            total_days-=1

        day_height = height/total_days

        bin_width = 1/48 * self.binsize/36000

        ts = np.array([a[0] for a in tsdata])
        times = np.array([a[1]+self.start for a in tsdata])


        for d in range(total_days):

            by = tly+(d+1)*day_height

            d1 = d

            valid = np.logical_and(times>(d1*24),times<=((d1+2)*24))

            if d1%2!=0:
                adj_times = times[valid] + 24
            else:
                adj_times = times[valid]

            adj_times = adj_times%48

            series = ts[valid]

            if len(series)==0:
                continue

            # normalize the series
            series = np.array(series)
            series = series/self.norm

            series = series*.90

            if self.cycles is not None and d<len(self.cycles):
                LD = self.cycles[d]
            else:
                LD = True

            if LD:
                ctx.set_source_rgb(223/255,223/255,223/255)
                ctx.rectangle(tlx+0/48*width, by-day_height, 6/48*width, day_height)
                ctx.fill()
                ctx.rectangle(tlx+18/48*width, by-day_height, 12/48*width, day_height)
                ctx.fill()
                ctx.rectangle(tlx+42/48*width, by-day_height, 6/48*width, day_height)
                ctx.fill()

                ctx.set_source_rgb(255/255, 239/255, 191/255)
                ctx.rectangle(tlx+6/48*width, by-day_height, 12/48*width, day_height)
                ctx.fill()
                ctx.rectangle(tlx+30/48*width, by-day_height, 12/48*width, day_height)
                ctx.fill()

            else:
                ctx.set_source_rgb(223/255,223/255,223/255)
                ctx.rectangle(tlx+0/48*width, by-day_height, 6/48*width, day_height)
                ctx.fill()
                ctx.rectangle(tlx+18/48*width, by-day_height, 12/48*width, day_height)
                ctx.fill()
                ctx.rectangle(tlx+42/48*width, by-day_height, 6/48*width, day_height)
                ctx.fill()

                ctx.set_source_rgb(246/255, 246/255, 246/255)
                ctx.rectangle(tlx+6/48*width, by-day_height, 12/48*width, day_height)
                ctx.fill()
                ctx.rectangle(tlx+30/48*width, by-day_height, 12/48*width, day_height)
                ctx.fill()



            for t in range(len(adj_times)):
                timepoint = adj_times[t]
                value = series[t]

                a_time = timepoint/48

                ctx.rectangle(tlx+a_time*width,by-value*day_height,bin_width*width,value*day_height)
                ctx.set_source_rgba(self.color[0]/255, self.color[1]/255, self.color[2]/255, 1)
                ctx.fill()

        for d in range(total_days):

            by = tly+(d+1)*day_height

            ctx.set_line_width(.002)
            ctx.set_source_rgb(0, 0, 0)
            ctx.move_to(tlx, by)
            ctx.line_to(tlx+48/48*width, by)
            ctx.stroke()



    def timeseries(self):

        behavior = self.behavior

        valid_files = [(os.path.join(self.directory, file), remove_leading_zeros(file.split('_')[1]))  for file in os.listdir(self.directory) if file.endswith('.csv') and '_'+self.model+'_' in file]

        valid_files.sort(key = lambda vf: vf[1])

        last_num = valid_files[-1][1]

        if len(valid_files)!=last_num+1:

            prev_num = -1
            for vf, num in valid_files:
                if num!=prev_num+1:
                    raise Exception(f'Missing number - {prev_num+1}')

        self.totalts = []

        col_index = -1

        continuous = False

        for vf, num in valid_files:

            dataframe = pd.read_csv(vf)

            if col_index==-1:
                behaviors = dataframe.columns.to_list()[1:]

                col_index = behaviors.index(behavior)


            top = np.argmax(dataframe[behaviors].to_numpy(), axis=1) == col_index
            values = np.max(dataframe[behaviors].to_numpy(), axis=1)

            values = top * values

            if continuous:
                frames = dataframe[behavior].to_list()
            else:
                frames = values

            self.totalts.extend(frames)





class Supervised_Set(Dataset):
    def __init__(self, training_sets, set_type="train", split=.2, seed=42, behaviors=None, seq_len=15):

        self.paths = training_sets

        self.dcls = []
        self.lbls = []

        if seq_len % 2 == 0:
            seq_len += 1

        self.seq_len = seq_len
        self.hsl = seq_len//2

        self.behaviors = []

        self.training_sequences = []
        self.training_labels = []

        self.testing_sequences = []
        self.testing_labels = []

        self.internal_counter = 0

        for training_set in training_sets:

            with open(training_set, 'r') as file:
                self.config = yaml.safe_load(file)

            if behaviors is None:

                behaviors = self.config['behaviors']

                for b in behaviors:
                    if b not in self.behaviors:
                        self.behaviors.append(b)

            else:
                for b in behaviors:
                    if b not in self.behaviors:
                        self.behaviors.append(b)

            instances = self.config['labels']

            training_insts = []
            testing_insts = []

            # generate the same train/test split every time
            random.seed(seed)

            groups = {}

            total_insts = {b: 0 for b in behaviors}

            for b in behaviors:
                for i in instances[b]:

                    vid_name = os.path.split(i['video'])[1]
                    group = vid_name.split('_')[1]

                    if group not in groups:
                        groups[group] = {b:[] for b in behaviors}

                    groups[group][b].append(i)
                    total_insts[b] += 1


            keys = list(groups.keys())
            random.shuffle(keys)

            binsts = {b:[] for b in behaviors}

            for key in keys:
                insts = groups[key]
                for b in behaviors:
                    binsts[b].extend(insts[b])

            for b in behaviors:
                splt = int((1-split)*len(binsts[b]))
                training_insts.extend(binsts[b][:splt])
                testing_insts.extend(binsts[b][splt:])

            random.shuffle(training_insts)
            random.shuffle(testing_insts)

            seqs = []
            labels = []

            ltrain = len(training_insts)
            ltest = len(testing_insts)

            if set_type == 'train':

                for i in range(ltrain):

                    print(f'Generating train instance: {i}/{ltrain}')

                    inst = training_insts[i]

                    start = int(inst['start'])
                    end = int(inst['end'])
                    video_path = inst['video']

                    cls_path = video_path.replace('.mp4', '_cls.h5')


                    with h5py.File(cls_path, 'r') as file:
                        cls = file['cls'][:]

                        video_mean = np.mean(cls)

                        if cls.shape[0]-self.hsl < start or self.hsl > end:
                            continue

                        start = max(self.hsl+1, start)-1
                        end = min(cls.shape[0]-(self.hsl+1), end)-1

                    inds = list(range(start, end))

                    for t in inds:

                        ind = t

                        with h5py.File(cls_path, 'r') as file:
                            clss = file['cls'][ind-self.hsl:ind+(self.hsl+1)]

                        clss = torch.from_numpy(clss - video_mean).half()


                        if clss.shape[0] != self.seq_len:
                            continue

                        seqs.append(clss)

                        label = self.behaviors.index(inst['label'])

                        labels.append(torch.tensor(label).long())



                all = list(zip(seqs, labels))

                random.shuffle(all)

                seqs, labels = zip(*all)

                self.training_sequences.extend(seqs)
                self.training_labels.extend(labels)

            elif set_type == 'test':


                for i in range(ltest):

                    print(f'Generating test instance: {i}/{ltest}')

                    inst = testing_insts[i]

                    start = int(inst['start'])
                    end = int(inst['end'])
                    video_path = inst['video']

                    cls_path = video_path.replace('.mp4', '_cls.h5')

                    video_mean = None


                    with h5py.File(cls_path, 'r') as file:
                        cls = file['cls'][:]

                        video_mean = np.mean(cls)

                        if cls.shape[0]-self.hsl < start or self.hsl > end:
                            continue

                        start = max(self.hsl+1, start)-1
                        end = min(cls.shape[0]-(self.hsl+1), end)-1

                    inds = list(range(start, end))

                    for t in inds:

                        ind = t

                        with h5py.File(cls_path, 'r') as file:
                            clss = file['cls'][ind-self.hsl:ind+(self.hsl+1)]

                        clss = torch.from_numpy(clss - video_mean).half()

                        if clss.shape[0] != self.seq_len:
                            continue

                        seqs.append(clss)

                        label = self.behaviors.index(inst['label'])

                        labels.append(torch.tensor(label).long())

                all = list(zip(seqs, labels))

                random.shuffle(all)

                seqs, labels = map(list, zip(*all))

                self.testing_sequences.extend(seqs)
                self.testing_labels.extend(labels)

        if set_type == 'train':

            all = list(zip(self.training_sequences, self.training_labels))
            random.shuffle(all)
            self.training_sequences, self.training_labels = map(list, zip(*all))

            for i in range(len(self.training_sequences)):

                self.dcls.append(self.training_sequences[i])

            self.lbls.extend(self.training_labels)

        elif set_type == 'test':

            all = list(zip(self.testing_sequences, self.testing_labels))
            random.shuffle(all)
            self.testing_sequences, self.testing_labels = map(list, zip(*all))

            for i in range(len(self.testing_sequences)):

                self.dcls.append(self.testing_sequences[i])

            self.lbls.extend(self.testing_labels)

        self.organized_sequences = {b: [] for b in self.behaviors}

        for l, d in zip(self.lbls, self.dcls):
            self.organized_sequences[self.behaviors[l.item()]].append(d)

    def __len__(self):
        return len(self.dcls) + (len(self.behaviors) - len(self.dcls)%len(self.behaviors))

    def __getitem__(self, idx):

        b = self.internal_counter % len(self.behaviors)
        self.internal_counter += 1

        if self.internal_counter % len(self.behaviors) == 0:
            self.internal_counter = 0

        dcls = self.organized_sequences[self.behaviors[b]][idx%len(self.organized_sequences[self.behaviors[b]])]
        lbl = torch.tensor(b).long()

        return dcls, lbl

def collate_fn(batch):

    dcls = [item[0] for item in batch]
    lbls = [item[1] for item in batch]

    dcls = torch.stack(dcls)
    lbls = torch.stack(lbls)

    return dcls, lbls

class encoder(nn.Module):

    def __init__(self, device='cuda'):
        super(encoder, self).__init__()

        self.device = device

        self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):

        B, S, H, W = x.shape

        x = x.to(self.device)

        x = x.unsqueeze(2).repeat(1, 1, 3, 1, 1)

        x = x.reshape(B*S, 3, H, W)

        with torch.no_grad():
            out = self.model(x)

        cls = out.last_hidden_state[:, 0, :].reshape(B, S, 768)

        return cls


active_streams = {}

recordings = ''

stop_threads = False

progresses = []

label_dict_path = None
label_dict = None
col_map = None

label_capture = None
label_videos = []
label_vid_index = -1
label_index = -1
label = -1
start = -1

task_queue = queue.Queue()

instance_stack = None

gpu_lock = threading.Lock()

classification_threads = []

actogram = None

class inference_thread(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name

    def run(self):
        global stop_threads
        global progresses
        global task_queue
        global gpu_lock

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enc = encoder(self.device).to(self.device)

        while True:
            progresses = []
            if recordings!='':
                videos = []
                sub_dirs = [os.path.join(recordings, d) for d in os.listdir(recordings) if os.path.isdir(os.path.join(recordings, d))]
                for sd in sub_dirs:

                    sub_sub_dirs = [os.path.join(sd, d) for d in os.listdir(sd) if os.path.isdir(os.path.join(sd, d))]
                    for ssd in sub_sub_dirs:
                        videos.extend([os.path.join(ssd, vid) for vid in os.listdir(ssd) if vid.endswith('.mp4') and not os.path.exists(os.path.join(ssd, vid.replace('.mp4', '_cls.h5')))])

                valid_videos = []
                for v in videos:
                    cap = cv2.VideoCapture(v)

                    if not cap.isOpened():
                        continue

                    valid_videos.append(v)

                videos = valid_videos

                if len(videos)==0:

                    time.sleep(1)
                    continue

                progresses = [0 for v in videos]

                m = 100/len(progresses)

                for iv, v in enumerate(videos):
                    with gpu_lock:

                        try:
                            cap = cv2.VideoCapture(v)

                            if not cap.isOpened():
                                progresses[iv] = -1

                                time.sleep(5)
                                continue

                            file_path = os.path.splitext(v)[0]+'_cls.h5'
                            print(f"encoding video '{v}' -> '{file_path}'")

                            vr = VideoReader(v, ctx=cpu(0))

                            frames = vr.get_batch(range(0, len(vr), 1)).asnumpy()

                            frames = torch.from_numpy(frames[:, :, :, 1]/255).half()

                            batch_size = 256

                            with torch.no_grad():
                                clss = []

                                for i in range(0, len(frames), batch_size):
                                    batch = frames[i:i+batch_size]

                                    progresses[iv] = (i)/len(frames)*m

                                    with torch.no_grad() and autocast():
                                        out = self.enc(batch.unsqueeze(1).to(self.device))

                                    batch = batch.to('cpu')

                                    out = out.squeeze(1).to('cpu')
                                    clss.extend(out)


                            with h5py.File(file_path, 'w') as file:
                                file.create_dataset('cls', data=torch.stack(clss).numpy())
                            
                            clss = None
                            batch = None
                            frames = None
                            vr = None
                            out = None

                            torch.cuda.empty_cache()

                            progresses[iv] = m

                        except Exception as e:
                            progresses[iv] = -1
                            print('Error processing video:', v)
                            continue

                    time.sleep(1)

                time.sleep(1)
            else:
                time.sleep(1)

    def get_id(self):

        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def raise_exception(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
              ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print('Exception raise failure')

class training_thread(threading.Thread):
    def __init__(self, name, config, dataset, batch_size, learning_rate, epochs, sequence_length):
        threading.Thread.__init__(self)
        self.name = name

        self.config = config
        self.dataset = dataset

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.sequence_length = sequence_length

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def run(self):
        global gpu_lock

        with gpu_lock:

            datasets, dataset = os.path.split(os.path.split(self.config)[0])
            models = os.path.join(os.path.split(datasets)[0], 'models')

            model_dir = os.path.join(models, dataset)
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)

            model_path = os.path.join(model_dir, 'model.pth')
            performance_path = os.path.join(model_dir, 'performance.yaml')

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            batch_size=self.batch_size
            lr=self.learning_rate

            train_set = Supervised_Set([self.dataset], split=.25, set_type='train', seed=42, seq_len=self.sequence_length)
            test_set = Supervised_Set([self.dataset], split=.25, set_type='test', seed=42, seq_len=self.sequence_length)

            behaviors = train_set.behaviors

            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

            criterion = nn.CrossEntropyLoss()

            best_f1 = 0
            best_report = None

            epochs = self.epochs

            for trials in range(10):

                model = classifier(in_features=768, out_features=len(behaviors), seq_len=self.sequence_length).to(device)

                optimizer = optim.Adam(model.parameters(), lr=lr)

                for e in range(epochs):

                    for t, param_group in enumerate(optimizer.param_groups):
                        param_group["lr"] = (epochs-e)/epochs * 0.0005 + (e/epochs) * 0.00001

                    for i, (d, l) in enumerate(train_loader):

                        d = d.to(device).float()
                        l = l.to(device)

                        optimizer.zero_grad()

                        B = d.shape[0]

                        lstm_logits, linear_logits, rawm = model(d)

                        logits = lstm_logits + linear_logits

                        inv_loss = criterion(logits, l)

                        rawm = (rawm - rawm.mean(dim=0))

                        covm = (rawm @ rawm.T)/rawm.shape[0]
                        covm_loss = torch.sum(torch.pow(self.off_diagonal(covm), 2))/rawm.shape[1]

                        loss = inv_loss + covm_loss

                        loss.backward()
                        optimizer.step()

                        #print(f'F1: {best_f1} Epoch: {e} Batch: {i} Total Loss: {loss.item()}')

                    actuals = []
                    predictions = []

                    for i, (d, l) in enumerate(test_loader):

                        d = d.to(device).float()
                        l = l.to(device)

                        with torch.no_grad():

                            lstm_logits, linear_logits = model.forward_nodrop(d)

                            logits = lstm_logits + linear_logits

                        actuals.extend(l.cpu().numpy())
                        predictions.extend(logits.argmax(1).cpu().numpy())


                    report_dict = classification_report(actuals, predictions, target_names=behaviors, output_dict=True)

                    wf1score = report_dict['weighted avg']['f1-score']

                    if best_f1<wf1score:
                        best_f1 = wf1score
                        best_report = report_dict

                        torch.save(model, model_path)


            if best_report:
                with open(performance_path, 'w+') as file:
                    yaml.dump(best_report, file, allow_unicode=True)

                for b in behaviors:
                    #config_path, behavior, group, value
                    update_metrics(self.config, b, 'Precision', round(best_report[b]['precision'], 2))
                    update_metrics(self.config, b, 'Recall', round(best_report[b]['recall'], 2))
                    update_metrics(self.config, b, 'F1 Score', round(best_report[b]['f1-score'], 2))

                with open(self.config, 'r+') as file:
                    config = yaml.safe_load(file)

                config['model'] = model_path

                with open(self.config, 'w+') as file:
                    yaml.dump(config, file, allow_unicode=True)

                config_path = os.path.join(os.path.split(model_path)[0], 'config.yaml')

                config = {
                    'seq_len': self.sequence_length,
                    'behaviors': behaviors
                }

                with open(config_path, 'w+') as file:
                    yaml.dump(config, file, allow_unicode=True)



    def get_id(self):

        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def raise_exception(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
              ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print('Exception raise failure')

class classification_thread(threading.Thread):
    def __init__(self, model_path, whitelist):
        threading.Thread.__init__(self)

        self.model_path = model_path
        self.whitelist = whitelist

    def run(self):
        global gpu_lock

        while True:

            time.sleep(1)

            with gpu_lock:

                dataset_name = os.path.split(os.path.split(self.model_path)[0])[1]

                config_path = os.path.join(os.path.split(self.model_path)[0], 'config.yaml')

                with open(config_path, 'r+') as file:
                    config = yaml.safe_load(file)

                seq_len = config['seq_len']
                behaviors = config['behaviors']

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                model = torch.load(self.model_path)

                try:
                    model.eval()
                except:
                    state_dict = model
                    model = classifier(in_features=768, out_features=len(behaviors), seq_len=seq_len)
                    model.load_state_dict(state_dict=state_dict)
                    model.eval()

                model.to(device)

                all_videos = []

                if recordings=='':
                    continue
                else:
                    sub_dirs = [os.path.join(recordings, d) for d in os.listdir(recordings) if os.path.isdir(os.path.join(recordings, d))]


                    if len(sub_dirs)==0:
                        continue
                    else:

                        for sd in sub_dirs:

                            sub_sub_dirs = [os.path.join(sd, d) for d in os.listdir(sd) if os.path.isdir(os.path.join(sd, d))]

                            for ssd in sub_sub_dirs:

                                all_videos.extend([os.path.join(ssd, v) for v in os.listdir(ssd) if v.endswith('_cls.h5')])

                valid_videos = []
                for v in all_videos:
                    for wl in self.whitelist:
                        if wl in v:
                            valid_videos.append(v)



                for clsfile in valid_videos:
                    outputfile = clsfile.replace('_cls.h5', '_' + dataset_name + '_outputs.csv')

                    if os.path.isfile(outputfile):
                        continue

                    print(f"classifying '{clsfile}' -> '{outputfile}'")

                    with h5py.File(clsfile, 'r') as file:
                        cls = np.array(file['cls'][:])

                    cls = torch.from_numpy(cls - np.mean(cls, axis=0)).half()


                    predictions = []

                    batch = []

                    if len(cls)<seq_len:
                        continue

                    for ind in range(seq_len//2, len(cls)-seq_len//2):


                        batch.append(cls[ind-seq_len//2:ind+seq_len//2+1])

                        if len(batch)>=4096 or ind==len(cls)-seq_len//2-1:

                            batch = torch.stack(batch)

                            with torch.no_grad() and autocast():

                                logits = model.forward_nodrop(batch.to(device))

                                probs = torch.softmax(logits, dim=1)

                                predictions.extend(probs.detach().cpu().numpy())

                            batch = []

                    total_predictions = []

                    for ind in range(len(cls)):

                        if ind<seq_len//2:
                            total_predictions.append(predictions[0])
                        elif ind>=len(cls)-seq_len//2:
                            total_predictions.append(predictions[-1])
                        else:
                            total_predictions.append(predictions[ind-seq_len//2])

                    total_predictions = np.array(total_predictions)

                    dataframe = pd.DataFrame(total_predictions, columns=behaviors)

                    dataframe.to_csv(outputfile)

                    torch.cuda.empty_cache()




    def get_id(self):

        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def raise_exception(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
              ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print('Exception raise failure')

thread = inference_thread('inference')

tthread = None

def tab20_map(val):

    if val<10:
        return val*2
    else:
        return (val - 10)*2 + 1

def add_instance():
    global label_videos
    global label_vid_index
    global label_capture
    global label
    global label_index
    global start

    global label_dict
    global col_map

    global instance_stack

    stemp = min(start, label_index)
    eInd = max(start, label_index)
    sInd = stemp

    # check for collisions
    labels = label_dict['labels']
    behaviors = label_dict['behaviors']

    for i, b in enumerate(behaviors):
        sub_labels = labels[b]

        for l in sub_labels:
            if l['video']==label_videos[label_vid_index]:
                if sInd < l['end'] and sInd > l['start']:
                    label = -1
                    raise Exception('Overlapping behavior region! Behavior not recorded.')
                elif eInd < l['end'] and eInd > l['start']:
                    label = -1
                    raise Exception('Overlapping behavior region! Behavior not recorded.')

    behavior = label_dict['behaviors'][label]

    instance = {
        'video': label_videos[label_vid_index],
        'start': sInd,
        'end': eInd,
        'label': behavior
    }


    label_dict['labels'][behavior].append(instance)

    instance_stack.append(instance)

    # save the label dictionary
    with open(label_dict_path, 'w+') as file:
        yaml.dump(label_dict, file, allow_unicode=True)

    update_counts()



def fill_colors(frame):
    global label_videos
    global label_vid_index
    global label_capture
    global label
    global label_index
    global start

    global label_dict
    global col_map

    behaviors = label_dict['behaviors']

    labels = label_dict['labels']

    cur_video = label_videos[label_vid_index]
    amount_of_frames = label_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    for i, b in enumerate(behaviors):

        sub_labels = labels[b]

        color = str(col_map(tab20_map(i))).lstrip('#')
        color = np.flip(np.array([int(color[i:i+2], 16) for i in (0, 2, 4)]))

        for l in sub_labels:
            if l['video'] != cur_video:
                continue
            sInd = l['start']
            eInd = l['end']

            marker_posS = int(frame.shape[1] * sInd/amount_of_frames)
            marker_posE = int(frame.shape[1] * eInd/amount_of_frames)

            frame[-49:, marker_posS:marker_posE+1,] = color

    if label!=-1:
        color = str(col_map(tab20_map(label))).lstrip('#')
        color = np.flip(np.array([int(color[i:i+2], 16) for i in (0, 2, 4)]))

        stemp = min(start, label_index)
        eInd = max(start, label_index)

        sInd = stemp

        marker_posS = int(frame.shape[1] * sInd/amount_of_frames)
        marker_posE = int(frame.shape[1] * eInd/amount_of_frames)

        frame[-49:, marker_posS:marker_posE+1,] = color

    return frame

eel.init('frontend')
eel.browsers.set_path('electron', 'node_modules/electron/dist/electron')


@eel.expose
def get_progress_update():
    if len(progresses)==0:
        eel.inferLoadBar(False)()
    else:
        eel.inferLoadBar(progresses)()

@eel.expose
def make_recording_dir(root, sub_dir, camera_name):

    if sub_dir=='':
        sub_dir = datetime.now().strftime('%Y%m%d')

    if not os.path.exists(os.path.join(root, sub_dir)):
        os.mkdir(os.path.join(root, sub_dir))

    cam_session = camera_name + '-' + datetime.now().strftime('%I%M%S-%p')

    if not os.path.exists(os.path.join(root, sub_dir, cam_session)):
        os.mkdir(os.path.join(root, sub_dir, cam_session))

        return os.path.join(root, sub_dir, cam_session)

    return False

@eel.expose
def project_exists(project_directory):
    global recordings

    project = project_directory

    cameras = os.path.join(project, 'cameras')
    recordings = os.path.join(project, 'recordings')
    models = os.path.join(project, 'models')
    data_sets = os.path.join(project, 'data_sets')

    if os.path.exists(project):
        if os.path.exists(cameras) and os.path.exists(recordings) and os.path.exists(models) and os.path.exists(data_sets):
            return True, {'project':project, 'cameras':cameras, 'recordings':recordings, 'models':models, 'data_sets':data_sets}
        else:
            return False, None
    else:
        return False, None

@eel.expose
def create_project(parent_directory, project_name):

    global recordings

    # main project directory
    project = os.path.join(parent_directory, project_name)

    # make the names of the directories
    cameras = os.path.join(project, 'cameras')
    recordings = os.path.join(project, 'recordings')
    models = os.path.join(project, 'models')
    data_sets = os.path.join(project, 'data_sets')

    # check to see if the project already exists
    if os.path.exists(project):
        print('Project already exists. Please choose a different name or location.')
        return False, None
    else:
        print('Creating project...')

    # make all those directories
    os.mkdir(project)
    os.mkdir(cameras)
    os.mkdir(recordings)
    os.mkdir(models)
    os.mkdir(data_sets)

    print(f'Project creation successful!')

    return True, {'project':project, 'cameras':cameras, 'recordings':recordings, 'models':models, 'data_sets':data_sets}

@eel.expose
def make_actogram(root, sub_dir, model, behavior, framerate, binsize, start, color, threshold, norm, lightcycle):

    global actogram

    framerate = int(framerate)
    binsize = int(binsize)*framerate*60
    start = float(start)

    threshold = float(threshold)/100
    color = str(color.lstrip('#'))
    color = np.array([int(color[i:i+2], 16) for i in (0, 2, 4)])

    directory = os.path.join(recordings, root, sub_dir)

    actogram = Actogram(directory, model, behavior, framerate, start, binsize, color, threshold, int(norm), lightcycle, width=500, height=500)

@eel.expose
def adjust_actogram(framerate, binsize, start, color, threshold, norm, lightcycle):

    framerate = int(framerate)
    binsize = int(binsize)*framerate*60
    start = float(start)
    
    threshold = float(threshold)/100
    color = str(color.lstrip('#'))
    color = np.array([int(color[i:i+2], 16) for i in (0, 2, 4)])

    global actogram

    if actogram is not None:
        actogram.framerate = framerate
        actogram.binsize = binsize
        actogram.start = start
        actogram.color = color 
        actogram.threshold = threshold
        actogram.norm = int(norm)
        actogram.lightcycle = lightcycle

        actogram.draw()



@eel.expose
def recording_structure():

    global recordings

    if recordings=='':
        return None
    else:
        structure = {}

        for d in os.listdir(recordings):
            dpath = os.path.join(recordings, d)
            if os.path.isdir(dpath):
                structure[d] = {}

                for subd in os.listdir(dpath):
                    sdpath = os.path.join(dpath, subd)
                    if os.path.isdir(sdpath):
                        structure[d][subd] = {}

                        for f in os.listdir(sdpath):
                            fpath = os.path.join(sdpath, f)
                            if f.endswith('.csv'):
                                try:
                                    model = f.split('_')[-2]

                                    if model in structure[d][subd].keys():
                                        continue

                                    dataframe = pd.read_csv(fpath)

                                    behaviors = dataframe.columns.to_list()

                                    structure[d][subd][model] = behaviors[1:]
                                except:
                                    continue

                            else:
                                continue

                    else:
                        continue

            else:
                continue

        return structure

@eel.expose
def datasets(dataset_directory):
    dsets = {}

    for dataset in os.listdir(dataset_directory):
        if os.path.isdir(os.path.join(dataset_directory, dataset)):

            dataset_config = os.path.join(dataset_directory, dataset, 'config.yaml')

            with open(dataset_config, 'r') as file:
                dconfig = yaml.safe_load(file)

            dsets[dataset] = dconfig


    if len(dsets)>0:
        return dsets
    else:
        return False

@eel.expose
def create_dataset(dataset_directory, name, behaviors, recordings):

    global label_dict_path
    global label_dict

    rt = get_record_tree()

    if not rt:
        return False

    whitelist = []

    for r in recordings:
        whitelist.append(r+'\\')

    directory = os.path.join(dataset_directory, name)

    if os.path.exists(directory):
        return False
    else:
        os.mkdir(directory)

        dataset_config = os.path.join(directory, 'config.yaml')

        label_file = os.path.join(directory, 'labels.yaml')

        metrics = {b:{'Train #':0, 'Test #':0, 'Precision':'N/A', 'Recall':'N/A', 'F1 Score':'N/A'} for b in behaviors}

        dconfig = {
            'name':name,
            'behaviors':behaviors,
            'whitelist':whitelist,
            'model':None,
            'metrics':metrics
        }

        labelconfig = {
            'behaviors':behaviors,
            'labels':{b:[] for b in behaviors}
        }

        with open(dataset_config, 'w+') as file:
            yaml.dump(dconfig, file, allow_unicode=True)

        with open(label_file, 'w+') as file:
            yaml.dump(labelconfig, file, allow_unicode=True)

        label_dict_path = label_file
        label_dict = labelconfig

        return True

@eel.expose
def train_model(dataset_directory, name, batch_size, learning_rate, epochs, sequence_length):
    #name, config, dataset, batch_size, learning_rate, epochs, sequence_length

    global tthread

    config = os.path.join(dataset_directory, name, 'config.yaml')
    dataset = os.path.join(dataset_directory, name, 'labels.yaml')

    tthread = training_thread(name, config, dataset, int(batch_size), float(learning_rate), int(epochs), int(sequence_length))

    tthread.start()

@eel.expose
def start_classification(datasets, dataset, whitelist):

    global classification_threads

    config_path = os.path.join(datasets, dataset, 'config.yaml')

    with open(config_path, 'r+') as file:
        config = yaml.safe_load(file)

    model_path = config['model']
    if model_path:
        if os.path.exists(model_path):
            cthread = classification_thread(model_path, whitelist)
            cthread.start()

            classification_threads.append(cthread)


@eel.expose
def ping_cameras(camera_directory):
    names = []
    cameras = {}

    for camera in os.listdir(camera_directory):
        if os.path.isdir(os.path.join(camera_directory, camera)):
            if camera in active_streams.keys():
                continue

            config = os.path.join(camera_directory, camera, 'config.yaml')

            with open(config, 'r') as file:
                cconfig = yaml.safe_load(file)

            names.append(cconfig['name'])

            print(f'Loading camera: {cconfig["name"]} at {cconfig["rtsp_url"]}...')

            rtsp_url = cconfig['rtsp_url']

            frame_location = os.path.join(camera_directory, camera, 'frame.jpg')

            if os.path.exists(frame_location):
                os.remove(frame_location)

            command = f"ffmpeg -loglevel panic -rtsp_transport tcp -i {rtsp_url} -vf \"select=eq(n\,34)\" -vframes 1 -y \"{frame_location}\""

            subprocess.Popen(command, shell=True)

            print(f'Finished loading camera: {cconfig["name"]} at {cconfig["rtsp_url"]}...')

            cameras[camera] = frame_location

    return names, cameras

@eel.expose
def update_camera_frames(camera_directory):

    for camera in os.listdir(camera_directory):
        if os.path.isdir(os.path.join(camera_directory, camera)):

            frame_location = os.path.join(camera_directory, camera, 'frame.jpg')
            if os.path.exists(frame_location):
                frame = cv2.imread(frame_location)

                ret, frame = cv2.imencode('.jpg', frame)

                frame = frame.tobytes()

                blob = base64.b64encode(frame)
                blob = blob.decode("utf-8")

                eel.updateImageSrc(camera, blob)()

@eel.expose
def camera_names(camera_directory):

    names = []

    for camera in os.listdir(camera_directory):
        if os.path.isdir(os.path.join(camera_directory, camera)):

            names.append(camera)

    return names

@eel.expose
def create_camera(camera_directory, name, rtsp_url, framerate=10, resolution=256, crop_left_x=0, crop_top_y=0, crop_width=1, crop_height=1):

    # set up a folder for the camera
    camera = os.path.join(camera_directory, name)

    print('Creating camera...')
    os.mkdir(camera)

    # set up the camera config
    camera_config = {
        'name':name,
        'rtsp_url':rtsp_url,
        'framerate':framerate,
        'resolution':resolution,
        'crop_left_x':crop_left_x,
        'crop_top_y':crop_top_y,
        'crop_width':crop_width,
        'crop_height':crop_height
    }

    # save the camera config
    with open(os.path.join(camera, 'config.yaml'), 'w+') as file:
        yaml.dump(camera_config, file, allow_unicode=True)

    return True, name, camera_config

@eel.expose
def update_camera(camera_directory, name, rtsp_url, framerate=10, resolution=256, crop_left_x=0, crop_top_y=0, crop_width=1, crop_height=1):

    # set up a folder for the camera
    camera = os.path.join(camera_directory, name)

    # check to see if the camera already exists
    if os.path.exists(camera):
        # must be an update
        print('Updating camera...')

        # find the camera config
        camera_config = os.path.join(camera, 'config.yaml')

        if os.path.exists(camera_config):
            # load the camera config
            with open(camera_config, 'r') as file:
                cconfig = yaml.safe_load(file)

            cconfig['name'] = name
            cconfig['rtsp_url'] = rtsp_url
            cconfig['framerate'] = framerate
            cconfig['resolution'] = resolution
            cconfig['crop_left_x'] = crop_left_x
            cconfig['crop_top_y'] = crop_top_y
            cconfig['crop_width'] = crop_width
            cconfig['crop_height'] = crop_height

            # save the camera config
            with open(camera_config, 'w+') as file:
                yaml.dump(cconfig, file, allow_unicode=True)
        else:
            # remove the camera directory and start fresh
            shutil.rmtree(camera)
            create_camera(camera_directory, name, rtsp_url, framerate, resolution, crop_left_x, crop_top_y, crop_width, crop_height)

    else:
        # must be a new camera
        create_camera(camera_directory, name, rtsp_url, framerate, resolution, crop_left_x, crop_top_y, crop_width, crop_height)

@eel.expose
def test_camera(camera_directory, name, rtsp_url):

    # set up a folder for the camera
    camera = os.path.join(camera_directory, name)

    # check to see if the camera already exists
    if os.path.exists(camera):

        test_frame = os.path.join(camera_directory, name, 'frame.jpg')

        command = f"ffmpeg -rtsp_transport tcp -i {rtsp_url} -vf \"select=eq(n\,34)\" -vframes 1 -y {test_frame}"
        try:
            subprocess.call(command, shell=True)
            print('RTSP functional!')
        except:
            raise Exception('Either the RTSP url is wrong or ffmpeg is not installed properly. You may need to include a username and password in your RTSP url if you see an authorization error. You may run this function again with safe=False to generate the camera regardless of image acquisition.')

    else:
        raise Exception('The camera does not exist. Please create the camera first before testing it.')

@eel.expose
def get_cam_settings(camera_directory, name):
    # set up a folder for the camera
    camera = os.path.join(camera_directory, name)

    # check to see if the camera already exists
    if os.path.exists(camera):

        # find the camera config
        camera_config = os.path.join(camera, 'config.yaml')

        with open(camera_config, 'r') as file:
            cconfig = yaml.safe_load(file)

        return cconfig

    return False

@eel.expose
def remove_camera(camera_directory, name):

    # set up a folder for the camera
    camera = os.path.join(camera_directory, name)

    # check to see if the camera already exists
    if os.path.exists(camera):
        print('Removing camera...')
        shutil.rmtree(camera)
    else:
        raise Exception('The camera does not exist. Please create the camera first before removing it.')

@eel.expose
def start_camera_stream(camera_directory, name, destination, segment_time, duration=None):

    global active_streams

    if name in active_streams.keys():
        return False

    config = os.path.join(camera_directory, name, 'config.yaml')

    with open(config, 'r') as file:
        cconfig = yaml.safe_load(file)

    rtsp_url = cconfig['rtsp_url']
    framerate = cconfig['framerate']
    scale = cconfig['resolution']

    cw = cconfig['crop_width']
    ch = cconfig['crop_height']

    cx = cconfig['crop_left_x']
    cy = cconfig['crop_top_y']

    if not os.path.exists(destination):
        os.mkdir(destination)

    destination = os.path.join(destination, f'{name}_%05d.mp4')

    command = [
        'ffmpeg', '-loglevel', 'panic', '-rtsp_transport', 'tcp', '-i', str(rtsp_url),
        '-r', str(framerate),
        '-filter_complex', f"[0:v]crop=(iw*{cw}):(ih*{ch}):(iw*{cx}):(ih*{cy}),scale={scale}:{scale}[cropped]",
        '-map', '[cropped]', '-f', 'segment', '-segment_time', str(segment_time),
        '-reset_timestamps', '1',
        '-hls_flags', 'temp_file', '-y', destination
    ]

    process = subprocess.Popen(command, stdin=subprocess.PIPE, shell=True)

    active_streams[name] = process

    return True

@eel.expose
def get_record_tree():
    global recordings

    rt = {}

    if recordings=='':
        return False
    else:
        sub_dirs = [d for d in os.listdir(recordings) if os.path.isdir(os.path.join(recordings, d))]

        if len(sub_dirs)==0:
            return False
        else:
            rt = {sd:[] for sd in sub_dirs}

            for sd in sub_dirs:

                sub_sub_dirs = [d for d in os.listdir(os.path.join(recordings,sd)) if os.path.isdir(os.path.join(recordings,sd, d))]

                rt[sd] = sub_sub_dirs

    return rt

@eel.expose
def pop_instance():

    global instance_stack
    global label_dict

    last_inst = instance_stack[-1]

    beh = None
    ind = None

    for b in label_dict['behaviors']:
        for i,inst in enumerate(label_dict['labels'][b]):
            if inst['video']==last_inst['video'] and inst['start']==last_inst['start'] and inst['end']==last_inst['end']:
                beh = b
                ind = i
    if beh==None or ind==None:
        return
    else:
        del label_dict['labels'][beh][ind]

    # save the label dictionary
    with open(label_dict_path, 'w+') as file:
        yaml.dump(label_dict, file, allow_unicode=True)

    update_counts()

    nextFrame(0)


@eel.expose
def label_frame(value):

    global label_dict

    global label
    global label_index
    global start

    behaviors = label_dict['behaviors']

    if value>=len(behaviors):
        return False

    if value==label:
        add_instance()
        label = -1
    elif label==-1:
        label = value
        start = label_index
    else:
        label = -1
        raise Exception('Label does not match that that was started.')

@eel.expose
def nextFrame(shift):
    global label_capture
    global label_videos
    global label_vid_index
    global label_index
    global label
    global start

    if shift<=0:
        shift-=1

    if label_capture.isOpened():

        amount_of_frames = label_capture.get(cv2.CAP_PROP_FRAME_COUNT)

        label_index+=shift
        label_index%=amount_of_frames



        label_capture.set(cv2.CAP_PROP_POS_FRAMES, label_index)

        ret, frame = label_capture.read()

        if ret:

            frame = cv2.resize(frame, (500, 500))

            temp = np.zeros((frame.shape[0]+50, frame.shape[1], frame.shape[2]))

            temp[:-50,:,:] = frame

            temp[-50,:,:] = 0

            temp[-49:,:,:] = 100

            temp = fill_colors(temp)

            marker_pos = int(frame.shape[1] * label_index/amount_of_frames)

            if marker_pos!=0 and marker_pos!=frame.shape[1]-1:
                temp[-45:-5, marker_pos-1:marker_pos+2, :] = 255
            else:
                if marker_pos==0:
                    temp[-45:-5, marker_pos:marker_pos+2, :] = 255
                else:
                    temp[-45:-5, marker_pos-1:marker_pos+1, :] = 255


            ret, frame = cv2.imencode('.jpg', temp)

            frame = frame.tobytes()

            blob = base64.b64encode(frame)
            blob = blob.decode("utf-8")

            eel.updateLabelImageSrc(blob)()

@eel.expose
def nextVideo(shift):
    global label_capture
    global label_videos
    global label_vid_index
    global label_index
    global label
    global start

    start = -1
    label_vid_index = label_vid_index+shift
    label = -1
    label_index = -1

    video = label_videos[label_vid_index%len(label_videos)]

    label_capture = cv2.VideoCapture(video)

    if not label_capture.isOpened():

        recovered = False
        for i in range(len(label_videos)):
            label_vid_index+=shift
            video = label_videos[label_vid_index%len(label_videos)]
            label_capture = cv2.VideoCapture(video)

            if label_capture.isOpened():
                recovered = True
                break

        if not recovered:
            raise Exception('No valid videos in the dataset.')


    nextFrame(1)

@eel.expose
def start_labeling(root, dataset_name):

    global label_dict_path
    global label_dict

    global col_map

    global recordings
    global label_capture
    global label_videos
    global label_vid_index
    global label_index
    global label
    global start

    global instance_stack

    label_capture = None
    label_videos = []
    label_vid_index = -1
    label_index = -1
    label = -1
    start = -1
    instance_stack = []

    dataset_config = os.path.join(root, dataset_name,'config.yaml')
    label_file = os.path.join(root, dataset_name,'labels.yaml')

    label_dict_path = label_file

    cm = Colormap('seaborn:tab20')

    col_map = cm

    if not os.path.exists(label_dict_path):
        return False

    if not os.path.exists(dataset_config):
        return False

    with open(label_dict_path, 'r') as file:
        label_dict = yaml.safe_load(file)

    with open(dataset_config, 'r') as file:
        dconfig = yaml.safe_load(file)

    whitelist = dconfig['whitelist']

    all_videos = []

    if recordings=='':
        return False
    else:
        sub_dirs = [os.path.join(recordings, d) for d in os.listdir(recordings) if os.path.isdir(os.path.join(recordings, d))]

        if len(sub_dirs)==0:
            return False
        else:

            for sd in sub_dirs:

                sub_sub_dirs = [os.path.join(sd, d) for d in os.listdir(sd) if os.path.isdir(os.path.join(sd, d))]

                for ssd in sub_sub_dirs:

                    all_videos.extend([os.path.join(ssd, v) for v in os.listdir(ssd) if v.endswith('.mp4')])

    valid_videos = []
    for v in all_videos:
        for wl in whitelist:
            if wl in v:
                valid_videos.append(v)

    if len(valid_videos)==0:
        return False
    else:
        label_videos = valid_videos

    nextVideo(1)


    return label_dict['behaviors'], [str(col_map(tab20_map(i))) for i in range(len(label_dict['behaviors']))]

@eel.expose
def update_counts():
    global label_dict_path
    global label_dict

    config_path = os.path.join(os.path.split(label_dict_path)[0], 'config.yaml')

    for b in label_dict['behaviors']:
        insts = label_dict['labels'][b]

        update_metrics(config_path, b, 'Train #', int(round(len(insts)*.75)))
        update_metrics(config_path, b, 'Test #', int(round(len(insts)*.25)))

        eel.updateCount(b, len(insts))()


@eel.expose
def update_metrics(config_path, behavior, group, value):

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    config['metrics'][behavior][group] = value

    with open(config_path, 'w+') as file:
        yaml.dump(config, file, allow_unicode=True)


@eel.expose
def get_active_streams():

    if len(active_streams.keys())>0:
        return list(active_streams.keys())
    return False

@eel.expose
def stop_camera_stream(camera_name):
    global active_streams

    if camera_name in active_streams.keys():
        active_streams[camera_name].communicate(input=b'q')
        active_streams.pop(camera_name)

        return True

    return False

@eel.expose
def kill_streams():
    global active_streams
    global stop_threads
    global thread
    global tthread

    for stream in active_streams.keys():
        active_streams[stream].communicate(input=b'q')

    stop_threads = True
    thread.raise_exception()
    thread.join()

    if tthread:
        tthread.raise_exception()
        tthread.join()

    for cthread in classification_threads:
        cthread.raise_exception()
        cthread.join()

eel.start('frontend/index.html', mode='electron', block=False)

thread.start()

while True:
    eel.sleep(1.0)
