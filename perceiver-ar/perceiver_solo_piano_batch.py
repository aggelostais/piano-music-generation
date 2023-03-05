#!/usr/bin/env python
# coding: utf-8

# # Perceiver Solo Piano Batch (ver. 2022.11.30)
# 
# ***
# 
# Powered by tegridy-tools: https://github.com/asigalov61/tegridy-tools
# 
# ***
# 
# WARNING: This complete implementation is a functioning model of the Artificial Intelligence. Please excercise great humility, care, and respect. https://www.nscai.gov/
# 
# ***
# 
# #### Project Los Angeles
# 
# #### Tegridy Code 2022
# 
# ***

# # (Setup Environment)

# In[1]:


#@title nvidia-smi gpu check
get_ipython().system('nvidia-smi')
get_ipython().system('export CUDA_VISIBLE_DEVICES=0')


# In[2]:


#@title Install all dependencies (run only once per session)

# !git clone https://github.com/asigalov61/Perceiver-Music-Transformer
# !pip install einops
# !pip install torch
# !pip install torch-summary

# !pip install tqdm
# !pip install matplotlib

# !pip install fluidsynth #Pip does not work for some reason. Only apt works
# !pip install FluidSynth
# !pip install midi2audio
# !pip install music21


# In[3]:


#@title Import all needed modules

print('Loading needed modules. Please wait...',flush=True)
import os, datetime, random,math, copy, pickle
from collections import OrderedDict
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torchsummary import summary

print('Loading core modules...',flush=True)
import TMIDIX
from perceiver_ar_pytorch import PerceiverAR
from autoregressive_wrapper import AutoregressiveWrapper

os.chdir('/home/astais/perceiver-ar')
print('Done!',flush=True)


# ## Set Model & Basic Directories

# In[25]:


model_name="perceiver-ar_GiantMIDI-Piano"
model_filename="perceiver-ar_GiantMIDI-Piano_75800_steps_0.0192_loss_0.9424_acc*.pth"
# model_name="perceiver-ar_Los-Angeles-MIDI-Dataset-segment"
# model_filename="perceiver-ar_Los-Angeles-MIDI-Dataset-segment_164400_steps_0.0037_loss*.pth"
# model_name="perceiver-ar_pretrained"
# model_filename="Pretrained.pth"
primers_base="/data/data1/users/astais/Unprocessed-Datasets/"
models_base="/data/data1/users/astais/Saved-Models/perceiver-ar/"
save_base="/data/data1/users/astais/Midi-Outputs/perceiver-ar/"
save_addr=save_base+model_name+"/"

for addr in [save_addr]:
    if not os.path.exists(addr):
        os.makedirs(addr)


# ## (Download Model)

# In[26]:


# #@title Download Perceiver Pre-Trained Solo Piano Model
# !wget --no-check-certificate -O 'Perceiver-Solo-Piano-Model.pth' "https://onedrive.live.com/download?cid=8A0D502FC99C608F&resid=8A0D502FC99C608F%2118753&authkey=AMmtup34-lfGqyA"


# ## Load Primers List

# In[27]:


file_to_read = open(primers_base+"primers.pickle", "rb")
primers_dict = pickle.load(file_to_read)
print(len(primers_dict),flush=True)


# ## Load Model Checkpoint

# In[28]:


#@title Load/Reload the model
full_path_to_model_checkpoint = models_base+model_filename
print("Loading model: "+full_path_to_model_checkpoint,flush=True)

# constants
SEQ_LEN = 4096 * 4 # Total of 16k
PREFIX_SEQ_LEN = (4096 * 4) - 1024 # 15.3k

model = PerceiverAR(
    num_tokens = 512, # vocabulary size
    dim = 1024,
    depth = 24,
    heads = 16,
    dim_head = 64,
    cross_attn_dropout = 0.5,
    max_seq_len = SEQ_LEN,
    cross_attn_seq_len = PREFIX_SEQ_LEN
)
model = AutoregressiveWrapper(model)
model.cuda()
state_dict = torch.load(full_path_to_model_checkpoint)
model.load_state_dict(state_dict)
model.eval()
print('Done!',flush=True)

# Model stats
# summary(model)


# ## Loading Primer MIDI

# In[29]:


#==================================================

# Memories augmentator

def augment(inputs):
  outs = []
  outy = []
  for i in range(1, 12):
    out1 = []
    out2 = []
    for j in range(0, len(inputs), 4):
      note = inputs[j:j+4]
      aug_note1 = copy.deepcopy(note)
      aug_note2 = copy.deepcopy(note)
      aug_note1[2] += i
      aug_note2[2] -= i
      out1.append(aug_note1)
      out2.append(aug_note2)
    outs.append(out1[random.randint(0, int(len(out1) / 2)):random.randint(int(len(out1) / 2), len(out1))])
    outs.append(out2[random.randint(0, int(len(out2) / 2)):random.randint(int(len(out2) / 2), len(out2))])

  for i in range(64):
    outy.extend(random.choice(outs))

  outy1 = []
  for o in outy:
    outy1.extend(o)

  return outy1

#==================================================

# Batch Generation Parameters
number_of_prime_tokens = 512 #@param {type:"slider", min:256, max:1020, step:4}
number_of_tokens_to_generate = 512 #@param {type:"slider", min:64, max:512, step:32}
temperature = 0.8 #@param {type:"slider", min:0.1, max:1, step:0.1}


# ## Batch Generation, Single Mode, Primer List

# In[36]:


# Iterate through primers dictionary
for name,path in tqdm(primers_dict.items()):
    #==========================Loading Primer=========================================
    primer=path
#     print(name)
#     print(path)
    score = TMIDIX.midi2ms_score(open(primer, 'rb').read())
    events_matrix = []
    itrack = 1
    while itrack < len(score):
        for event in score[itrack]:         
            if event[0] == 'note' and event[3] != 9:
                events_matrix.append(event)
        itrack += 1
    if len(events_matrix) > 0:
        # Sorting...
        events_matrix.sort(key=lambda x: x[4], reverse=True)
        events_matrix.sort(key=lambda x: x[1])
        # recalculating timings
        for e in events_matrix:
            e[1] = int(e[1] / 10)
            e[2] = int(e[2] / 20)
        # final processing...
        inputs = []
        inputs.extend([126+0, 126+128, 0+256, 0+384]) # Intro/Zero sequence
        pe = events_matrix[0]

        for e in events_matrix:
            time = max(0, min(126, e[1]-pe[1]))
            dur = max(1, min(126, e[2]))
            ptc = max(1, min(126, e[4]))
            vel = max(1, min(126, e[5]))
            inputs.extend([time+0, dur+128, ptc+256, vel+384])
            pe = e
#     print('Loaded primer MIDI file!')
    
    #==========================Generating Output=========================================
    #inp = augment(inputs)
    inp = inputs * math.ceil((4096 * 4) / len(inputs))
    inp = inp[:(4096 * 4)]
    inp = inp[(512+len(inputs[:number_of_prime_tokens])):] + inputs[:number_of_prime_tokens]
    inp = torch.LongTensor(inp).cuda()
    out = model.generate(inp[None, ...], 
                         number_of_tokens_to_generate, 
                         temperature=temperature)  
    out1 = out.cpu().tolist()[0]
    # Filename
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name=save_addr+model_name+'_'+name+'_'+current_time

    # Song with primer tokens
    if len(out1) != 0:
        song = inputs[:number_of_prime_tokens] + out1 # output song
        song_f = []
        time = 0
        dur = 0
        vel = 0
        pitch = 0
        channel = 0
        son = []
        song1 = []
        for s in song:
          if s > 127:
            son.append(s)
          else:
            if len(son) == 4:
              song1.append(son)
            son = []
            son.append(s)

        for s in song1:
            channel = 0 # Piano
            time += s[0] * 10          
            dur = (s[1]-128) * 20
            pitch = (s[2]-256)
            vel = (s[3]-384)
            if pitch != 0:                     
              song_f.append(['note', time, dur, channel, pitch, vel ])

        detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                            output_signature = 'perceiver-ar',  
                                                            output_file_name = file_name, 
                                                            track_name=model_name+'_'+name,
                                                            list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 53, 19, 0, 0, 0, 0],
                                                            number_of_ticks_per_quarter=500)


    # Trimmed song without primer tokens
    if len(out1) != 0:
        song = out1 # output song
        song_f = []
        time = 0
        dur = 0
        vel = 0
        pitch = 0
        channel = 0
        son = []
        song1 = []
        for s in song:
          if s > 127:
            son.append(s)
          else:
            if len(son) == 4:
              song1.append(son)
            son = []
            son.append(s)

        for s in song1:
                  # Set the channel to 0 (Piano)
                  channel = 0 
                  # Increase the time by the first element of s multiplied by 10
                  time += s[0] * 10          
                  # Calculate the duration of the note
                  dur = (s[1]-128) * 20
                  # Calculate the pitch of the note
                  pitch = (s[2]-256)
                  # Calculate the velocity of the note
                  vel = (s[3]-384)
                  # If the pitch is not 0, append the note to the song
                  if pitch != 0:                     
                    song_f.append(['note', time, dur, channel, pitch, vel ])

       # Convert the song to MIDI
          detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                              # Set the output signature
                                                              output_signature = 'perceiver-ar',  
                                                              # Set the output file name
                                                              output_file_name = file_name+"_no-primer", 
                                                              # Set the track name
                                                              track_name=model_name+'_'+name+"_no-primer",
                                                              # Set the list of MIDI patches
                                                              list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 53, 19, 0, 0, 0, 0],
                                                              # Set the number of ticks per quarter
                                                              number_of_ticks_per_quarter=500)

    # Print "Done!"
    print("Done!")