# Generate ouput midis for multiple models, primers and input sequences

import string
from pyparsing import str_type
from model import MusicTransformer, MusicTransformerDecoder
from custom.layers import *
from custom import callback
import params as par
from tensorflow.python.keras.optimizer_v2.adam import Adam
from data import Data
import os, sys, random
import datetime
import argparse
from pathlib import Path
from midi_processor.processor import decode_midi, encode_midi

def get_random_files(ext, top=os.getcwd()):
    file_list = list(Path(top).glob(f"**/*.{ext}"))
    if not len(file_list):
        return f"No files matched that extension: {ext}"
    rand = random.randint(0, len(file_list) - 1)
    return file_list[rand]

parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument('--load_path', help='Default saved model path', type=str, required=True)
parser.add_argument('--save_path', help='Default Save Path', type=str, required=True)
parser.add_argument('--input_path', help='Default Primers Path',type=str, required= True)

# Optional parameters
parser.add_argument('--models', default="maestro-v3.0.0, GiantMIDI-Piano", type=str)
parser.add_argument('--primers', default="1,8", type=str, help='Primer number name, 0 for random')
parser.add_argument('--token_num', default="128,256,512,1024", type=str)
parser.add_argument('--num', default=50, help='Number of midis to generate per model/primer/token number.', type=int)
parser.add_argument('--num_skip', default=0, help='Number of previously created midis to skip.', type=int)
parser.add_argument('--max_seq', default=2048, help="Maximum generated note sequence. Should match the trained model's.", type=int)
parser.add_argument('--mode', default='dec')
parser.add_argument('--beam', default=None, type=int)
parser.add_argument('--length', default=2048, type=int)
args = parser.parse_args()

# set arguments
max_seq = args.max_seq
mode = args.mode
beam = args.beam
length = args.length
def_input=args.input_path
num=args.num
def_save_path=args.save_path
def_load_path=args.load_path

models=args.models.replace(" ", "").split(",")
token_num=args.token_num.replace(" ", "").split(",")
token_num = list(map(int, token_num))
primers=args.primers.replace(" ", "").split(",")
primers = list(map(int, primers))

print(models)
print(token_num)
print(primers)

current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# For all the models
for model in models:
    # Set model load path
    print(">> Generating for "+model+" model")
    load_path=def_load_path+model   # model load path

    # Create model save path
    os.makedirs(def_save_path+'music-transformer_'+model, exist_ok=True)
    save_path=def_save_path+'music-transformer_'+model
        
    # Choose generation mode
    if mode == 'enc-dec':
        print(">> Generating with original seq2seq wise... beam size is {}".format(beam))
        mt = MusicTransformer(
                embedding_dim=256,
                vocab_size=par.vocab_size,
                num_layer=6,
                max_seq=2048,
                dropout=0.2,
                debug=False, loader_path=load_path)
    else:
        print(">> Generating with decoder wise... beam size is {}".format(beam))
        mt = MusicTransformerDecoder(loader_path=load_path)


    # for different primers
    for primer_num in range(primers[0],primers[-1]+1):
        
        # primer code for random primer
        if(primer_num==0):
            print(">> Generating with random primer.")
        
        # specific primer
        else:
            print(">> Generating with primer "+str(primer_num)+" as primer.")
            inputs=encode_midi(def_input+str(primer_num)+'.mid')

        # for different primer token numbers
        for primer_tokens in token_num:
            print(">> Generating with "+str(primer_tokens)+" tokens as input from primer.")
            
            # Create primer, tokens number save path
            os.makedirs(save_path+'/music-transformer_'+model+'_'+str(primer_num)+'_'+str(primer_tokens)+'/', exist_ok=True)
            primer_save_path=save_path+'/music-transformer_'+model+'_'+str(primer_num)+'_'+str(primer_tokens)+'/'
            
            # Generate num multiple midis
            for x in range(num):

                # Create midi file for the generated sequence
                current_time = datetime.datetime.now().strftime('%Y%m%d')
                file_name='music-transformer_'+model+'_'+str(primer_num)+'_'+str(primer_tokens)+'_'+current_time+'_'+str(x+1+args.num_skip)+'.mid'
                midi_save_path=primer_save_path+file_name
                open(midi_save_path, 'w').close()

                # Change primer for every generated midi
                if(primer_num==0):
                    inputs=encode_midi(str(get_random_files("mid",def_input)))

                # Retry generating midi until no error 
                while True:
                    try:
                        # Generate midi
                        result = mt.generate(inputs[:primer_tokens], beam=beam, length=length, tf_board=True)

                        # Decode midi according to mode
                        if mode == 'enc-dec':
                            decode_midi(list(inputs[-1*par.max_seq:]) + list(result[1:]), file_path=midi_save_path)
                        else:
                            decode_midi(result, file_path=midi_save_path)

                        print("Midi "+str(x+1+args.num_skip)+" generated.",flush=True)
                    except:
                        continue
                    else:
                        break