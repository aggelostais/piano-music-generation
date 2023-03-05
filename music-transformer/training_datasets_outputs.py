# Version 2023.01.12
# Create training-datasets real outputs, primers suitable for human evaluation
from model import MusicTransformer, MusicTransformerDecoder
from custom.layers import *
import params as par
from tqdm import tqdm
import os, datetime, argparse, pickle
from midi_processor.processor import decode_midi, encode_midi

parser = argparse.ArgumentParser()

parser.add_argument('--basic_dir', default="/data/data1/users/astais/")
parser.add_argument('--token_num', default=512, type=int)
parser.add_argument('--max_seq', default=2048, help="Maximum generated note sequence. Should match the trained model's.", type=int)
parser.add_argument('--mode', default='dec')
parser.add_argument('--beam', default=None, type=int)
parser.add_argument('--length', default=1024, type=int)
args = parser.parse_args()

# set arguments
max_seq = args.max_seq
mode = args.mode
beam = args.beam
length = args.length
basic_dir=args.basic_dir
token_num=args.token_num

current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# Set Model & Basic Directories
save_path=basic_dir+"Midi-Outputs/training_datasets"      # Basic save path
for addr in [save_path,save_path+"/no_primer/",save_path+"/primer/"]:
    os.makedirs(addr, exist_ok=True)           # Create model save paths

print(">> Generating for training datasets outputs")

# Load Primers List
file_to_read = open(basic_dir+"Unprocessed-Datasets/"+"primers.pickle", "rb")
primers_dict = pickle.load(file_to_read)
print(len(primers_dict),flush=True)

# for different primers
for name,path in tqdm(primers_dict.items()):
    
    primer=path
    inputs=encode_midi(primer)

    # Create midi files for the generated sequence
    current_time = datetime.datetime.now().strftime('%Y%m%d')
    primer_save_path=basic_dir+"Midi-Outputs/primers/primer_"+name+'_'+current_time+'.mid'
    midi_save_path=save_path+"/primer/training-datasets_"+name+'_'+current_time+'.mid'
    midi_save_path2=save_path+"/no_primer/training-datasets_"+name+'_'+current_time+"_no-primer.mid"
    # If file exists skip
    if(os.path.exists(midi_save_path)):
        continue
    open(midi_save_path, 'w').close()
    open(midi_save_path2, 'w').close()

    # Retry until no error 
    while True:
        try:
                primer=inputs[:512]     # primer
                result = inputs[:1024]  # take only 1024 tokens of primer output
                result2=result[-512:]   # take the last 512 tokens out of the total 1024

                # Decode midi according to mode
                if mode == 'enc-dec':
                    decode_midi(list(inputs[-1*par.max_seq:]) + list(result[1:]), file_path=midi_save_path)
                    decode_midi(list(inputs[-1*par.max_seq:]) + list(result2[1:]), file_path=midi_save_path2)
                else:
                    decode_midi(result, file_path=midi_save_path)
                    decode_midi(result2, file_path=midi_save_path2)
                    decode_midi(primer, file_path=primer_save_path)

                print("Midi "+name+" generated.",flush=True)
        except:
            continue
        else:
            break