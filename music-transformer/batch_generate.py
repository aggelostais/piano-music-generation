# Version 2023.01.04
# Generate ouput midis for multiple models, primers and input sequences
from model import MusicTransformer, MusicTransformerDecoder
from custom.layers import *
import params as par
from tqdm import tqdm
import os, datetime, argparse, pickle
from midi_processor.processor import decode_midi, encode_midi

parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument('--basic_dir', default="/data/data1/users/astais/")
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--model_filepath', type=str, required=True)
# Optional parameters
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
model_name="music-transformer_"+args.model_name
load_path=basic_dir+"Saved-Models/music-transformer/"+args.model_filepath # Loaded model path name
save_path=basic_dir+"Midi-Outputs/music-transformer/"+model_name+"/"      # Basic save path
for addr in [save_path,save_path+"/no_primer/",save_path+"/primer/"]:
    os.makedirs(addr, exist_ok=True)           # Create model save paths

print(">> Generating for "+model_name+" model")

# Load Primers List
file_to_read = open(basic_dir+"Unprocessed-Datasets/"+"primers.pickle", "rb")
primers_dict = pickle.load(file_to_read)
print(len(primers_dict),flush=True)

# Choose generation mode
if mode == 'enc-dec':
    print(">> Generating with original seq2seq wise... beam size is {}".format(beam))
    mt = MusicTransformer(
            embedding_dim=256,
            vocab_size=par.vocab_size,
            num_layer=6,
            max_seq=1024,
            dropout=0.2,
            debug=False, loader_path=load_path)
else:
    print(">> Generating with decoder wise... beam size is {}".format(beam))
    mt = MusicTransformerDecoder(loader_path=load_path)


# for different primers
for name,path in tqdm(primers_dict.items()):
    
    primer=path
    inputs=encode_midi(primer)

    # Create midi files for the generated sequence
    current_time = datetime.datetime.now().strftime('%Y%m%d')
    midi_save_path=save_path+"/primer/"+model_name+'_'+name+'_'+current_time+'.mid'
    midi_save_path2=save_path+"/no_primer/"+model_name+'_'+name+'_'+current_time+"_no-primer.mid"
    # If file exists skip
    if(os.path.exists(midi_save_path)):
        continue
    open(midi_save_path, 'w').close()
    open(midi_save_path2, 'w').close()

    # Retry generating midi until no error 
    while True:
        try:
            #Generate midi
                result = mt.generate(inputs[:token_num], beam=beam, length=length, tf_board=True)
                result2=result[-512:]   # no primer result

                # Decode midi according to mode
                if mode == 'enc-dec':
                    decode_midi(list(inputs[-1*par.max_seq:]) + list(result[1:]), file_path=midi_save_path)
                    decode_midi(list(inputs[-1*par.max_seq:]) + list(result2[1:]), file_path=midi_save_path2)
                else:
                    decode_midi(result, file_path=midi_save_path)
                    decode_midi(result2, file_path=midi_save_path2)

                print("Midi "+name+" generated.",flush=True)
        except:
            continue
        else:
            break