from model import MusicTransformer, MusicTransformerDecoder
from custom.layers import *
from custom import callback
import params as par
from tensorflow.python.keras.optimizer_v2.adam import Adam
from data import Data
import os
import datetime
import argparse
from midi_processor.processor import decode_midi, encode_midi

current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
gen_log_dir = 'logs/mt_decoder/generate_'+current_time+'/generate'
gen_summary_writer = tf.summary.create_file_writer(gen_log_dir)

parser = argparse.ArgumentParser()

# Set default parameters
parser.add_argument('--dataset_name',type=str)
parser.add_argument('--max_seq', default=2048, help="Maximum generated note sequence. Should match the trained model's.", type=int)
parser.add_argument('--load_path', help='Saved model path', type=str)
parser.add_argument('--mode', default='dec')
parser.add_argument('--beam', default=None, type=int)
parser.add_argument('--length', default=2048, type=int)
parser.add_argument('--input',default='/data/ironman/data1/users/astais/Unprocessed-Datasets/Primers/primer_1.mid', help='Input Sequence',type=str)
parser.add_argument('--input_tokens',default='300',help='Number of input tokens to include as primer',type=int)
parser.add_argument('--num', default=50, help='Number of midis to generate', type=int)
parser.add_argument('--num_skip', default=0, help='Number of previously created midis to skip')
parser.add_argument('--save_path', default='/data/data1/users/astais/Midi-Outputs/MusicTransformer-tensorflow2.0/', help='Default Save Path', type=str)

args = parser.parse_args()

# set arguments
max_seq = args.max_seq
load_path = args.load_path
mode = args.mode
beam = args.beam
length = args.length
dataset_name=args.dataset_name
inputs=encode_midi(args.input)
primer_num=args.input[-5] # Primer number printed in midi output name
num=args.num
default_path=args.save_path
input_tokens=args.input_tokens
num_skip=args.num_skip

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

os.makedirs(default_path+dataset_name, exist_ok=True)


# Generate midis
for x in range(num):

    # Create midi file for the generated sequence
    current_time = datetime.datetime.now().strftime('%Y%m%d')
    file_name=dataset_name+'_'+str(primer_num)+'_'+str(input_tokens)+'_'+current_time+'_'+str(x+1+num_skip)+'.mid'
    # print(file_name,flush=True)
    save_path=default_path+dataset_name+'/'+file_name
    open(save_path, 'w').close()

    # Generate midi
    with gen_summary_writer.as_default():
        result = mt.generate(inputs[:input_tokens], beam=beam, length=length, tf_board=True)

    # Decode midi according to mode
    if mode == 'enc-dec':
        decode_midi(list(inputs[-1*par.max_seq:]) + list(result[1:]), file_path=save_path)
    else:
        decode_midi(result, file_path=save_path)

    print("Midi "+str(x+1)+" generated.",flush=True)

os.rmdir(gen_log_dir)


    
