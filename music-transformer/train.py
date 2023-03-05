# Version 2022.12.26
from model import MusicTransformerDecoder
from custom.layers import *
from custom import callback
import params as par
from tensorflow.python.keras.optimizer_v2.adam import Adam
from data import Data
import utils, os, argparse, datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

tf.executing_eagerly()
parser = argparse.ArgumentParser()

parser.add_argument('--l_r', default=None, help='Base Learning Rate for Adam Algorithm', type=float)
parser.add_argument('--batch_size', default=2, help='batch size', type=int)
parser.add_argument('--dataset_name', help='Training Dataset Name', type=str)
parser.add_argument('--max_seq', default=2048, help='Maximum Token Sequence', type=int)
parser.add_argument('--epochs', default=1500, type=int)
parser.add_argument('--basic_dir', default="/users/pa21/ptzouv/astais/", help='Basic directory for Saved-Models,Training-Outputs and Processed-Datasets folders', type=str)
parser.add_argument('--load_dir', default=None, help='Partially trained model path which training will be continued', type=str)
parser.add_argument('--is_reuse', default=False)
parser.add_argument('--multi_gpu', default=True)
parser.add_argument('--num_layers', default=6, type=int) 
parser.add_argument('--dropout', default=0.2, type=float) 
args = parser.parse_args()

# set arguments
l_r = args.l_r
batch_size = args.batch_size
max_seq = args.max_seq
epochs = args.epochs
is_reuse = args.is_reuse
load_path = args.load_dir
multi_gpu = args.multi_gpu
num_layer = args.num_layers
dropout=args.dropout

# Set dataset, training output and save directories
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M')
dataset_name=args.dataset_name
basic_dir=args.basic_dir
dataset_addr=basic_dir+"Processed-Datasets/music-transformer/"+'music-transformer_'+dataset_name+'/'
base_save_dir=basic_dir+'Saved-Models/music-transformer/'+'music-transformer_'+dataset_name+'_'+current_time+'/'
train_out_dir=basic_dir+'Training-Outputs/music-transformer/'+'music-transformer_'+dataset_name+'_'+current_time+'/'
for addr in [base_save_dir,train_out_dir]:
    if not os.path.exists(addr):
        os.makedirs(addr)

# load data
dataset = Data(dataset_addr)

# load model
learning_rate = callback.CustomSchedule(par.embedding_dim) if l_r is None else l_r
opt = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# define model
mt = MusicTransformerDecoder(
            embedding_dim=256,
            vocab_size=par.vocab_size,
            num_layer=num_layer,
            max_seq=max_seq,
            dropout=dropout, 
            debug=False, loader_path=load_path)
mt.compile(optimizer=opt, loss=callback.transformer_dist_train_loss)

# define tensorboard writer
train_log_dir = basic_dir+'logs/mt_decoder/'+current_time+'/train'
eval_log_dir = basic_dir+'logs/mt_decoder/'+current_time+'/eval'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)

# Print Model Parameters
print('\n====================================================', flush=True)
print("Music Transformer Parameters \nBatch Size: "+ str(batch_size),flush=True)
print("Dataset Directory: "+dataset_addr,flush=True)
print("Dataset Files: "+str(dataset),flush=True)
if (load_path==None):
    print("Load Directory: None",flush=True)
else:
    print("Load Directory: "+load_path,flush=True)
print("Save Directory: "+base_save_dir,flush=True)
print("Model Layers: "+str(num_layer),flush=True)
print("Dropout: "+str(dropout),flush=True)


# Train, Evaluation Metrics Arrays
train_losses = []
val_losses = []
train_accs = []
val_accs = []

best_val_acc        = 0.000
best_val_acc_step  = -1
best_val_loss       = float("inf")
best_val_loss_step = -1

# Training
idx = 0
batch_range=len(dataset.files) // batch_size
for e in range(epochs):
    mt.reset_metrics()
    for b in range(batch_range):   # For current batch
        try:
            batch_x, batch_y = dataset.slide_seq2seq_batch(batch_size, max_seq)
        except:
            continue
        result_metrics = mt.train_on_batch(batch_x, batch_y)

        # Every 100 validate model
        if b % 100 == 0:
            eval_x, eval_y = dataset.slide_seq2seq_batch(batch_size, max_seq, 'eval')
            eval_result_metrics, weights = mt.evaluate(eval_x, eval_y)
            if b == 0:
                tf.summary.histogram("target_analysis", batch_y, step=e)
                tf.summary.histogram("source_analysis", batch_x, step=e)

            tf.summary.scalar('loss', result_metrics[0], step=idx)
            tf.summary.scalar('accuracy', result_metrics[1], step=idx)

            if b == 0:
                mt.sanity_check(eval_x, eval_y, step=e)

            tf.summary.scalar('loss', eval_result_metrics[0], step=idx)
            tf.summary.scalar('accuracy', eval_result_metrics[1], step=idx)
            for i, weight in enumerate(weights):
                with tf.name_scope("layer_%d" % i):
                    with tf.name_scope("w"):
                        utils.attention_image_summary(weight, step=idx)
                # for i, weight in enumerate(weights):
                #     with tf.name_scope("layer_%d" % i):
                #         with tf.name_scope("_w0"):
                #             utils.attention_image_summary(weight[0])
                #         with tf.name_scope("_w1"):
                #             utils.attention_image_summary(weight[1])
            idx += 1
            print('\n====================================================', flush=True)
            print('Epoch/Batch: {}/{}'.format(e, b), flush=True)
            print('Train >>>> Loss: {:6.6}, Accuracy: {}'.format(result_metrics[0], result_metrics[1]), flush=True)
            print('Eval >>>> Loss: {:6.6}, Accuracy: {}'.format(eval_result_metrics[0], eval_result_metrics[1]), flush=True)
            
            if(b==0):
                # Save training metrics
                train_loss=result_metrics[0]
                train_acc=result_metrics[1]
                val_loss=eval_result_metrics[0]
                val_acc=eval_result_metrics[1]

                train_losses.append(train_loss)
                train_accs.append(train_acc)
                val_losses.append(val_loss)
                val_accs.append(val_acc)

                # Save Model & Plot Graphs: Trained for necessery batches and new evaluation loss or accuracy best
                if (val_loss < best_val_loss and val_loss<2) or (val_acc > best_val_acc and val_acc>0.3):

                    # Check if val_acc or val_loss are best so far
                    if(val_loss < best_val_loss and val_loss<2):
                            best_val_loss       = val_loss
                            best_val_loss_step = e*batch_range
                            best_loss_dir = base_save_dir+'music-transformer_'+dataset_name+'_'+ str(e*batch_range)+'_steps_' + str(round(float(val_losses[-1]), 4)) + '_loss'
                            os.makedirs(best_loss_dir)
                            mt.save(best_loss_dir)

                    elif(val_acc > best_val_acc and val_acc>0.3):
                            best_val_acc = val_acc
                            best_val_acc_step  = e*batch_range
                            best_acc_dir = base_save_dir+'music-transformer_'+dataset_name+'_'+ str(e*batch_range)+'_steps_' + str(round(float(val_accs[-1]), 4)) + '_acc'
                            os.makedirs(best_acc_dir)
                            mt.save(best_acc_dir)
                    
                    print('\n====================================================', flush=True)
                    print("Best val acc step:", best_val_acc_step,flush=True)
                    print("Best val acc:", best_val_acc, flush=True)
                    print("Best val loss step:", best_val_loss_step, flush=True)
                    print("Best val loss:", best_val_loss, flush=True)
                    print('Saved model progress.',flush=True)
                    
                    # Plotting Training, Evaluation Graphs
                    fig, ax = plt.subplots(dpi=300)
                    ax.plot([batch_range*i for i in range(len(train_losses))] ,train_losses, '#607B8B')
                    ax.set(xlabel='Steps', ylabel ='Training Loss', title='Music-Transformer '+dataset_name+': Training Loss')
                    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                    ax.grid(which='minor', alpha=0.2)
                    ax.grid(which='major', alpha=0.5)
                    fig.savefig(train_out_dir+'music-transformer_'+dataset_name+'_train-loss_'+str(e*batch_range) + '_steps.png')
                    plt.close()

                    # Training acc graph
                    fig, ax = plt.subplots(dpi=300)
                    ax.plot([batch_range*i for i in range(len(train_accs))] ,train_accs, '#607B8B')
                    ax.set(xlabel='Steps', ylabel ='Training Accuracy', title='Music-Transformer '+dataset_name+': Training Accuracy')
                    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                    ax.grid(which='minor', alpha=0.2)
                    ax.grid(which='major', alpha=0.5)
                    fig.savefig(train_out_dir+'music-transformer_'+dataset_name+'_train-acc_'+str(e*batch_range) + '_steps.png')
                    plt.close()

                    # Validation loss graph
                    fig, ax = plt.subplots(dpi=300)
                    ax.plot([batch_range*i for i in range(len(val_losses))] ,val_losses, '#5D478B')
                    ax.set(xlabel='Steps', ylabel ='Validation Loss', title='Music-Transformer '+dataset_name+': Validation Loss')
                    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                    ax.grid(which='minor', alpha=0.2)
                    ax.grid(which='major', alpha=0.5)
                    fig.savefig(train_out_dir+'music-transformer_'+dataset_name+'_val-loss_'+str(e*batch_range) + '_steps.png')
                    plt.close()

                    # Validation acc graph
                    fig, ax = plt.subplots(dpi=300)
                    plt.plot([batch_range*i for i in range(len(val_accs))] ,val_accs, '#5D478B')
                    ax.set(xlabel='Steps', ylabel ='Validation Accuracy', title='Music-Transformer '+dataset_name+': Validation Accuracy')
                    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                    ax.grid(which='minor', alpha=0.2)
                    ax.grid(which='major', alpha=0.5)
                    fig.savefig(train_out_dir+'music-transformer_'+dataset_name+'_val-acc_'+str(e*batch_range) + '_steps.png')
                    plt.close()

                    # Training-Validation Loss Graph
                    fig, ax = plt.subplots(dpi=300)
                    x_train=np.array([batch_range*i for i in range(len(train_losses))])
                    y_train=np.array(train_losses)
                    ax.plot(x_train,y_train, '#607B8B')
                    x_val=np.array([batch_range*i for i in range(len(val_losses))])
                    y_val=np.array(val_losses)
                    ax.plot(x_val,y_val, '#5D478B')
                    ax.set(xlabel='Steps', ylabel ='Loss', title='Music-Transformer '+dataset_name+': Training-Validation Loss')
                    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                    ax.legend(['Training Loss','Validation Loss'], loc='upper right')
                    ax.grid(which='minor', alpha=0.2)
                    ax.grid(which='major', alpha=0.5)
                    fig.savefig(train_out_dir+'music-transformer_'+dataset_name+'_train,val-loss_'+str(e*batch_range) + '_steps.png')
                    plt.close()

                    # Training-Validation Accuracy Graph
                    fig, ax = plt.subplots(dpi=300)
                    x_train=np.array([batch_range*i for i in range(len(train_accs))])
                    y_train=np.array(train_accs)
                    ax.plot(x_train,y_train, '#607B8B')
                    x_val=np.array([batch_range*i for i in range(len(val_accs))])
                    y_val=np.array(val_accs)
                    ax.plot(x_val,y_val, '#5D478B')
                    ax.set(xlabel='Steps', ylabel ='Accuracy', title='Music-Transformer '+dataset_name+': Training-Validation Accuracy')
                    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                    ax.legend(['Training Accuracy','Validation Accuracy'], loc='lower right')
                    ax.grid(which='minor', alpha=0.2)
                    ax.grid(which='major', alpha=0.5)
                    fig.savefig(train_out_dir+'music-transformer_'+dataset_name+'_train,val-acc_'+str(e*batch_range) + '_steps.png')
                    plt.close() 

                    # Save training, validation value arrays
                    with open(train_out_dir+'music-transformer_'+dataset_name+"_"+str(e*batch_range) + '_steps.npy', 'wb') as f:
                        np.save(f, np.array(train_losses))
                        np.save(f, np.array(train_accs))
                        np.save(f, np.array(val_losses))
                        np.save(f, np.array(val_accs))                    

    # Early Stopping: If best val_loss or acc hasn't appeared for 100*batch_range steps stop training
    if e>50 and (val_accs[-1] < best_val_acc and val_losses[-1] < best_val_loss and e>best_val_loss_step+100*batch_range and e>best_val_acc_step+100*batch_range):
        print('\n====================================================', flush=True)
        print("Early Stopping: Training Stopped!", flush=True)
        break

# Save training, validation value arrays
with open(train_out_dir+'music-transformer_'+dataset_name+"_"+current_time+'.npy', 'wb') as f:
    np.save(f, np.array(train_losses))
    np.save(f, np.array(train_accs))
    np.save(f, np.array(val_losses))
    np.save(f, np.array(val_accs))

# Plotting Training, Evaluation Graphs
fig, ax = plt.subplots(dpi=300)
ax.plot([batch_range*i for i in range(len(train_losses))] ,train_losses, '#607B8B')
ax.set(xlabel='Steps', ylabel ='Training Loss', title='Music-Transformer '+dataset_name+': Training Loss')
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)
fig.savefig(train_out_dir+'music-transformer_'+dataset_name+'_train-loss_'+str(e*batch_range) + '_steps.png')
plt.close()

# Training acc graph
fig, ax = plt.subplots(dpi=300)
ax.plot([batch_range*i for i in range(len(train_accs))] ,train_accs, '#607B8B')
ax.set(xlabel='Steps', ylabel ='Training Accuracy', title='Music-Transformer '+dataset_name+': Training Accuracy')
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)
fig.savefig(train_out_dir+'music-transformer_'+dataset_name+'_train-acc_'+str(e*batch_range) + '_steps.png')
plt.close()

# Validation loss graph
fig, ax = plt.subplots(dpi=300)
ax.plot([batch_range*i for i in range(len(val_losses))] ,val_losses, '#5D478B')
ax.set(xlabel='Steps', ylabel ='Validation Loss', title='Music-Transformer '+dataset_name+': Validation Loss')
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)
fig.savefig(train_out_dir+'music-transformer_'+dataset_name+'_val-loss_'+str(e*batch_range) + '_steps.png')
plt.close()

# Validation acc graph
fig, ax = plt.subplots(dpi=300)
plt.plot([batch_range*i for i in range(len(val_accs))] ,val_accs, '#5D478B')
ax.set(xlabel='Steps', ylabel ='Validation Accuracy', title='Music-Transformer '+dataset_name+': Validation Accuracy')
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)
fig.savefig(train_out_dir+'music-transformer_'+dataset_name+'_val-acc_'+str(e*batch_range) + '_steps.png')
plt.close()

# Training-Validation Loss Graph
fig, ax = plt.subplots(dpi=300)
x_train=np.array([batch_range*i for i in range(len(train_losses))])
y_train=np.array(train_losses)
ax.plot(x_train,y_train, '#607B8B')

x_val=np.array([batch_range*i for i in range(len(val_losses))])
y_val=np.array(val_losses)
ax.plot(x_val,y_val, '#5D478B')

ax.set(xlabel='Steps', ylabel ='Loss', title='Music-Transformer '+dataset_name+': Training-Validation Loss')
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.legend(['Training Loss','Validation Loss'], loc='upper right')
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)
fig.savefig(train_out_dir+'music-transformer_'+dataset_name+'_train,val-loss_'+str(e*batch_range) + '_steps.png')
plt.close()

# Training-Validation Accuracy Graph
fig, ax = plt.subplots(dpi=300)
x_train=np.array([batch_range*i for i in range(len(train_accs))])
y_train=np.array(train_accs)
ax.plot(x_train,y_train, '#607B8B')

x_val=np.array([batch_range*i for i in range(len(val_accs))])
y_val=np.array(val_accs)
ax.plot(x_val,y_val, '#5D478B')

ax.set(xlabel='Steps', ylabel ='Accuracy', title='Music-Transformer '+dataset_name+': Training-Validation Accuracy')
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.legend(['Training Accuracy','Validation Accuracy'], loc='lower right')
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)
fig.savefig(train_out_dir+'music-transformer_'+dataset_name+'_train,val-acc_'+str(e*batch_range) + '_steps.png')
plt.close()