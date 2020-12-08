import config

from transformer import VanillaTransformer
import loss_functions

from data_loader import DataLoader
import utils
import datetime

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter


# check cuda
if torch.cuda.is_available():
    config.device = torch.device('cuda')
else:
    config.device = torch.device('cpu')
print('| Summary - Device Info : {}'.format(torch.cuda.device))

# load data
dataset = DataLoader(config.pickle_dir+"high/", config.pickle_dir+"low/")


# define model
model = VanillaTransformer(
                d_model = 256, 
                nhead = 4, 
                num_encoder_layers = 2,
                num_decoder_layers = 2, 
                dim_feedforward = config.vocab_size, #?
                dropout = 0.1,
                activation = "relu",
                source_vocab_length = 388,
                target_vocab_length = 391
)
model.to(config.device)
print(model)


# define optimizer and criterion
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
criterion = loss_functions.TransformerLoss()
# criterion = loss_functions.SmoothCrossEntropyLoss(config.label_smooth, config.vocab_size, config.pad_token)


# define tensorboard writer
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = 'logs/'+config.experiment+'/'+current_time+'/train'
eval_log_dir = 'logs/'+config.experiment+'/'+current_time+'/eval'
train_summary_writer = SummaryWriter(train_log_dir)
eval_summary_writer = SummaryWriter(eval_log_dir)


# Train Start
print(">> Train start...")
idx = 0
for e in range(config.epochs):
    print(">>> [Epoch was updated]")
    for b in range(len(dataset.X) // config.batch_size):
        model.train()

        src, trg = dataset.batch(config.batch_size, config.max_seq, 'train')

        src = torch.from_numpy(src).contiguous().to(config.device, non_blocking=True, dtype=torch.int)
        # l = max_seq+2
        trg = torch.from_numpy(trg).contiguous().to(config.device, non_blocking=True, dtype=torch.int)

        #change to shape (bs , max_seq_len+1) , Since right shifted
        trg_input = trg[:, :-1]
        #print(trg_input.size(0))
        targets = trg[:, 1:]

        src_mask = (src != 0)
        src_mask = src_mask.float().masked_fill(src_mask == 0, float('-inf')).masked_fill(src_mask == 1, float(0.0))

        trg_mask = (trg_input != 0)
        trg_mask = trg_mask.float().masked_fill(trg_mask == 0, float('-inf')).masked_fill(trg_mask == 1, float(0.0))

        size = trg_input.shape[1]
        
        #print(size)
        np_mask = torch.triu(torch.ones(size, size)==1).transpose(0,1)
        np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
        np_mask = np_mask.to(config.device)

        preds = model(src, trg_input, 
                        tgt_mask = np_mask, 
                        src_mask = src_mask, 
                        tgt_key_padding_mask=trg_mask)

        preds = preds.transpose(0,1).contiguous().view(-1, preds.size(-1))
        train_loss = criterion(preds, targets)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_summary_writer.add_scalar('loss', train_loss, global_step=idx)


        if b % 100 == 0:
            model.eval()
            eval_x, eval_y = dataset.batch(2, config.max_seq, 'eval')
            eval_x = torch.from_numpy(eval_x).contiguous().to(config.device, dtype=torch.int)
            eval_y = torch.from_numpy(eval_y).contiguous().to(config.device, dtype=torch.int)
            eval_preiction = model.forward(eval_x)
            eval_loss = criterion(eval_preiction, eval_y)
            torch.save(model.state_dict(), config.model_dir+'/train-{}.pth'.format(e))

            if b == 0:
                train_summary_writer.add_histogram("target_analysis", batch_y, global_step=e)
                train_summary_writer.add_histogram("source_analysis", batch_x, global_step=e)


            eval_summary_writer.add_scalar('loss', eval_loss, global_step=idx)

            print('\n====================================================')
            print('Epoch:{}/{}'.format(e, config.epochs))
            print('Batch: {}/{}'.format(b, len(dataset.X) // config.batch_size))
            print('Train >>>> Loss: {:6.6}'.format(train_loss))
            print('Eval >>>> Loss: {:6.6}'.format(eval_loss))
            
        torch.cuda.empty_cache()
        idx += 1


torch.save(model.state_dict(), config.model_dir+'/final.pth'.format(idx))
eval_summary_writer.close()
train_summary_writer.close()


