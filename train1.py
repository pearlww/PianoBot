import config

from music_transformer import MusicTransformer
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
model = MusicTransformer(
            embedding_dim=config.embedding_dim,
            vocab_size=config.vocab_size,
            num_layer=config.num_layers,
            max_seq= config.max_seq,
            dropout=config.dropout,
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
        batch_x, batch_y = dataset.batch(config.batch_size, config.max_seq, 'train') 
        # l = max_seq
        batch_x = torch.from_numpy(batch_x).contiguous().to(config.device, non_blocking=True, dtype=torch.int)
        # l = max_seq+2
        batch_y = torch.from_numpy(batch_y).contiguous().to(config.device, non_blocking=True, dtype=torch.int)

        # right shifted,  l = max_seq +1
        target_inputs = batch_y[:, :-1]
        targets = batch_y[:, 1:]

        # print(len(batch_x[0]))
        # print(len(batch_y[0]))
        # print(len(target_inputs[0]))
        # print(len(targets[0]))

        preds = model.forward(batch_x, target_inputs)
        
        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_summary_writer.add_scalar('loss', loss, global_step=idx)


        if b % 100 == 0:
            model.eval()
            eval_x, eval_y = dataset.batch(2, config.max_seq, 'eval')
            eval_x = torch.from_numpy(eval_x).contiguous().to(config.device, dtype=torch.int)
            eval_y = torch.from_numpy(eval_y).contiguous().to(config.device, dtype=torch.int)
            eval_preds = model.forward(eval_x)
            eval_loss = criterion(eval_preds, eval_y)
            torch.save(model.state_dict(), config.model_dir+'/train-{}.pth'.format(e))

            if b == 0:
                train_summary_writer.add_histogram("target_analysis", batch_y, global_step=e)
                train_summary_writer.add_histogram("source_analysis", batch_x, global_step=e)


            eval_summary_writer.add_scalar('loss', eval_loss, global_step=idx)

            print('\n====================================================')
            print('Epoch:{}/{}'.format(e, config.epochs))
            print('Batch: {}/{}'.format(b, len(dataset.X) // config.batch_size))
            print('Train >>>> Loss: {:6.6}'.format(loss))
            print('Eval >>>> Loss: {:6.6}'.format(eval_loss))
            
        torch.cuda.empty_cache()
        idx += 1


torch.save(model.state_dict(), config.model_dir+'/final.pth'.format(idx))
eval_summary_writer.close()
train_summary_writer.close()


