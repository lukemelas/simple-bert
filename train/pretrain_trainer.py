import os 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm 

class Trainer():
    
    def __init__(self, logger=None):
        ''' The trainer simply holds the global training step and the logger. '''
        self.logger = logger
        self.global_step = 0
        
    def train(self, args, model, dataloader, optimizer, epoch):
        '''Train for a single epoch on a training dataset'''
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(dataloader, desc=f"[Epoch {epoch+1:3d}] Batch ")):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_not_next = batch

            # Forward 
            logits_lm, logits_sc = model(input_ids, segment_ids, input_mask, masked_pos)

            # Masked LM and sequence classification losses
            loss_lm = F.cross_entropy(logits_lm.transpose(1, 2), masked_ids, reduction='none')
            loss_lm = (loss_lm * masked_weights.float()).mean()
            loss_sc = F.cross_entropy(logits_sc, is_not_next)
            loss = loss_lm + loss_sc
            
            # Multi-gpu / gradient accumulation
            if args.n_gpu > 1: # note: use .mean() to average on multi-gpu 
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1: # accumulate gradient for small batch sizes
                loss = loss / args.gradient_accumulation_steps

            # Backward
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            total_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16: # modify l.r. with warmup (if args.fp16 is False, this is automatic)
                    lr_this_step = args.learning_rate * \
                        warmup_linear(model.global_step/num_train_optimization_steps, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                self.global_step += 1

            if self.logger:
                # TODO: log learning rate
                self.logger.add_scalar('train/loss_total', loss.item(), self.global_step)
                self.logger.add_scalar('train/loss_lm', loss_lm.item(), self.global_step)
                self.logger.add_scalar('train/loss_sc', loss_sc.item(), self.global_step)
                
            if (self.global_step + 1) % args.checkpoint_every == 0:
                self.save(args, model)

        if self.logger:
            self.logger.info(f'Train loss: {total_loss/len(dataloader.dataset):.3f}')
              
    def evaluate(self):
        # Validation is not implemented -- pretrain for as long as possible
        raise NotImplementedError()

    def save(self, args, model, name=None):
        ''' Save a trained model and the associated configuration '''
        model_name = f"model-{self.global_step}.pth" if name is None else name 
        model_to_save = model.module if hasattr(model, 'module') else model # for nn.DataParallel
        model_file = os.path.join(args.output_dir, model_name)
        torch.save(model_to_save.state_dict(), model_file)
        return model_file
    
    