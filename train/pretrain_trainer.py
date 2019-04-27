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
        
#     def evaluate(self, args, model, dataloader, criterion):
#         ''' Evaluate model on the dev/test set '''
#         model.eval()
#         total_loss = 0
#         all_logits = None
#         all_labels = None
#         for step, batch in enumerate(tqdm(dataloader, desc=f"Validation")):
#             batch = tuple(t.to(args.device) for t in batch)
#             input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next = batch

#             # Forward
#             with torch.no_grad():
#                 logits = model(input_ids, segment_ids, input_mask)

#             # Calculate loss
#             labels = labels.view(-1)
#             logits = logits.view(-1) if args.output_mode == 'regression' else logits.view(-1, args.num_labels)
#             loss = criterion(logits, labels)

#             # Statistics
#             total_loss += loss.mean().item()
#             logits = logits.detach().cpu().numpy()
#             labels = labels.detach().cpu().numpy()
#             all_logits = logits if all_logits is None else np.append(all_logits, logits, axis=0)
#             all_labels = labels if all_labels is None else np.append(all_labels, labels, axis=0)

#         # Calculate prediction metrics (i.e. accuracy) and log
#         average_loss = total_loss / len(dataloader.dataset)
#         predictions = np.squeeze(all_logits) if args.output_mode == 'regression' else np.argmax(all_logits, axis=1)
#         result = compute_metrics(args.task_name, predictions, all_labels, 
#             logits=all_logits / all_logits.sum(axis=1, keepdims=True))

#         # Log
#         if self.logger:
#             self.logger.add_scalar('val/val_loss', average_loss, self.global_step)
#             self.logger.info(f"Val loss: {average_loss:.3f}")
#             for key in sorted(result.keys()):
#                 self.logger.add_scalar(f'val/{key}', result[key], self.global_step)
#                 self.logger.info(f"Val {key}: {result[key]:.3f}")
#         return average_loss, result

    def train(self, args, model, dataloader, optimizer, epoch):
        '''Train for a single epoch on a training dataset'''
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(dataloader, desc=f"[Epoch {epoch+1:3d}] Batch ")):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next = batch

            # Forward 
            logits_lm, logits_sc = model(input_ids, segment_ids, input_mask, masked_pos)

            # Masked LM and sequence classification losses
            loss_lm = F.cross_entropy(logits_lm.transpose(1, 2), masked_ids, reduction='none')
            loss_lm = (loss_lm * masked_weights.float()).mean()
            loss_sc = F.cross_entropy(logits_sc, is_next)
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
                
    def save(self, args, model, name=None):
        ''' Save a trained model and the associated configuration '''
        model_name = f"model-{self.global_step}.pth" if name is None else name 
        model_to_save = model.module if hasattr(model, 'module') else model # for nn.DataParallel
        model_file = os.path.join(args.output_dir, model_name)
        torch.save(model_to_save.state_dict(), model_file)
        return model_file
    
    