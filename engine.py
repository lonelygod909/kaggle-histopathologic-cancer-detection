import torch
import torch.nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def train_one_epoch(model, dataloader, criterion, optimizer, rank, device, scheduler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"")
    for i, (images, labels) in enumerate(pbar):
        try:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels.float())
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            running_loss += loss.item()

            total += labels.size(0)
            correct += (predicted == labels.float()).sum().item()
            
            current_loss = running_loss / (i + 1)
            current_acc = 100 * correct / total
            pbar.set_postfix({
                'loss': f'{current_loss:.6f}',
                'acc': f'{current_acc:.2f}%'
            })
            
        
        except Exception as e:
            print(f"训练中发生错误: {str(e)}")
            continue
        
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds_probs = [] 
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"评估中")
        for i, (images, labels) in enumerate(pbar):
            try:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels.float())
                
                probs = torch.sigmoid(outputs)
                predicted_classes = torch.where(probs >= 0.5, 
                                      torch.ones_like(outputs, dtype=torch.float32), 
                                      torch.zeros_like(outputs, dtype=torch.float32))
            
                running_loss += loss.item()
                total += labels.size(0)
                correct += (predicted_classes == labels.float()).sum().item()
                
                all_preds_probs.extend(probs.cpu().numpy()) 
                all_labels.extend(labels.cpu().numpy())     
                
                current_loss = running_loss / (i + 1)
                current_acc = 100 * correct / total
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{current_acc:.2f}%'
                })
                
            except Exception as e:
                print(f"评估中发生错误: {str(e)}")
                continue
            
    epoch_loss = running_loss / len(dataloader) if len(dataloader) > 0 else float('inf')
    epoch_acc = 100 * correct / total if total > 0 else 0
    epoch_auc = 0.0

    if total > 0 and len(np.unique(all_labels)) > 1:
        epoch_auc = roc_auc_score(np.array(all_labels).flatten(), np.array(all_preds_probs).flatten())
   
    return epoch_loss, epoch_acc, epoch_auc 
    