"""
python train.py --image_dir '../autodl-tmp/data/train' \
                --model_name resnet18 \
                --pretrained \
                --batch_size 64 \
                --num_epochs 8 \
                --k_folds 5 \
                --seed 42
"""

import warnings
warnings.filterwarnings("ignore")
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from engine import train_one_epoch, evaluate
from dataset import TumorDataset, CustomCollate
from model import create_model
import pandas as pd
import gc
import time
import numpy as np
from sklearn.model_selection import KFold
from torchvision import transforms
from distri_utils import setup, cleanup, launch_process, find_free_port, kill_child_processes
from logger import TrainingLogger
import psutil
import sys
import argparse
import random
from focal_loss import FocalLoss

def feed_seed(seed, rank):

    process_seed = seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(process_seed)
        torch.cuda.manual_seed_all(process_seed)  

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(rank, world_size, port,
                    image_dir='../autodl-tmp/data/train',
                    model_name='resnet34',
                    freeze_encoder=False, 
                    pretrained=True, 
                    batch_size=64,
                    num_epochs=16,
                    k_folds=5,
                    use_k_folds=False,
                    seed=3407,
                    resume=None
                    ):

    launch_process(rank)
    
    try:
        if hasattr(mp, 'get_context'):
            mp.get_context('spawn')
        
        time.sleep(rank * 2.0)  
        
        feed_seed(seed, rank)
        
        torch.cuda.empty_cache()
        
        setup(rank, world_size, port)
        device = torch.device(f"cuda:{rank}")
        
        batch_size = batch_size  
        num_epochs = num_epochs  
        image_dir = image_dir
        train_labels = "train_labels.csv"
        train_labels = pd.read_csv(train_labels)
        train_labels = {row['id'] + '.tif': row['label'] for _, row in train_labels.iterrows()}
        num_workers = 8   
        
        k_folds = k_folds

        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomVerticalFlip(), 
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),            
            transforms.RandomApply([
                transforms.RandomGrayscale(p=0.2),  
                transforms.RandomSolarize(threshold=128, p=0.2),  
                transforms.RandomPosterize(bits=4, p=0.2),  
                transforms.RandomInvert(p=0.2),  
                transforms.RandomEqualize(p=0.2),  
                transforms.RandomAutocontrast(p=0.2),  
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2)  
                ]),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5, fill=0),  
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.75, 1.333)),
            
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ])

        val_transform = transforms.Compose([
            # # transforms.CenterCrop(64),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        train_dataset = TumorDataset(image_dir, train_labels, transform=train_transform)

        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        indices = list(range(len(train_dataset)))
        fold_splits = list(kfold.split(indices))
        
        if rank == 0:
            log_dir = f'logs/{model_name}_epochs{num_epochs}_fr{freeze_encoder}_pr{pretrained}'
            os.makedirs(log_dir, exist_ok=True)
            logger = TrainingLogger(log_dir=log_dir)
        
        if rank == 0:
            print("计算类别权重...")
            label_counts = {}
            for _, label in train_labels.items():
                label_counts[label] = label_counts.get(label, 0) + 1
            
            total_samples = sum(label_counts.values())
            print(f"数据集分布: {label_counts}")
            print(f"总样本数: {total_samples}")
            
            class_weights = {}
            for label, count in label_counts.items():
                class_weights[label] = total_samples / (len(label_counts) * count)
            
            print(f"计算的类别权重: {class_weights}")
            
            pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32)
            print(f"正类权重 (pos_weight): {pos_weight.item():.4f}")
        else:
            pos_weight = torch.tensor([1.0], dtype=torch.float32)
        
        if dist.is_initialized():
            pos_weight = pos_weight.to(device)
            dist.broadcast(pos_weight, src=0)

        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception as e:
                print(f"Rank {rank}: 同步屏障超时 - {str(e)}")
                if dist.is_initialized():
                    cleanup()
                time.sleep(5)
                setup(rank, world_size, port)
        else:
            print(f"Rank {rank}: 进程组未初始化，跳过同步屏障")
            return

        fold_results = []
        
        for fold in range(k_folds):
                
            if rank == 0:
                print(f"\n开始训练折 {fold+1}/{k_folds}")
                
            train_indices, val_indices = fold_splits[fold]
            
            train_subset = torch.utils.data.Subset(train_dataset, train_indices)
            
            train_sampler = DistributedSampler(
                train_subset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False  
            )
            
            train_loader = torch.utils.data.DataLoader(
                train_subset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=num_workers,
                collate_fn=CustomCollate(),
                persistent_workers=False,  
            )
            
            val_loader = None
            if rank == 0:
                val_dataset = TumorDataset(image_dir, train_labels, transform=val_transform)
                val_subset = torch.utils.data.Subset(val_dataset, val_indices)
                val_loader = torch.utils.data.DataLoader(
                    val_subset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    collate_fn=CustomCollate(),
                    persistent_workers=False,  
                )
                print(f"训练集大小: {len(train_subset)}, 验证集大小: {len(val_subset)}")
            
            torch.cuda.synchronize(device)  
            dist.barrier()
            
            model = create_model(num_classes=1, model_name=model_name, freeze_encoder=freeze_encoder, pretrained=pretrained) 
            model = model.to(device)

            if resume:
                if os.path.isfile(resume):
                    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
                    state_dict = torch.load(resume, map_location=map_location)
                    model.load_state_dict(state_dict)
                
            
            model = DDP(
                model, 
                device_ids=[rank],   
            )
            
            # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
            criterion = FocalLoss(alpha=pos_weight.item(), gamma=2.0).to(device)

            optimizer = torch.optim.AdamW([
                {'params': model.module.encoder.parameters(), 'lr': 1e-4, 'weight_decay': 0.0001},
                {'params': model.module.predict_head.parameters(), 'lr': 1e-4, 'weight_decay': 0.0001}
            ])

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=len(train_loader) * num_epochs,
                eta_min=2e-5, 
                last_epoch=-1,  
            )
            
            if rank == 0:
                print(f"Fold {fold+1} - 模型和优化器初始化完成")
            
            best_val_auc = 0
            best_val_acc = 0
            
            for epoch in range(num_epochs):

                if epoch == 2 and freeze_encoder:
                    if rank == 0:
                        print("解冻编码器...")
                    model.module.unfreeze_encoder()
                
                train_sampler.set_epoch(epoch)
                
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, rank, device, scheduler=scheduler)
                
                if rank == 0:
                    val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)
                    logger.log_epoch(epoch+1, train_loss, train_acc, val_loss, val_acc, val_auc)
                    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}")
                    
                    if val_auc > best_val_auc:
                        best_val_auc = val_auc

                        torch.save(model.module.state_dict(), f"{log_dir}/model_fold_{fold+1}.pth")
                        print(f"saved to {log_dir}/model_fold_{fold+1}.pth: Fold {fold+1}, Epoch {epoch+1}, Val Acc: {val_acc:.2f}%")
            
            
            if hasattr(train_loader, '_iterator'):
                del train_loader._iterator
            del train_loader
                
            if rank == 0 and val_loader is not None:
                if hasattr(val_loader, '_iterator'):
                    del val_loader._iterator
                del val_loader
                
            if rank == 0:
                fold_results.append({
                    'fold': fold+1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'best_val_acc': best_val_acc
                })
                logger.log_fold(fold+1, train_loss, train_acc, val_loss, val_acc, val_auc)
            
            try:
                torch.cuda.synchronize(device)
                dist.barrier()
            except Exception as e:
                print(f"Rank {rank}: 清理前同步屏障失败 - {str(e)}")
            
            if 'model' in locals():
                model.to('cpu')
                del model
            
            if 'optimizer' in locals():
                del optimizer
            
            if 'scheduler' in locals():
                del scheduler
            
            if 'train_sampler' in locals():
                del train_sampler
            
            gc.collect()
            torch.cuda.empty_cache()
            
            kill_child_processes()
            

            try:
                torch.cuda.synchronize(device)
                dist.barrier()
            except Exception as e:
                print(f"Rank {rank}: 资源清理后同步屏障失败 - {str(e)}")
                break
            
            if rank == 0:
                print(f"完成第 {fold+1} 折训练，内存已清理")

            if not use_k_folds:
                break

        if rank == 0:
            try:
                cv_summary = logger.summarize_cv_results()
                print("\n交叉验证结果汇总:")
                for key, value in cv_summary.items():
                    print(f"{key}: {value:.4f}")
                
                logger.plot_metrics()
                
                best_fold = max(fold_results, key=lambda x: x['best_val_acc'])
                print(f"\n最佳模型来自第 {best_fold['fold']} 折，验证准确率: {best_fold['best_val_acc']:.4f}")
                
                # os.system(f"cp model_fold_{best_fold['fold']}.pth final_model.pth")
                print("已保存最终模型: final_model_s.pth")
            except Exception as e:
                print(f"汇总结果时出错: {str(e)}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"Rank {rank} 遇到错误: {e}")
        import traceback
        traceback.print_exc()
        if rank == 0 and 'logger' in locals():
            logger.log_error(e)
    
    finally:
        print(f"Rank {rank}清理资源...")
        for name in list(locals()):
            if name not in ['rank', 'dist', 'cleanup', 'port', 'world_size']:
                try:
                    del locals()[name]
                except:
                    pass
        
        gc.collect()
        torch.cuda.empty_cache()
        
        kill_child_processes()
        
        if dist.is_initialized():
            cleanup()
        
        print(f"Rank {rank}完成清理")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="分布式训练脚本")
    parser.add_argument('--image_dir', type=str, default='../autodl-tmp/data/train', help='图像数据目录')
    parser.add_argument('--model_name', type=str, default='resnet18', help='模型名称')
    parser.add_argument('--freeze_encoder', action='store_true', default=False, help='是否冻结编码器')
    parser.add_argument('--pretrained', action='store_true', default=False, help='是否使用预训练模型')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=16, help='训练轮数')
    parser.add_argument('--k_folds', type=int, default=5, help='折数量')
    parser.add_argument('--use_k_folds', action='store_true', default=False, help='是否使用多折')
    parser.add_argument('--seed', type=int, default=3407, help='种子')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的模型路径')
    args = parser.parse_args()
    
    if hasattr(mp, 'set_start_method'):
        mp.set_start_method('spawn', force=True)
    
    world_size = min(2, torch.cuda.device_count())
    
    port = find_free_port()
    print(f"使用端口 {port} 进行分布式训练")
    
    if not torch.cuda.is_available():
        print("错误: CUDA不可用")
        sys.exit(1)
        
    if torch.cuda.device_count() < world_size:
        print(f"警告: 需要 {world_size} 个GPU，但只找到 {torch.cuda.device_count()} 个")
        world_size = torch.cuda.device_count()
    
    print(f"使用 {world_size} 个GPU进行训练，GPU列表:")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    显存: {torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024:.2f} GB")
    
    try:
        if 'MASTER_ADDR' in os.environ:
            del os.environ['MASTER_ADDR']
        if 'MASTER_PORT' in os.environ:
            del os.environ['MASTER_PORT']
        
        torch.cuda.empty_cache()
            
        mp.spawn(
            train_model,
            args=(world_size, 
                    port, 
                    args.image_dir, 
                    args.model_name, 
                    args.freeze_encoder, 
                    args.pretrained, 
                    args.batch_size, 
                    args.num_epochs, 
                    args.k_folds,
                    args.use_k_folds,
                    args.seed,
                    args.resume
                    ),  
            nprocs=world_size,
            join=True
        )

    except KeyboardInterrupt:
        parent = psutil.Process(os.getpid())
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except:
                pass
