"""
    python test.py --model_path logs/convnext_tiny_epochs16_frFalse_prTrue/model_fold_1.pth --test_dir ../autodl-tmp/data/test --output_file submission_convnext_tiny.csv --model_name convnext_tiny --batch_size 32 --use_tta
"""

import os
import torch
import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import create_model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tta_transforms = [
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: transforms.functional.hflip(x)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: transforms.functional.vflip(x)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: transforms.functional.hflip(transforms.functional.vflip(x))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: transforms.functional.rotate(x, 90)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: transforms.functional.rotate(x, 180)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: transforms.functional.rotate(x, 270)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: transforms.functional.adjust_gamma(x, gamma=1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: transforms.functional.adjust_gamma(x, gamma=0.8)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: transforms.functional.adjust_contrast(x, contrast_factor=1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: transforms.functional.adjust_contrast(x, contrast_factor=0.8)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: transforms.functional.adjust_saturation(x, saturation_factor=1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: transforms.functional.adjust_saturation(x, saturation_factor=0.8)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: transforms.functional.adjust_hue(x, hue_factor=0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: transforms.functional.adjust_hue(x, hue_factor=-0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
]

class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None, use_tta=False):
        self.image_dir = image_dir
        self.transform = transform
        self.use_tta = use_tta
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
        print(f"找到 {len(self.image_filenames)} 个测试图像")

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, filename)
        
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        filename_without_ext = filename.replace('.tif', '')
        
        if self.use_tta:
            return image, filename_without_ext
        else:
            if self.transform:
                image = self.transform(image)
            return image, filename_without_ext

def custom_collate_fn(batch):
    images, filenames = zip(*batch)
    
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images, 0)
    else:
        images = list(images)
    
    return images, list(filenames)

def safe_load_model(model_path, model_name, device):
    try:
        model = create_model(num_classes=1, model_name=model_name, freeze_encoder=False, pretrained=True)
        
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v 
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=True)
        
        return model
        
    except Exception as e:
        print(f"模型加载失败: {str(e)}")

def predict_test_set(model_path, test_dir, output_file, model_name='resnet34', batch_size=32, sample_submission_path='sample_submission.csv', use_tta=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    print(f"加载模型: {model_path}")
    model = safe_load_model(model_path, model_name, device)
    model = model.to(device)
    model.eval()
    print("模型加载完成")
    
    predictions = {}
    tta_counts = {}
    
    if use_tta:
        print(f"开始TTA预测... 共{len(tta_transforms)}个变换")
        
        for tta_idx, tta_transform in enumerate(tta_transforms):
            
            print(f"正在应用TTA变换 {tta_idx + 1}/{len(tta_transforms)}")
            
            test_dataset = TestDataset(test_dir, transform=tta_transform, use_tta=False)
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            with torch.no_grad():
                pbar = tqdm(test_loader, desc=f"TTA变换 {tta_idx + 1}")
                for images, filenames in pbar:
                    try:
                        images = images.to(device)
                        outputs = model(images)
                        probs = torch.sigmoid(outputs)
                        
                        for filename, prob in zip(filenames, probs.cpu().numpy()):
                            prob_value = prob[0]
                            
                            if filename not in predictions:
                                predictions[filename] = prob_value
                                tta_counts[filename] = 1
                            else:
                                current_count = tta_counts[filename]
                                current_avg = predictions[filename]
                                

                                new_avg = (current_avg * current_count + prob_value) / (current_count + 1)
                                
                                predictions[filename] = new_avg
                                tta_counts[filename] = current_count + 1
                                
                    except Exception as e:
                        print(f"TTA变换 {tta_idx + 1} 预测时发生错误: {str(e)}")
                        continue
    
    else:
        test_dataset = TestDataset(test_dir, transform=transform, use_tta=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print("开始标准预测...")
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="正在预测测试集")
            for images, filenames in pbar:
                try:
                    images = images.to(device)
                    outputs = model(images)
                    probs = torch.sigmoid(outputs)
                    
                    for filename, pred in zip(filenames, probs.cpu().numpy()):
                        predictions[filename] = pred[0]
                        
                except Exception as e:
                    print(f"预测时发生错误: {str(e)}")
                    continue
    
    submission = pd.DataFrame({
        'id': list(predictions.keys()),
        'label': list(predictions.values())
    })
    
    sample_df = pd.read_csv(sample_submission_path)
    submission = submission.set_index('id').reindex(sample_df['id']).reset_index()

    # 保存CSV
    submission.to_csv(output_file, index=False)
    
    if use_tta:
        print(f"TTA预测完成! 每个样本平均使用了 {np.mean(list(tta_counts.values())):.1f} 个变换")
    
    print(f"预测完成! 结果已保存到: {output_file}")
    print(f"预测统计: 总数={len(predictions)}, 正例={submission['label'].sum():.1f}, 负例={len(submission) - submission['label'].sum():.1f}")
    
    print("\n前5行预测结果:")
    print(submission.head())
    
    return submission

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="预测测试集并生成提交文件")
    parser.add_argument('--model_path', type=str, default='logs/resnet18_epochs8_frFalse_prTrue/model_fold_1.pth', 
                        help='训练好的模型路径')
    parser.add_argument('--test_dir', type=str, default='../../autodl-tmp/data/test', 
                        help='测试数据目录')
    parser.add_argument('--output_file', type=str, default='submission_resnet18.csv', 
                        help='输出的提交文件路径')
    parser.add_argument('--model_name', type=str, default='resnet18', 
                        help='模型名称')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='批量大小')
    parser.add_argument('--use_tta', action='store_true', 
                        help='是否使用测试时增强 (TTA)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        exit(1)
    
    if not os.path.exists(args.test_dir):
        print(f"错误: 测试数据目录不存在: {args.test_dir}")
        exit(1)
    
    predict_test_set(
        args.model_path,
        args.test_dir,
        args.output_file,
        args.model_name,
        args.batch_size,
        use_tta=args.use_tta
    )

