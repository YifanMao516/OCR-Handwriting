#!/usr/bin/env python
"""
将IAM数据集从共享位置转换为LMDB格式
适配实际的IAM目录结构
"""

import os
import sys
import lmdb
import cv2
import numpy as np
from tqdm import tqdm
import argparse

def checkImageIsValid(imageBin):
    """检查图片是否有效"""
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True

def writeCache(env, cache):
    """写入缓存到LMDB"""
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def parse_iam_annotations(annotation_file):
    """解析IAM标注文件"""
    samples = []
    
    with open(annotation_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 跳过注释行
            if line.startswith('#'):
                continue
            
            parts = line.strip().split(' ')
            if len(parts) >= 9:
                # 获取图片ID
                image_id = parts[0]
                
                # 获取文本（从第9个字段开始）
                text = ' '.join(parts[8:])
                
                # 只保留英文字符、数字和空格
                # ABINet默认只支持36个字符（26字母+10数字）
                filtered_text = ''.join(c for c in text if c.isalnum() or c.isspace())
                
                if filtered_text.strip():  # 确保文本不为空
                    samples.append((image_id, filtered_text.strip()))
    
    return samples

def create_iam_lmdb(input_dir, output_dir, split='test', max_samples=None):
    """
    创建IAM的LMDB数据集
    
    Args:
        input_dir: IAM原始数据目录 (如 /root/autodl-tmp/ocr/datasets/iam)
        output_dir: LMDB输出目录
        split: 数据集划分 (test/train/val)
        max_samples: 最大样本数（用于测试）
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取标注文件（现在直接在根目录下）
    annotation_file = os.path.join(input_dir, 'lines.txt')
    if not os.path.exists(annotation_file):
        print(f"错误: 找不到标注文件 {annotation_file}")
        return False
    
    print(f"解析标注文件: {annotation_file}")
    all_samples = parse_iam_annotations(annotation_file)
    print(f"总共找到 {len(all_samples)} 个标注")
    
    # 根据split划分数据
    if split == 'test':
        # 使用最后500个样本作为测试集
        samples = all_samples[-500:]
    elif split == 'train':
        # 使用前80%作为训练集
        train_size = int(len(all_samples) * 0.8)
        samples = all_samples[:train_size]
    else:  # val
        # 使用中间部分作为验证集
        train_size = int(len(all_samples) * 0.8)
        samples = all_samples[train_size:-500]
    
    # 如果指定了最大样本数，截取
    if max_samples:
        samples = samples[:max_samples]
    
    print(f"使用 {len(samples)} 个样本作为 {split} 集")
    
    # 创建LMDB环境
    env = lmdb.open(output_dir, map_size=1099511627776)
    cache = {}
    cnt = 1
    valid_cnt = 0
    missing_images = []
    
    for image_id, text in tqdm(samples, desc=f"Creating {split} LMDB"):
        # 构建图片路径
        # IAM的实际路径结构: a01/a01-000u/a01-000u-00.png
        parts = image_id.split('-')
        if len(parts) >= 3:
            folder1 = parts[0]  # a01
            folder2 = '-'.join(parts[:3])  # a01-000u
            image_path = os.path.join(input_dir, folder1, folder2, f"{image_id}.png")
            
            # 如果第一种路径不存在，尝试另一种
            if not os.path.exists(image_path) and len(parts) >= 2:
                folder2 = '-'.join(parts[:2])  # a01-000
                image_path = os.path.join(input_dir, folder1, folder2, f"{image_id}.png")
            
            if os.path.exists(image_path):
                # 读取图片
                try:
                    with open(image_path, 'rb') as f:
                        imageBin = f.read()
                    
                    if checkImageIsValid(imageBin):
                        # 添加到缓存
                        imageKey = f'image-{cnt:09d}'.encode()
                        labelKey = f'label-{cnt:09d}'.encode()
                        cache[imageKey] = imageBin
                        cache[labelKey] = text.encode()
                        valid_cnt += 1
                        
                        if cnt % 1000 == 0:
                            writeCache(env, cache)
                            cache = {}
                            print(f"已处理 {cnt} 个样本，有效 {valid_cnt} 个")
                        
                        cnt += 1
                    else:
                        print(f"无效图片: {image_path}")
                except Exception as e:
                    print(f"读取图片失败 {image_path}: {e}")
            else:
                missing_images.append(image_id)
                if len(missing_images) <= 5:  # 只显示前5个缺失的
                    print(f"找不到图片: {image_path}")
    
    # 写入剩余缓存
    nSamples = valid_cnt
    cache[b'num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    env.close()
    
    print(f"\n✅ 成功创建 {split} LMDB!")
    print(f"   位置: {output_dir}")
    print(f"   样本数: {nSamples}")
    if missing_images:
        print(f"   缺失图片数: {len(missing_images)}")
    
    return True

def verify_lmdb(lmdb_path):
    """验证创建的LMDB数据集"""
    env = lmdb.open(lmdb_path, readonly=True)
    with env.begin() as txn:
        num_samples = txn.get(b'num-samples')
        if num_samples:
            print(f"LMDB包含 {num_samples.decode()} 个样本")
            
            # 显示前3个样本
            for i in range(1, min(4, int(num_samples.decode()) + 1)):
                label_key = f'label-{i:09d}'.encode()
                label = txn.get(label_key)
                if label:
                    print(f"  样本{i}: {label.decode()[:50]}...")
    env.close()

def main():
    parser = argparse.ArgumentParser(description='将IAM数据集转换为LMDB格式')
    parser.add_argument('--input_dir', type=str, 
                        default='/root/autodl-tmp/ocr/datasets/iam',
                        help='IAM原始数据目录')
    parser.add_argument('--output_base', type=str,
                        default='data/evaluation',
                        help='LMDB输出基础目录')
    parser.add_argument('--split', type=str, 
                        default='test',
                        choices=['test', 'train', 'val'],
                        help='数据集划分')
    parser.add_argument('--max_samples', type=int,
                        default=None,
                        help='最大样本数（用于快速测试）')
    parser.add_argument('--verify', action='store_true',
                        help='验证创建的LMDB')
    
    args = parser.parse_args()
    
    # 输出目录
    output_dir = os.path.join(args.output_base, f'IAM_{args.split}')
    
    print("=" * 60)
    print("IAM数据集 -> LMDB转换工具")
    print("=" * 60)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"数据划分: {args.split}")
    if args.max_samples:
        print(f"最大样本数: {args.max_samples}")
    print("=" * 60)
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"❌ 错误: 输入目录不存在: {args.input_dir}")
        print("请确保IAM数据集已经下载到指定位置")
        return
    
    # 检查关键文件
    lines_txt = os.path.join(args.input_dir, 'lines.txt')
    if not os.path.exists(lines_txt):
        print(f"❌ 错误: 找不到lines.txt文件: {lines_txt}")
        return
    
    # 执行转换
    success = create_iam_lmdb(args.input_dir, output_dir, args.split, args.max_samples)
    
    if success and args.verify:
        print("\n验证LMDB:")
        verify_lmdb(output_dir)
    
    if success:
        print("\n下一步:")
        print(f"1. 下载预训练模型到 workdir/ 目录")
        print(f"2. 运行测试: python main.py --config=configs/train_abinet.yaml --phase test --test_root {output_dir}")

if __name__ == '__main__':
    main()
