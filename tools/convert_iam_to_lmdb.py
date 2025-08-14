#!/usr/bin/env python3
import os
import lmdb
import cv2
from tqdm import tqdm
import argparse

def create_iam_lmdb(iam_path, words_file, output_path):
    with open(words_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    image_list = []
    for line in lines:
        if line.startswith('#'):
            continue
        parts = line.strip().split(' ')
        if len(parts) < 9:
            continue
            
        image_id = parts[0]  # a01-000u-00-00
        status = parts[1]
        if status != 'ok':
            continue
            
        # 提取路径: a01-000u-00-00 -> a01/a01-000u/a01-000u-00.png
        writer_id = image_id.split('-')[0]  # a01
        form_id = '-'.join(image_id.split('-')[:2])  # a01-000u
        line_id = '-'.join(image_id.split('-')[:3])  # a01-000u-00
        image_path = os.path.join(iam_path, writer_id, form_id, f"{line_id}.png")
        
        # 真实文本是最后的部分
        gt_text = ' '.join(parts[8:])
        
        if os.path.exists(image_path):
            image_list.append((image_path, gt_text))
        else:
            print(f"Image not found: {image_path}")
    
    print(f"Found {len(image_list)} valid images")
    
    if len(image_list) == 0:
        print("No images found! Check paths.")
        return
    
    # Create LMDB
    os.makedirs(output_path, exist_ok=True)
    env = lmdb.open(output_path, map_size=1099511627776)
    
    with env.begin(write=True) as txn:
        for idx, (img_path, gt_text) in enumerate(tqdm(image_list)):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            _, img_encoded = cv2.imencode('.png', img)
            img_bytes = img_encoded.tobytes()
            
            img_key = f'image-{idx:09d}'.encode()
            label_key = f'label-{idx:09d}'.encode()
            
            txn.put(img_key, img_bytes)
            txn.put(label_key, gt_text.encode())
        
        txn.put(b'num-samples', str(len(image_list)).encode())
    
    env.close()
    print(f"LMDB dataset created at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--iam_path', required=True)
    parser.add_argument('--words_file', required=True)
    parser.add_argument('--output_path', required=True)
    
    args = parser.parse_args()
    create_iam_lmdb(args.iam_path, args.words_file, args.output_path)
