#!/usr/bin/env python3
"""Training modello YOLOv8n per rilevamento cialde caffe.

Uso:
    python scripts/train_yolo.py --data dataset/cialde.yaml --epochs 100

Il dataset deve essere in formato YOLO con struttura:
    dataset/
        images/
            train/
            val/
        labels/
            train/
            val/
        cialde.yaml
"""

import argparse
import os


def create_dataset_yaml(data_dir: str, output_path: str):
    """Crea il file YAML per il dataset."""
    yaml_content = f"""
path: {os.path.abspath(data_dir)}
train: images/train
val: images/val

names:
  0: coffee_pod
"""
    with open(output_path, 'w') as f:
        f.write(yaml_content.strip())
    print(f"Dataset YAML creato: {output_path}")


def train(args):
    """Avvia il training YOLOv8n."""
    from ultralytics import YOLO

    # Modello base
    model = YOLO('yolov8n.pt')

    # Training
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project='runs/detect',
        name='cialde',
        patience=20,
        save=True,
        plots=True,
    )

    # Esporta in TensorRT per Jetson
    if args.export_tensorrt:
        print("Esportazione in TensorRT FP16...")
        best_model = YOLO(os.path.join(results.save_dir, 'weights', 'best.pt'))
        best_model.export(format='engine', half=True, imgsz=args.imgsz)
        print("Export TensorRT completato!")

    # Copia il modello nella directory config
    import shutil
    best_pt = os.path.join(results.save_dir, 'weights', 'best.pt')
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
    dest = os.path.join(config_dir, 'yolo_cialde.pt')
    shutil.copy2(best_pt, dest)
    print(f"Modello copiato in: {dest}")


def main():
    parser = argparse.ArgumentParser(description='Training YOLOv8n cialde caffe')
    parser.add_argument('--data', type=str, required=True,
                        help='Percorso al file YAML del dataset')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default='0',
                        help='Device: 0 per GPU, cpu per CPU')
    parser.add_argument('--export-tensorrt', action='store_true',
                        help='Esporta in TensorRT FP16 dopo il training')
    parser.add_argument('--create-yaml', type=str, default=None,
                        help='Crea dataset YAML dalla directory specificata')

    args = parser.parse_args()

    if args.create_yaml:
        create_dataset_yaml(args.create_yaml, args.data)
        return

    train(args)


if __name__ == '__main__':
    main()
