import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

class HelmetDataset(Dataset):
    """
    Helmet/No-Helmet dataset için PyTorch Dataset.
    """
    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir: Dataset ana klasörü
            split: 'train', 'val', veya 'test'
            transform: Torchvision transforms
        """
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.samples = []
        self.labels = []
        
        helmet_dir = self.data_dir / 'helmet'
        no_helmet_dir = self.data_dir / 'no_helmet'
        
        for img_path in helmet_dir.glob('*.png'):
            self.samples.append(img_path)
            self.labels.append(1)
        for img_path in helmet_dir.glob('*.jpg'):
            self.samples.append(img_path)
            self.labels.append(1)
        
        for img_path in no_helmet_dir.glob('*.png'):
            self.samples.append(img_path)
            self.labels.append(0)
        for img_path in no_helmet_dir.glob('*.jpg'):
            self.samples.append(img_path)
            self.labels.append(0)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class LightweightCNN(nn.Module):
    """
    Hafif CNN modeli - Helmet/No-Helmet classification için.
    """
    def __init__(self, num_classes=2):
        super(LightweightCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CNNClassifier:
    """
    CNN-based helmet classifier.
    """
    def __init__(self, device=None):
        if device is None:
            try:
                if torch.cuda.is_available():
                    self.device = torch.device('cuda')
                else:
                    self.device = torch.device('cpu')
            except Exception as e:
                print(f"⚠️  CUDA error: {e}, falling back to CPU")
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        print(f"🖥️  Device: {self.device}")
        
        self.model = LightweightCNN(num_classes=2).to(self.device)
        
        self.train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def prepare_data(self, dataset_dir, batch_size=32):
        """
        DataLoader'ları hazırlar.
        """
        print("📊 Dataset yükleniyor...")
        
        train_dataset = HelmetDataset(dataset_dir, split='train', transform=self.train_transform)
        val_dataset = HelmetDataset(dataset_dir, split='val', transform=self.test_transform)
        test_dataset = HelmetDataset(dataset_dir, split='test', transform=self.test_transform)
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        print(f"✓ Train: {len(train_dataset)} samples")
        print(f"✓ Val: {len(val_dataset)} samples")
        print(f"✓ Test: {len(test_dataset)} samples")
    
    def train(self, num_epochs=20, learning_rate=0.001):
        """
        Modeli eğitir.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        
        best_val_acc = 0.0
        
        print(f"\n🎯 CNN modeli eğitiliyor...")
        print(f"Epochs: {num_epochs}, Learning Rate: {learning_rate}")
        print("="*60)
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            train_loss = train_loss / len(self.train_loader)
            train_acc = 100. * train_correct / train_total
            
            val_loss, val_acc = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save('cnn_classifier_best.pth')
                print(f"  ✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
            
            print("-"*60)
        
        print(f"\n✅ Training tamamlandı! Best Val Acc: {best_val_acc:.2f}%")
        self.plot_training_history()
    
    def validate(self):
        """
        Validation seti üzerinde değerlendirme yapar.
        """
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(self.val_loader)
        val_acc = 100. * val_correct / val_total
        
        return val_loss, val_acc
    
    def evaluate(self):
        """
        Test seti üzerinde değerlendirme yapar.
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        print("\n📊 Test seti değerlendiriliyor...")
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader):
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        print("\n" + "="*60)
        print("📊 TEST SONUÇLARI")
        print("="*60)
        
        acc = accuracy_score(all_labels, all_preds)
        print(f"\nAccuracy: {acc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=['no_helmet', 'helmet']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(all_labels, all_preds)
        print(cm)
        
        self.plot_confusion_matrix(cm)
        
        return {
            'accuracy': acc,
            'predictions': all_preds,
            'probabilities': all_probs,
            'confusion_matrix': cm
        }
    
    def plot_training_history(self):
        """
        Training history görselleştirir.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(epochs, self.train_accs, 'b-', label='Train Acc')
        ax2.plot(epochs, self.val_accs, 'r-', label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('cnn_classifier_training_history.png', dpi=150)
        print(f"\n✓ Training history kaydedildi: cnn_classifier_training_history.png")
        plt.close()
    
    def plot_confusion_matrix(self, cm):
        """
        Confusion matrix görselleştirir.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['no_helmet', 'helmet'],
                    yticklabels=['no_helmet', 'helmet'])
        plt.title('CNN Classifier - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('cnn_classifier_confusion_matrix.png', dpi=150)
        print(f"✓ Confusion matrix kaydedildi: cnn_classifier_confusion_matrix.png")
        plt.close()
    
    def predict(self, image_path, return_proba=False):
        """
        Tek bir görüntü için tahmin yapar.
        """
        self.model.eval()
        
        image = Image.open(image_path).convert('RGB')
        image = self.test_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image)
            probs = torch.softmax(output, dim=1)[0]
            pred = output.argmax(1).item()
        
        if return_proba:
            return {
                'prediction': pred,
                'confidence': float(probs[pred]),
                'probabilities': {'no_helmet': float(probs[0]), 'helmet': float(probs[1])}
            }
        else:
            return pred
    
    def save(self, filepath):
        """
        Modeli kaydeder.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }, filepath)
    
    def load(self, filepath):
        """
        Modeli yükler.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accs = checkpoint.get('train_accs', [])
        self.val_accs = checkpoint.get('val_accs', [])
        print(f"✓ Model yüklendi: {filepath}")

def main():
    dataset_dir = '/home/berhan/Development/personal/HelmetClassCorrector/dataset'
    
    print("="*60)
    print("CNN CLASSIFIER TRAINING")
    print("="*60)
    print("Architecture: Lightweight CNN (3 Conv layers)")
    print("Input Size: 64x64")
    print("="*60)
    
    classifier = CNNClassifier()
    
    classifier.prepare_data(dataset_dir, batch_size=32)
    
    classifier.train(num_epochs=20, learning_rate=0.001)
    
    classifier.load('cnn_classifier_best.pth')
    results = classifier.evaluate()
    
    print("\n" + "="*60)
    print("✅ TRAINING TAMAMLANDI")
    print("="*60)
    print(f"Final Test Accuracy: {results['accuracy']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
