import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from spafe.features.lfcc import lfcc

############################################
# DEVICE
############################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

############################################
# WAV2VEC LOAD
############################################

print("Loading Wav2Vec2 model...")
wav_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
wav_model.eval()
wav_model.to(device)

def extract_wav2vec(waveform, sr=16000):
    """Extract Wav2Vec2 features from waveform"""
    inputs = wav_processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = wav_model(**inputs)
    
    features = outputs.last_hidden_state
    features = features.mean(dim=1).squeeze(0)
    
    return features

############################################
# LFCC EXTRACTION (USING SPAFE - MATCHES TRAINING)
############################################

def extract_lfcc(audio_path, num_ceps=20):
    """Extract LFCC features using spafe (matches training)"""
    try:
        # Load audio
        waveform, sr = sf.read(audio_path, dtype="float32")
        
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        
        wav_numpy = waveform
        
        # Extract LFCC using spafe (SAME AS TRAINING)
        feats = lfcc(
            wav_numpy,
            fs=sr,
            num_ceps=num_ceps,
            nfilts=40,
            nfft=512,
            low_freq=0,
            high_freq=sr//2
        )
        
        # Convert to tensor and transpose
        lfcc_tensor = torch.tensor(feats, dtype=torch.float32).T
        
        return lfcc_tensor, waveform, sr
    
    except Exception as e:
        print(f"Error extracting LFCC: {e}")
        return None, None, None

############################################
# MODEL
############################################

class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((1, 2))
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((1, 2))
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(128 + 768, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)
    
    def forward(self, lfcc, wav):
        x = lfcc.unsqueeze(1)
        
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.conv3(x)))
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        combined = torch.cat((x, wav), dim=1)
        
        out = self.dropout(torch.relu(self.fc1(combined)))
        return self.fc2(out)

############################################
# LOAD MODEL
############################################

print("Loading trained Fusion model...")
model = FusionModel().to(device)
model_path = "best_wav2vec_lfcc_model.pth"

if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found!")
    sys.exit(1)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✓ Model loaded successfully\n")

############################################
# MAIN TEST FUNCTION
############################################

def test_audio(audio_path):
    """Test a single audio file"""
    
    print(f"Testing audio file: {audio_path}")
    print("-" * 50)
    
    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio file not found at {audio_path}")
        return
    
    # Extract LFCC features
    print("Extracting LFCC features...")
    lfcc_feat, waveform, sr = extract_lfcc(audio_path)
    
    if lfcc_feat is None:
        print("❌ Failed to extract LFCC features")
        return
    
    # Extract Wav2Vec features
    print("Extracting Wav2Vec2 features...")
    wav_feat = extract_wav2vec(waveform, sr)
    
    # Prepare tensors
    lfcc_feat = lfcc_feat.unsqueeze(0).to(device)  # Add batch dimension
    wav_feat = wav_feat.unsqueeze(0).to(device)   # Add batch dimension
    
    # Make prediction
    print("Making prediction...")
    with torch.no_grad():
        logits = model(lfcc_feat, wav_feat)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    # Display results
    print("-" * 50)
    print("\n📊 RESULTS:")
    print("=" * 50)
    
    confidence = max(prob, 1 - prob) * 100
    
    if prob > 0.5:
        print(f"🚨 FAKE (SPOOF) DETECTED")
        print(f"   Spoof Confidence: {prob*100:.2f}%")
    else:
        print(f"✅ GENUINE (BONAFIDE)")
        print(f"   Bonafide Confidence: {(1-prob)*100:.2f}%")
    
    print("=" * 50)
    print(f"Raw Score: {prob:.4f}")
    print(f"Overall Confidence: {confidence:.2f}%\n")

############################################
# MAIN
############################################

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python testaudio.py <audio_file_path>")
        print("\nExample:")
        print("  python testaudio.py sample.wav")
        print("  python testaudio.py C:\\path\\to\\audio.flac")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    test_audio(audio_file)
