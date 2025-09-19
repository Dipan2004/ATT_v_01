"""
Diagnostic Test - Debug Face Verification Pipeline Issues

This script will help identify why the pipeline always returns similarity = 1.0
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# Import our modules
from ml_module.detector import detect_and_preprocess
from ml_module.embeddings import get_embedding
from ml_module.verify import verify

def analyze_face_preprocessing(image_path):
    """Analyze face preprocessing step by step."""
    print(f"\n🔍 ANALYZING: {image_path}")
    
    # Load original image
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Could not load image")
        return None
        
    print(f"   📸 Original image shape: {image.shape}")
    print(f"   📊 Original image stats: min={image.min()}, max={image.max()}, mean={image.mean():.2f}")
    
    # Process face
    face = detect_and_preprocess(image_path)
    if face is None:
        print("❌ No face detected")
        return None
        
    print(f"   👤 Processed face shape: {face.shape}")
    print(f"   📊 Processed face stats: min={face.min():.4f}, max={face.max():.4f}, mean={face.mean():.4f}")
    
    return face

def analyze_embeddings(face1, face2, label1="Face1", label2="Face2"):
    """Analyze embedding generation."""
    print(f"\n🧠 EMBEDDING ANALYSIS")
    
    # Generate embeddings
    emb1 = get_embedding(face1)
    emb2 = get_embedding(face2)
    
    if emb1 is None or emb2 is None:
        print("❌ Embedding generation failed")
        return None, None
        
    print(f"   {label1} embedding stats:")
    print(f"      Shape: {emb1.shape}")
    print(f"      L2 norm: {np.linalg.norm(emb1):.6f}")
    print(f"      Min/Max: {emb1.min():.6f} / {emb1.max():.6f}")
    print(f"      Mean/Std: {emb1.mean():.6f} / {emb1.std():.6f}")
    
    print(f"   {label2} embedding stats:")
    print(f"      Shape: {emb2.shape}")
    print(f"      L2 norm: {np.linalg.norm(emb2):.6f}")
    print(f"      Min/Max: {emb2.min():.6f} / {emb2.max():.6f}")
    print(f"      Mean/Std: {emb2.mean():.6f} / {emb2.std():.6f}")
    
    # Check if embeddings are identical
    if np.allclose(emb1, emb2, atol=1e-6):
        print("   ⚠️  WARNING: Embeddings are nearly identical!")
        print("   🔍 Checking specific differences...")
        diff = np.abs(emb1 - emb2)
        print(f"      Max difference: {diff.max():.10f}")
        print(f"      Mean difference: {diff.mean():.10f}")
        print(f"      Non-zero differences: {np.count_nonzero(diff > 1e-10)}")
    
    return emb1, emb2

def detailed_similarity_analysis(emb1, emb2):
    """Detailed similarity analysis."""
    print(f"\n🔐 SIMILARITY ANALYSIS")
    
    # Cosine similarity (manual calculation)
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    manual_cosine = dot_product / (norm1 * norm2)
    
    print(f"   📊 Manual calculations:")
    print(f"      Dot product: {dot_product:.10f}")
    print(f"      Norm1: {norm1:.10f}")
    print(f"      Norm2: {norm2:.10f}")
    print(f"      Manual cosine: {manual_cosine:.10f}")
    
    # Using our verify function
    result = verify(emb1, emb2, threshold=0.7, method='cosine')
    print(f"   🔧 Pipeline result: {result}")
    
    # Euclidean distance
    euclidean_dist = np.linalg.norm(emb1 - emb2)
    print(f"   📏 Euclidean distance: {euclidean_dist:.10f}")
    
    return result

def test_with_identical_faces():
    """Test with identical face to verify pipeline works."""
    print(f"\n🧪 TESTING WITH IDENTICAL FACES")
    
    # Create a dummy face
    dummy_face = np.random.randn(160, 160, 3).astype('float32')
    dummy_face = (dummy_face - 127.5) / 127.5
    
    # Generate embeddings for the same face
    emb1 = get_embedding(dummy_face)
    emb2 = get_embedding(dummy_face)  # Same face
    
    if emb1 is not None and emb2 is not None:
        similarity = np.dot(emb1, emb2)
        print(f"   ✅ Identical faces similarity: {similarity:.10f}")
        
        # Now test with different random face
        dummy_face2 = np.random.randn(160, 160, 3).astype('float32')
        dummy_face2 = (dummy_face2 - 127.5) / 127.5
        emb3 = get_embedding(dummy_face2)
        
        if emb3 is not None:
            similarity2 = np.dot(emb1, emb3)
            print(f"   📊 Different faces similarity: {similarity2:.10f}")
    
def main():
    """Main diagnostic function."""
    print("🚨 FACE VERIFICATION DIAGNOSTIC TEST")
    print("=" * 60)
    
    # Find test images
    images_dir = Path("images")
    images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    if len(images) < 2:
        print("❌ Need at least 2 images for testing")
        return
        
    # Test with identical faces first
    test_with_identical_faces()
    
    # Analyze preprocessing for both images
    face1 = analyze_face_preprocessing(str(images[0]))
    face2 = analyze_face_preprocessing(str(images[1]))
    
    if face1 is None or face2 is None:
        print("❌ Face detection failed")
        return
        
    # Analyze embeddings
    emb1, emb2 = analyze_embeddings(face1, face2, 
                                   label1=f"Image1({images[0].name})", 
                                   label2=f"Image2({images[1].name})")
    
    if emb1 is None or emb2 is None:
        print("❌ Embedding generation failed")
        return
        
    # Detailed similarity analysis
    result = detailed_similarity_analysis(emb1, emb2)
    
    print("\n" + "=" * 60)
    print("🔍 DIAGNOSIS COMPLETE")
    
    # Final diagnosis
    if np.allclose(emb1, emb2, atol=1e-4):
        print("❌ PROBLEM IDENTIFIED: Embeddings are too similar!")
        print("   Possible causes:")
        print("   1. Face preprocessing is corrupting images")
        print("   2. FaceNet model is not working correctly")
        print("   3. Images are being processed identically")
        print("\n💡 SUGGESTED FIXES:")
        print("   1. Try different face detection method")
        print("   2. Check FaceNet model installation")
        print("   3. Verify image preprocessing pipeline")
    else:
        print("✅ Embeddings look different - similarity calculation might be the issue")

if __name__ == "__main__":
    main()