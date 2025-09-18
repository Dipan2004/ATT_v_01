

import logging
import sys
import cv2
import numpy as np
from pathlib import Path

# Import modules
try:
    import face_recognition
    from ml_module.detector import detect_and_preprocess
    FACE_RECOGNITION_AVAILABLE = True
except ImportError as e:
    print(f"❌ Import Error: {e}")
    FACE_RECOGNITION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InteractiveFaceVerification:
    """Interactive face verification with multiple input methods."""
    
    def __init__(self, threshold=0.6):
        self.threshold = threshold
        if not FACE_RECOGNITION_AVAILABLE:
            print("❌ face_recognition library not available!")
            sys.exit(1)
    
    def verify_two_images(self, image1_path, image2_path):
        """Verify two specific image files."""
        print(f"\n🔍 VERIFYING TWO IMAGES:")
        print(f"   Image 1: {image1_path}")
        print(f"   Image 2: {image2_path}")
        
        try:
            # Load images
            img1 = cv2.imread(image1_path)
            img2 = cv2.imread(image2_path)
            
            if img1 is None:
                print(f"❌ Could not load: {image1_path}")
                return None
            if img2 is None:
                print(f"❌ Could not load: {image2_path}")
                return None
            
            # Convert to RGB
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            
            # Get face encodings
            encodings1 = face_recognition.face_encodings(img1_rgb)
            encodings2 = face_recognition.face_encodings(img2_rgb)
            
            if len(encodings1) == 0:
                print(f"❌ No face found in: {image1_path}")
                return None
            if len(encodings2) == 0:
                print(f"❌ No face found in: {image2_path}")
                return None
            
            # Calculate distance
            distance = face_recognition.face_distance([encodings1[0]], encodings2[0])[0]
            is_match = distance <= self.threshold
            confidence = max(0, 1 - (distance / 1.2))
            
            result = {
                "image1": Path(image1_path).name,
                "image2": Path(image2_path).name,
                "distance": distance,
                "threshold": self.threshold,
                "match": is_match,
                "confidence": confidence
            }
            
            # Display results
            print(f"\n📊 RESULTS:")
            print(f"   Distance: {distance:.4f}")
            print(f"   Threshold: {self.threshold}")
            print(f"   Match: {'✅ YES' if is_match else '❌ NO'}")
            print(f"   Confidence: {confidence:.2%}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def list_available_images(self):
        """List all available images in the images directory."""
        images_dir = Path("images")
        if not images_dir.exists():
            print("❌ 'images/' directory not found")
            return []
        
        images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        if not images:
            print("❌ No images found in 'images/' directory")
            return []
        
        print(f"\n📁 AVAILABLE IMAGES ({len(images)} found):")
        for i, img in enumerate(images, 1):
            print(f"   {i}. {img.name}")
        
        return images
    
    def interactive_selection(self):
        """Interactive image selection menu."""
        images = self.list_available_images()
        if not images:
            return
        
        try:
            print(f"\n🎯 SELECT TWO IMAGES TO COMPARE:")
            
            # Select first image
            while True:
                choice1 = input(f"Enter number for first image (1-{len(images)}): ").strip()
                try:
                    idx1 = int(choice1) - 1
                    if 0 <= idx1 < len(images):
                        break
                    else:
                        print(f"❌ Please enter a number between 1 and {len(images)}")
                except ValueError:
                    print("❌ Please enter a valid number")
            
            # Select second image
            while True:
                choice2 = input(f"Enter number for second image (1-{len(images)}): ").strip()
                try:
                    idx2 = int(choice2) - 1
                    if 0 <= idx2 < len(images):
                        if idx2 != idx1:
                            break
                        else:
                            print("❌ Please select a different image")
                    else:
                        print(f"❌ Please enter a number between 1 and {len(images)}")
                except ValueError:
                    print("❌ Please enter a valid number")
            
            # Verify selected images
            img1_path = str(images[idx1])
            img2_path = str(images[idx2])
            
            return self.verify_two_images(img1_path, img2_path)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            return None
    
    def batch_compare_all(self):
        """Compare all possible pairs of images."""
        images = self.list_available_images()
        if len(images) < 2:
            print("❌ Need at least 2 images for comparison")
            return
        
        print(f"\n🔄 COMPARING ALL PAIRS ({len(images)} images):")
        results = []
        
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                img1, img2 = str(images[i]), str(images[j])
                print(f"\n{'='*50}")
                result = self.verify_two_images(img1, img2)
                if result:
                    results.append(result)
        
        # Summary
        if results:
            print(f"\n📋 BATCH SUMMARY:")
            matches = sum(1 for r in results if r['match'])
            print(f"   Total comparisons: {len(results)}")
            print(f"   Matches found: {matches}")
            print(f"   Non-matches: {len(results) - matches}")
            
            # Show best and worst matches
            sorted_results = sorted(results, key=lambda x: x['distance'])
            print(f"\n🏆 BEST MATCH (lowest distance):")
            best = sorted_results[0]
            print(f"   {best['image1']} vs {best['image2']}: {best['distance']:.4f}")
            
            print(f"\n📊 WORST MATCH (highest distance):")
            worst = sorted_results[-1]
            print(f"   {worst['image1']} vs {worst['image2']}: {worst['distance']:.4f}")
    
    def test_with_webcam(self):
        """Test with live webcam feed."""
        print("\n📹 WEBCAM TESTING (Press 'q' to quit)")
        
        # First, select a reference image
        images = self.list_available_images()
        if not images:
            print("❌ Need at least one reference image")
            return
        
        print("Select a reference image to compare against:")
        while True:
            try:
                choice = int(input(f"Enter image number (1-{len(images)}): ")) - 1
                if 0 <= choice < len(images):
                    ref_image_path = str(images[choice])
                    break
                else:
                    print(f"❌ Enter number between 1-{len(images)}")
            except ValueError:
                print("❌ Enter a valid number")
        
        # Load reference encoding
        ref_img = cv2.imread(ref_image_path)
        ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        ref_encodings = face_recognition.face_encodings(ref_img_rgb)
        
        if not ref_encodings:
            print(f"❌ No face found in reference image: {ref_image_path}")
            return
        
        ref_encoding = ref_encodings[0]
        print(f"✅ Reference loaded: {Path(ref_image_path).name}")
        
        # Start webcam
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ Could not open webcam")
                return
            
            print("📹 Webcam started. Press SPACE to capture and compare, 'q' to quit")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Display frame
                cv2.putText(frame, "Press SPACE to compare, 'q' to quit", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Face Verification - Live Feed', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Spacebar pressed
                    # Compare current frame with reference
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    live_encodings = face_recognition.face_encodings(frame_rgb)
                    
                    if live_encodings:
                        distance = face_recognition.face_distance([ref_encoding], live_encodings[0])[0]
                        is_match = distance <= self.threshold
                        confidence = max(0, 1 - (distance / 1.2))
                        
                        print(f"\n📊 LIVE COMPARISON:")
                        print(f"   Distance: {distance:.4f}")
                        print(f"   Match: {'✅ YES' if is_match else '❌ NO'}")
                        print(f"   Confidence: {confidence:.2%}")
                    else:
                        print("❌ No face detected in current frame")
                
                elif key == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"❌ Webcam error: {e}")


def main_menu():
    """Main interactive menu."""
    verifier = InteractiveFaceVerification(threshold=0.6)
    
    while True:
        print("\n" + "="*60)
        print("🎯 INTERACTIVE FACE VERIFICATION PIPELINE")
        print("="*60)
        print("1. 📁 List available images")
        print("2. 🎯 Select two images to compare")
        print("3. 🔄 Compare all image pairs")
        print("4. 📹 Test with webcam (live)")
        print("5. ⚙️  Change threshold (current: {:.2f})".format(verifier.threshold))
        print("6. 🚪 Exit")
        print("="*60)
        
        try:
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == '1':
                verifier.list_available_images()
            
            elif choice == '2':
                verifier.interactive_selection()
            
            elif choice == '3':
                verifier.batch_compare_all()
            
            elif choice == '4':
                verifier.test_with_webcam()
            
            elif choice == '5':
                new_threshold = float(input("Enter new threshold (0.0-1.0): "))
                if 0.0 <= new_threshold <= 1.0:
                    verifier.threshold = new_threshold
                    print(f"✅ Threshold updated to: {new_threshold}")
                else:
                    print("❌ Please enter value between 0.0 and 1.0")
            
            elif choice == '6':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1-6")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    if not FACE_RECOGNITION_AVAILABLE:
        print("❌ Please install: pip install face_recognition")
        sys.exit(1)
    
    main_menu()