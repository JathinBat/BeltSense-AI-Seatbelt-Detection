"""
Unified Seatbelt Detection Training Script
Consolidates all training methods into one interactive interface.

Available Training Methods:
1. Dynamic Validation (Original) - Uses different validation sample each epoch
2. Fixed Validation (Filtered) - Uses preserved filtered_database_old dataset
3. Simple Dynamic - Simplified dynamic validation with validation set changes
4. Legacy Fixed Split - Traditional 80/20 split for comparison

All methods integrate with model_manager for latest model publishing.
"""

import os
import shutil
import random
import yaml
from pathlib import Path
import sys


def clear_screen():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def display_menu():
    """Display the training method selection menu"""
    clear_screen()
    print("=" * 70)
    print("🚗 UNIFIED SEATBELT DETECTION TRAINING")
    print("=" * 70)
    print()
    print("Available Training Methods:")
    print()
    print("1. 🔄 Dynamic Validation (Original)")
    print("   - Uses different validation sample each epoch")
    print("   - Most advanced sampling method")
    print("   - 25 epochs with reshuffling every 3 epochs")
    print()
    print("2. 📁 Fixed Validation (Filtered Database)")
    print("   - Uses preserved filtered_database_old dataset")
    print("   - Fixed train/val split throughout training")
    print("   - Does NOT modify original dataset")
    print()
    print("3. ⚡ Simple Dynamic Validation")
    print("   - Simplified dynamic approach")
    print("   - Changes validation set every 5 epochs")
    print("   - Good balance of simplicity and dynamic sampling")
    print()
    print("4. 📊 Legacy Fixed Split")
    print("   - Traditional 80/20 train/validation split")
    print("   - For comparison with other methods")
    print("   - Uses original archive dataset")
    print()
    print("5. ❌ Exit")
    print()
    print("=" * 70)


def get_user_choice():
    """Get and validate user choice"""
    while True:
        try:
            choice = input("Select training method (1-5): ").strip()
            if choice in ['1', '2', '3', '4', '5']:
                return int(choice)
            else:
                print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)


def count_images(directory):
    """Count images in directory"""
    return len(list(directory.glob("*.jpg"))) if directory.exists() else 0


def copy_images(source_dir, dest_dir, prefix=""):
    """Copy all images from source to destination with optional prefix"""
    if not source_dir.exists():
        print(f"WARNING: Source directory does not exist: {source_dir}")
        return
    
    image_files = list(source_dir.glob("*.jpg"))
    for img_file in image_files:
        if prefix:
            dest_name = f"{prefix}_{img_file.name}"
        else:
            dest_name = img_file.name
        
        dest_file = dest_dir / dest_name
        if not dest_file.exists():
            shutil.copy2(img_file, dest_file)


def create_data_yaml(dataset_path):
    """Create data.yaml file for YOLOv8 classification"""
    data_yaml = {
        'path': str(dataset_path),
        'train': 'train',
        'val': 'val',
        'names': {
            0: 'no_seatbelt',
            1: 'seatbelt'
        }
    }
    
    yaml_path = dataset_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    return yaml_path


def publish_model_results(results_dir, method_name):
    """Publish training results using model manager"""
    try:
        from model_manager import publish_latest
        
        print(f"\nPublishing model to seatbelt_model/latest...")
        result = publish_latest(results_dir)
        print(f"✓ Published to: {result['latest_dir']}")
        print(f"✓ Copied files: {len(result['copied'])}")
        
        # Create method-specific classifier
        create_classifier_script(results_dir, method_name)
        
    except Exception as e:
        print(f"Warning: Could not publish model - {e}")


def create_classifier_script(model_dir, method_name):
    """Create a method-specific classifier script"""
    script_name = f"seatbelt_classifier_{method_name.lower().replace(' ', '_')}.py"
    
    classifier_code = f'''"""
Seatbelt Detection Classifier - {method_name}
Generated automatically after training completion
"""

from ultralytics import YOLO
from PIL import Image
import sys
from pathlib import Path

def classify_image(image_path, model_path=None):
    """Classify a single image"""
    try:
        # Use latest model if no specific path provided
        if model_path is None:
            from model_manager import get_latest_model_path
            model_path = get_latest_model_path()
            if model_path is None:
                model_path = "seatbelt_model/latest/weights/best.pt"
        
        # Load model
        model = YOLO(model_path)
        
        # Run prediction
        results = model(image_path)
        result = results[0]
        
        # Get prediction
        class_name = result.names[result.probs.top1]
        confidence = result.probs.top1conf.item()
        
        return class_name, confidence
        
    except Exception as e:
        print(f"Error: {{e}}")
        return None, None

def main():
    if len(sys.argv) != 2:
        print("Usage: python {script_name} <image_path>")
        print("Example: python {script_name} test_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {{image_path}}")
        sys.exit(1)
    
    print(f"Classifying: {{Path(image_path).name}}")
    print(f"Method: {method_name}")
    print("-" * 40)
    
    prediction, confidence = classify_image(image_path)
    
    if prediction:
        print(f"Prediction: {{prediction}}")
        print(f"Confidence: {{confidence:.1%}}")
        print("-" * 40)
        
        if 'seatbelt' in prediction.lower() and 'no' not in prediction.lower():
            print("Status: SAFE - Seatbelt detected")
        else:
            print("Status: UNSAFE - No seatbelt detected")
    else:
        print("Classification failed")

if __name__ == "__main__":
    main()
'''
    
    classifier_path = Path(__file__).parent / script_name
    with open(classifier_path, 'w') as f:
        f.write(classifier_code)
    
    print(f"✓ Created classifier: {script_name}")


# Training Method 1: Dynamic Validation (Original)
def train_dynamic_validation():
    """Original dynamic validation training"""
    print("🔄 DYNAMIC VALIDATION TRAINING")
    print("=" * 50)
    
    base_path = Path(__file__).parent
    dataset_path = base_path / "seatbelt_dataset_dynamic"
    
    # Setup dataset structure
    seatbelt_source = base_path / "archive (2)" / "images.cv_6zo3bssqvgd8yvuq188n3s" / "data" / "train" / "seat_belt"
    no_seatbelt_source = base_path / "no seatbelt.v3i.yolov8" / "train" / "images"
    
    train_pool_seatbelt = dataset_path / "train_pool" / "seatbelt"
    train_pool_no_seatbelt = dataset_path / "train_pool" / "no_seatbelt"
    
    for dest_dir in [train_pool_seatbelt, train_pool_no_seatbelt]:
        dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images to pool if not already done
    if not any(train_pool_seatbelt.glob("*.jpg")):
        print("Setting up image pools...")
        copy_images(seatbelt_source, train_pool_seatbelt)
        copy_images(no_seatbelt_source, train_pool_no_seatbelt)
    
    # Count images
    seatbelt_count = count_images(train_pool_seatbelt)
    no_seatbelt_count = count_images(train_pool_no_seatbelt)
    
    print(f"Dataset Summary:")
    print(f"- Seatbelt images: {seatbelt_count}")
    print(f"- No-seatbelt images: {no_seatbelt_count}")
    print(f"- Total: {seatbelt_count + no_seatbelt_count}")
    print()
    
    if seatbelt_count < 10 or no_seatbelt_count < 10:
        print("❌ ERROR: Not enough images for training")
        return None
    
    # Dynamic training with epoch-based reshuffling
    try:
        # Import with fallback
        try:
            from ultralytics import YOLO
        except Exception:
            from ultralytics.yolo import YOLO
        
        print("🚀 Starting dynamic validation training...")
        print("- 25 epochs total")
        print("- Validation set reshuffled every 3 epochs")
        print()
        
        model = YOLO('yolov8n-cls.pt')
        total_epochs = 25
        reshuffle_every = 3
        results = None
        
        for session in range(0, total_epochs, reshuffle_every):
            current_epochs = min(reshuffle_every, total_epochs - session)
            print(f"Session {session//reshuffle_every + 1}: Epochs {session+1}-{session+current_epochs}")
            
            # Create new train/val split
            create_dynamic_split_original(dataset_path, session)
            
            # Train this session
            results = model.train(
                data=str(dataset_path),
                epochs=current_epochs,
                imgsz=224,
                batch=8,
                device='cpu',
                project='seatbelt_model',
                name='dynamic_training',
                resume=session > 0,
                cache=False,
                exist_ok=True
            )
        
        print("✓ Dynamic validation training completed!")
        return results.save_dir if results else None
        
    except Exception as e:
        print(f"❌ Dynamic training failed: {e}")
        return None


def create_dynamic_split_original(dataset_path, epoch, val_ratio=0.2):
    """Create dynamic split for original method"""
    train_pool_seatbelt = dataset_path / "train_pool" / "seatbelt"
    train_pool_no_seatbelt = dataset_path / "train_pool" / "no_seatbelt"
    
    train_active_seatbelt = dataset_path / "train" / "seatbelt"
    train_active_no_seatbelt = dataset_path / "train" / "no_seatbelt"
    val_active_seatbelt = dataset_path / "val" / "seatbelt"
    val_active_no_seatbelt = dataset_path / "val" / "no_seatbelt"
    
    # Create active directories
    for active_dir in [train_active_seatbelt, train_active_no_seatbelt, 
                       val_active_seatbelt, val_active_no_seatbelt]:
        active_dir.mkdir(parents=True, exist_ok=True)
        for img in active_dir.glob("*.jpg"):
            img.unlink()
    
    # Shuffle and split
    random.seed(epoch * 42)
    
    seatbelt_images = list(train_pool_seatbelt.glob("*.jpg"))
    no_seatbelt_images = list(train_pool_no_seatbelt.glob("*.jpg"))
    
    random.shuffle(seatbelt_images)
    random.shuffle(no_seatbelt_images)
    
    # Split
    sb_val_count = int(len(seatbelt_images) * val_ratio)
    nsb_val_count = int(len(no_seatbelt_images) * val_ratio)
    
    # Copy files
    for img in seatbelt_images[:sb_val_count]:
        shutil.copy2(img, val_active_seatbelt / img.name)
    for img in seatbelt_images[sb_val_count:]:
        shutil.copy2(img, train_active_seatbelt / img.name)
    for img in no_seatbelt_images[:nsb_val_count]:
        shutil.copy2(img, val_active_no_seatbelt / img.name)
    for img in no_seatbelt_images[nsb_val_count:]:
        shutil.copy2(img, train_active_no_seatbelt / img.name)


# Training Method 2: Fixed Validation (Filtered Database)
def train_fixed_validation():
    """Fixed validation using filtered_database_old"""
    print("📁 FIXED VALIDATION TRAINING (Filtered Database)")
    print("=" * 50)
    
    base_path = Path(__file__).parent
    dataset_path = base_path / "filtered_database_old"
    
    # Verify dataset exists
    if not dataset_path.exists():
        print("❌ ERROR: filtered_database_old not found!")
        print("Please ensure the filtered dataset is available.")
        return None
    
    # Count images
    train_seatbelt = count_images(dataset_path / "train" / "seatbelt")
    train_no_seatbelt = count_images(dataset_path / "train" / "no_seatbelt")
    val_seatbelt = count_images(dataset_path / "val" / "seatbelt")
    val_no_seatbelt = count_images(dataset_path / "val" / "no_seatbelt")
    
    print(f"Dataset Summary:")
    print(f"Training:")
    print(f"  - Seatbelt: {train_seatbelt}")
    print(f"  - No-seatbelt: {train_no_seatbelt}")
    print(f"  - Total: {train_seatbelt + train_no_seatbelt}")
    print(f"Validation:")
    print(f"  - Seatbelt: {val_seatbelt}")
    print(f"  - No-seatbelt: {val_no_seatbelt}")
    print(f"  - Total: {val_seatbelt + val_no_seatbelt}")
    print(f"Grand Total: {train_seatbelt + train_no_seatbelt + val_seatbelt + val_no_seatbelt}")
    print()
    
    # Clear cache files
    cache_files = list(dataset_path.glob("*.cache"))
    for cache_file in cache_files:
        cache_file.unlink(missing_ok=True)
    
    try:
        # Import with fallback
        try:
            from ultralytics import YOLO
        except Exception:
            from ultralytics.yolo import YOLO
        
        print("🚀 Starting fixed validation training...")
        print("- 25 epochs")
        print("- Fixed train/validation split")
        print("- Original dataset NOT modified")
        print()
        
        model = YOLO('yolov8n-cls.pt')
        
        results = model.train(
            data=str(dataset_path),
            epochs=25,
            imgsz=224,
            batch=8,
            device='cpu',
            project='seatbelt_model',
            name='filtered_training',
            cache=False,
            exist_ok=True
        )
        
        print("✓ Fixed validation training completed!")
        return results.save_dir if results else None
        
    except Exception as e:
        print(f"❌ Fixed validation training failed: {e}")
        return None


# Training Method 3: Simple Dynamic Validation
def train_simple_dynamic():
    """Simple dynamic validation with periodic reshuffling"""
    print("⚡ SIMPLE DYNAMIC VALIDATION TRAINING")
    print("=" * 50)
    
    base_path = Path(__file__).parent
    source_dataset = base_path / "filtered_database_old"
    
    if not source_dataset.exists():
        print("❌ ERROR: filtered_database_old not found!")
        return None
    
    # Create dynamic dataset
    dynamic_dataset = base_path / "filtered_dataset_dynamic_simple"
    
    # Setup pool structure
    train_pool_seatbelt = dynamic_dataset / "train_pool" / "seatbelt"
    train_pool_no_seatbelt = dynamic_dataset / "train_pool" / "no_seatbelt"
    
    for dest_dir in [train_pool_seatbelt, train_pool_no_seatbelt]:
        dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all images to pool (combining original train + val)
    if not any(train_pool_seatbelt.glob("*.jpg")):
        print("Creating image pool from filtered database...")
        
        # Copy from train folders
        copy_images(source_dataset / "train" / "seatbelt", train_pool_seatbelt, "train")
        copy_images(source_dataset / "train" / "no_seatbelt", train_pool_no_seatbelt, "train")
        
        # Copy from val folders
        copy_images(source_dataset / "val" / "seatbelt", train_pool_seatbelt, "val")
        copy_images(source_dataset / "val" / "no_seatbelt", train_pool_no_seatbelt, "val")
    
    # Count total images
    total_seatbelt = count_images(train_pool_seatbelt)
    total_no_seatbelt = count_images(train_pool_no_seatbelt)
    
    print(f"Dataset Pool Summary:")
    print(f"- Seatbelt images: {total_seatbelt}")
    print(f"- No-seatbelt images: {total_no_seatbelt}")
    print(f"- Total: {total_seatbelt + total_no_seatbelt}")
    print()
    
    try:
        # Import with fallback
        try:
            from ultralytics import YOLO
        except Exception:
            from ultralytics.yolo import YOLO
        
        print("🚀 Starting simple dynamic training...")
        print("- 25 epochs total")
        print("- Validation set changes every 5 epochs")
        print("- Uses combined train+val pool")
        print()
        
        epochs_total = 25
        validation_changes = 5
        
        current_model = None
        results = None
        
        for validation_round in range(1, (epochs_total // validation_changes) + 1):
            print(f"Validation Round {validation_round}/5")
            
            # Create new split
            create_simple_dynamic_split(dynamic_dataset, validation_round)
            
            # Clear cache
            cache_files = list(dynamic_dataset.glob("*.cache"))
            for cache_file in cache_files:
                cache_file.unlink(missing_ok=True)
            
            # Initialize or load model
            if current_model is None:
                model = YOLO('yolov8n-cls.pt')
            else:
                model = YOLO(str(current_model))
            
            # Train this round
            results = model.train(
                data=str(dynamic_dataset),
                epochs=validation_changes,
                imgsz=224,
                batch=8,
                device='cpu',
                project='seatbelt_model',
                name='dynamic_simple_training',
                cache=False,
                exist_ok=True,
                resume=False
            )
            
            if results and hasattr(results, 'save_dir'):
                current_model = results.save_dir / "weights" / "last.pt"
            print(f"✓ Round {validation_round} completed")
        
        print("✓ Simple dynamic training completed!")
        return results.save_dir if results and hasattr(results, 'save_dir') else None
        
    except Exception as e:
        print(f"❌ Simple dynamic training failed: {e}")
        return None


def create_simple_dynamic_split(dataset_path, round_num, val_ratio=0.2):
    """Create split for simple dynamic method"""
    train_pool_seatbelt = dataset_path / "train_pool" / "seatbelt"
    train_pool_no_seatbelt = dataset_path / "train_pool" / "no_seatbelt"
    
    train_active_seatbelt = dataset_path / "train" / "seatbelt"
    train_active_no_seatbelt = dataset_path / "train" / "no_seatbelt"
    val_active_seatbelt = dataset_path / "val" / "seatbelt"
    val_active_no_seatbelt = dataset_path / "val" / "no_seatbelt"
    
    # Create and clear active directories
    for active_dir in [train_active_seatbelt, train_active_no_seatbelt,
                       val_active_seatbelt, val_active_no_seatbelt]:
        active_dir.mkdir(parents=True, exist_ok=True)
        for img in active_dir.glob("*.jpg"):
            img.unlink()
    
    # Set seed for this round
    random.seed(round_num * 42)
    
    # Shuffle and split
    seatbelt_images = list(train_pool_seatbelt.glob("*.jpg"))
    no_seatbelt_images = list(train_pool_no_seatbelt.glob("*.jpg"))
    
    random.shuffle(seatbelt_images)
    random.shuffle(no_seatbelt_images)
    
    # Split images
    sb_val_count = int(len(seatbelt_images) * val_ratio)
    nsb_val_count = int(len(no_seatbelt_images) * val_ratio)
    
    # Copy to active directories
    for img in seatbelt_images[:sb_val_count]:
        shutil.copy2(img, val_active_seatbelt / img.name)
    for img in seatbelt_images[sb_val_count:]:
        shutil.copy2(img, train_active_seatbelt / img.name)
    for img in no_seatbelt_images[:nsb_val_count]:
        shutil.copy2(img, val_active_no_seatbelt / img.name)
    for img in no_seatbelt_images[nsb_val_count:]:
        shutil.copy2(img, train_active_no_seatbelt / img.name)


# Training Method 4: Legacy Fixed Split
def train_legacy_fixed():
    """Legacy training with traditional fixed split"""
    print("📊 LEGACY FIXED SPLIT TRAINING")
    print("=" * 50)
    
    base_path = Path(__file__).parent
    dataset_path = base_path / "seatbelt_dataset_legacy"
    
    # Setup dataset from original sources
    seatbelt_source = base_path / "archive (2)" / "images.cv_6zo3bssqvgd8yvuq188n3s" / "data" / "train" / "seat_belt"
    no_seatbelt_source = base_path / "no seatbelt.v3i.yolov8" / "train" / "images"
    
    train_seatbelt_dest = dataset_path / "train" / "seatbelt"
    train_no_seatbelt_dest = dataset_path / "train" / "no_seatbelt"
    val_seatbelt_dest = dataset_path / "val" / "seatbelt"
    val_no_seatbelt_dest = dataset_path / "val" / "no_seatbelt"
    
    # Create directories
    for dest_dir in [train_seatbelt_dest, train_no_seatbelt_dest, 
                     val_seatbelt_dest, val_no_seatbelt_dest]:
        dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup dataset if not already done
    if not any(train_seatbelt_dest.glob("*.jpg")):
        print("Setting up legacy dataset...")
        
        # Copy all images to train first
        copy_images(seatbelt_source, train_seatbelt_dest)
        copy_images(no_seatbelt_source, train_no_seatbelt_dest)
        
        # Move 20% to validation (fixed split)
        seatbelt_images = list(train_seatbelt_dest.glob("*.jpg"))
        no_seatbelt_images = list(train_no_seatbelt_dest.glob("*.jpg"))
        
        val_seatbelt_count = int(len(seatbelt_images) * 0.2)
        val_no_seatbelt_count = int(len(no_seatbelt_images) * 0.2)
        
        for img in seatbelt_images[:val_seatbelt_count]:
            shutil.move(str(img), str(val_seatbelt_dest / img.name))
        
        for img in no_seatbelt_images[:val_no_seatbelt_count]:
            shutil.move(str(img), str(val_no_seatbelt_dest / img.name))
    
    # Count images
    train_seatbelt = count_images(train_seatbelt_dest)
    train_no_seatbelt = count_images(train_no_seatbelt_dest)
    val_seatbelt = count_images(val_seatbelt_dest)
    val_no_seatbelt = count_images(val_no_seatbelt_dest)
    
    print(f"Dataset Summary:")
    print(f"Training: {train_seatbelt + train_no_seatbelt} images")
    print(f"Validation: {val_seatbelt + val_no_seatbelt} images")
    print(f"Total: {train_seatbelt + train_no_seatbelt + val_seatbelt + val_no_seatbelt}")
    print()
    
    try:
        # Import with fallback
        try:
            from ultralytics import YOLO
        except Exception:
            from ultralytics.yolo import YOLO
        
        print("🚀 Starting legacy fixed split training...")
        print("- 25 epochs")
        print("- Traditional 80/20 split")
        print("- Uses original archive dataset")
        print()
        
        model = YOLO('yolov8n-cls.pt')
        
        results = model.train(
            data=str(dataset_path),
            epochs=25,
            imgsz=224,
            batch=8,
            device='cpu',
            project='seatbelt_model',
            name='legacy_fixed_split',
            cache=False,
            exist_ok=True
        )
        
        print("✓ Legacy fixed split training completed!")
        return results.save_dir if results and hasattr(results, 'save_dir') else None
        
    except Exception as e:
        print(f"❌ Legacy training failed: {e}")
        return None


def main():
    """Main training interface"""
    training_methods = {
        1: ("Dynamic Validation (Original)", train_dynamic_validation),
        2: ("Fixed Validation (Filtered)", train_fixed_validation),
        3: ("Simple Dynamic Validation", train_simple_dynamic),
        4: ("Legacy Fixed Split", train_legacy_fixed)
    }
    
    while True:
        display_menu()
        choice = get_user_choice()
        
        if choice == 5:
            print("Exiting...")
            break
        
        method_name, training_function = training_methods[choice]
        
        print(f"\nStarting: {method_name}")
        print("=" * 50)
        
        try:
            results_dir = training_function()
            
            if results_dir:
                print("\n" + "=" * 70)
                print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
                print("=" * 70)
                print(f"Method: {method_name}")
                print(f"Results: {results_dir}")
                print(f"Weights: {results_dir}/weights/best.pt")
                
                # Publish results
                publish_model_results(results_dir, method_name)
                
                print("\n✓ Model published to seatbelt_model/latest/")
                print("✓ Use the Streamlit app to test your model!")
                
            else:
                print("\n❌ Training failed or was cancelled")
            
            input("\nPress Enter to return to menu...")
            
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            input("Press Enter to return to menu...")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            input("Press Enter to return to menu...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)