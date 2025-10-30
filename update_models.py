"""
Model Update Script
Automatically updates the latest model in both the Streamlit app and Flutter app
Run this after completing any training to ensure both apps use the newest model.
"""

import sys
from pathlib import Path
from model_manager import get_latest_model_path, copy_latest_model_to_flutter


def update_all_apps():
    """Update all apps to use the latest model"""
    print("=" * 60)
    print("🔄 UPDATING ALL APPS WITH LATEST MODEL")
    print("=" * 60)
    
    # Check if latest model exists
    latest_path = get_latest_model_path()
    if not latest_path:
        print("❌ No latest model found!")
        print("Please run a training session first using unified_training.py")
        return False
    
    print(f"✓ Latest model found: {latest_path}")
    print()
    
    # Check model file
    if not Path(latest_path).exists():
        print(f"❌ Model file does not exist: {latest_path}")
        return False
    
    model_size = Path(latest_path).stat().st_size / (1024 * 1024)  # MB
    print(f"📊 Model size: {model_size:.2f} MB")
    
    from datetime import datetime
    model_time = Path(latest_path).stat().st_mtime
    model_date = datetime.fromtimestamp(model_time).strftime("%Y-%m-%d %H:%M:%S")
    print(f"🕒 Model created: {model_date}")
    print()
    
    # Update Streamlit app
    print("🌐 Updating Streamlit Test App...")
    print("   - The Streamlit app automatically uses model_manager.get_latest_model_path()")
    print("   - No manual update needed")
    print("   - Run: python -m streamlit run seatbelt_test_app.py")
    print("   ✓ Streamlit app ready")
    print()
    
    # Update Flutter app
    print("📱 Updating Flutter App...")
    try:
        success = copy_latest_model_to_flutter()
        if success:
            print("   ✓ Model copied to Flutter assets")
            print("   - Location: Flutter APp/seatbelt_detector_app/assets/models/")
            print("   - Files: best.pt, latest.pt, seatbelt_model.pt")
            print("   - Rebuild Flutter app to use new model")
        else:
            print("   ❌ Failed to copy model to Flutter app")
            return False
    except Exception as e:
        print(f"   ❌ Error updating Flutter app: {e}")
        return False
    
    print()
    print("=" * 60)
    print("🎉 ALL APPS UPDATED SUCCESSFULLY!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Test Streamlit app: python -m streamlit run seatbelt_test_app.py")
    print("2. Rebuild Flutter app: cd 'Flutter APp/seatbelt_detector_app' && flutter build apk")
    print("3. Test Flutter app on device")
    print()
    
    return True


def show_current_status():
    """Show current model status"""
    print("=" * 60)
    print("📊 CURRENT MODEL STATUS")
    print("=" * 60)
    
    latest_path = get_latest_model_path()
    if latest_path:
        print(f"Latest model: {latest_path}")
        
        if Path(latest_path).exists():
            from datetime import datetime
            model_time = Path(latest_path).stat().st_mtime
            model_date = datetime.fromtimestamp(model_time).strftime("%Y-%m-%d %H:%M:%S")
            model_size = Path(latest_path).stat().st_size / (1024 * 1024)
            print(f"Created: {model_date}")
            print(f"Size: {model_size:.2f} MB")
            print("Status: ✓ Available")
        else:
            print("Status: ❌ File missing")
    else:
        print("Latest model: None found")
        print("Status: ❌ No model available")
    
    print()
    
    # Check Flutter assets
    flutter_models = Path(__file__).parent / "Flutter APp" / "seatbelt_detector_app" / "assets" / "models"
    if flutter_models.exists():
        print("Flutter app models:")
        for model_file in flutter_models.glob("*.pt"):
            model_time = model_file.stat().st_mtime
            model_date = datetime.fromtimestamp(model_time).strftime("%Y-%m-%d %H:%M:%S")
            model_size = model_file.stat().st_size / (1024 * 1024)
            print(f"  - {model_file.name}: {model_date} ({model_size:.2f} MB)")
    else:
        print("Flutter app models: None found")
    
    print()


def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        show_current_status()
    else:
        if update_all_apps():
            show_current_status()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nUpdate cancelled by user")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)