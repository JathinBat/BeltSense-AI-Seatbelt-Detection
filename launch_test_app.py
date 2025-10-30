"""
Smart launcher for the Seatbelt Detection Test App
Checks requirements and provides helpful guidance
"""

import sys
import subprocess
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version.split()[0]} - OK")
    return True

def check_streamlit():
    """Check if Streamlit is installed"""
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__} - OK")
        return True
    except ImportError:
        print("❌ Streamlit not installed")
        print("   Fix: pip install streamlit")
        return False

def check_ultralytics():
    """Check if Ultralytics is installed"""
    try:
        import ultralytics
        print(f"✅ Ultralytics - OK")
        return True
    except ImportError:
        print("❌ Ultralytics not installed")
        print("   Fix: pip install ultralytics")
        return False

def check_model_exists():
    """Check if a trained model exists"""
    model_dirs = list(Path(".").glob("seatbelt_model/*/weights/*.pt"))
    if model_dirs:
        latest_model = sorted(model_dirs, key=lambda x: x.stat().st_mtime)[-1]
        print(f"✅ Found trained model: {latest_model}")
        return True
    else:
        print("⚠️  No trained model found")
        print("   Run 'python run_training.py' first to train a model")
        return False

def install_requirements():
    """Offer to install missing requirements"""
    print("\n🔧 Would you like to install missing requirements? (y/n): ", end="")
    choice = input().lower().strip()
    
    if choice in ['y', 'yes']:
        print("Installing requirements...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "streamlit", "ultralytics"], check=True)
            print("✅ Requirements installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install requirements")
            return False
    return False

def main():
    print("🚗 Seatbelt Detection Test App Launcher")
    print("=" * 45)
    
    # Check requirements
    python_ok = check_python_version()
    streamlit_ok = check_streamlit()
    ultralytics_ok = check_ultralytics()
    model_exists = check_model_exists()
    
    # Install missing requirements if needed
    if not (streamlit_ok and ultralytics_ok):
        if python_ok and install_requirements():
            streamlit_ok = ultralytics_ok = True
        else:
            print("\n❌ Cannot start app - missing requirements")
            input("Press Enter to exit...")
            return
    
    if not model_exists:
        print("\n⚠️  You can still run the app, but you'll need to train a model first")
        print("   or specify a model path manually in the app")
    
    print("\n🚀 Starting Seatbelt Detection Test App...")
    print("   Browser will open automatically")
    print("   Press Ctrl+C in this window to stop")
    print("   App URL: http://localhost:8501")
    print("\n" + "="*45)
    
    try:
        # Launch the Streamlit app
        subprocess.run([sys.executable, "-m", "streamlit", "run", "seatbelt_test_app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting app: {e}")
        input("Press Enter to exit...")
    except FileNotFoundError:
        print("\n❌ seatbelt_test_app.py not found in current directory")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()