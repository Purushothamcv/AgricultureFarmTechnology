"""
Quick Setup Script for SmartAgri Authentication System
Run this script to set up your environment and test the authentication system
"""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a shell command and print status"""
    print(f"\n{'='*60}")
    print(f"📋 {description}")
    print(f"{'='*60}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Success!")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"❌ Error:")
        print(result.stderr)
        return False
    
    return True


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     SmartAgri Authentication System - Setup Script       ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Check if .env exists
    if not os.path.exists('.env'):
        print("\n⚠️  .env file not found!")
        print("Creating .env file from .env.example...")
        
        with open('.env.example', 'r') as example:
            with open('.env', 'w') as env:
                env.write(example.read())
        
        print("✅ .env file created!")
        print("\n📝 Please edit .env file with your MongoDB connection string:")
        print("   - For local MongoDB: MONGODB_URL=mongodb://localhost:27017")
        print("   - For MongoDB Atlas: MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/")
        print("\nPress Enter after you've configured .env...")
        input()
    
    # Install dependencies
    print("\n📦 Installing Python dependencies...")
    if not run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing required packages"
    ):
        print("\n❌ Failed to install dependencies. Please check your Python environment.")
        return
    
    # Check MongoDB connection
    print("\n🔍 Checking MongoDB connection...")
    print("Make sure MongoDB is running!")
    print("   - Local: Start with 'mongod' command")
    print("   - Atlas: Ensure you have internet connection")
    
    print("\n" + "="*60)
    print("✨ Setup Complete!")
    print("="*60)
    print("\n🚀 To start the server, run:")
    print("   uvicorn main_fastapi:app --reload --host 0.0.0.0 --port 8000")
    print("\n📚 API Documentation will be available at:")
    print("   http://localhost:8000/docs")
    print("\n📖 For detailed information, check AUTH_DOCUMENTATION.md")
    

if __name__ == "__main__":
    main()
