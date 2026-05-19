#!/usr/bin/env python3
"""MongoDB Atlas Connection Diagnostics"""
import socket
import subprocess
import sys

print("=" * 70)
print("MongoDB Atlas Connection Diagnostics")
print("=" * 70)

# Test 1: DNS Resolution
print("\n[TEST 1] DNS Resolution")
print("-" * 70)
try:
    host = "cluster0.bpdrfrc.mongodb.net"
    ip = socket.gethostbyname(host)
    print(f"[SUCCESS] DNS Resolved: {host} → {ip}")
except socket.gaierror as e:
    print(f"❌ DNS Failed: {e}")
    print("   → MongoDB Atlas hostname is not resolving")
    print("   → Check internet connection or DNS settings")
    sys.exit(1)

# Test 2: Network Connectivity (Ping)
print("\n[TEST 2] Network Connectivity (Ping)")
print("-" * 70)
try:
    result = subprocess.run(
        ["ping", "-n", "1", host],
        capture_output=True,
        timeout=5,
        text=True
    )
    if result.returncode == 0:
        print(f"[SUCCESS] Ping successful to {host}")
    else:
        print(f"❌ Ping failed to {host}")
        print(f"   Output: {result.stdout}")
except Exception as e:
    print(f"❌ Ping error: {e}")

# Test 3: Port Connectivity
print("\n[TEST 3] Port Connectivity (27017/TCP)")
print("-" * 70)
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    
    # Try to connect to MongoDB port 27017 on the cluster
    result = sock.connect_ex((ip, 27017))
    
    if result == 0:
        print(f"[SUCCESS] Port 27017 is OPEN on {host}")
    else:
        print(f"❌ Port 27017 REFUSED/UNREACHABLE on {host}")
        print("   Possible causes:")
        print("   1. IP address not whitelisted in MongoDB Atlas")
        print("   2. MongoDB Atlas cluster is paused/stopped")
        print("   3. Network firewall blocking the connection")
    
    sock.close()
except socket.timeout:
    print(f"⏱️  Connection timed out on port 27017")
    print("   Possible causes:")
    print("   1. Firewall blocking connection")
    print("   2. MongoDB Atlas cluster is down")
    print("   3. Network latency issue")
except Exception as e:
    print(f"❌ Socket error: {e}")

# Test 4: Environment Variables
print("\n[TEST 4] Environment Variables")
print("-" * 70)
from dotenv import load_dotenv
import os

load_dotenv()

mongodb_url = os.getenv("MONGODB_URL")
if mongodb_url:
    print(f"[SUCCESS] MONGODB_URL is set")
    print(f"   {mongodb_url[:70]}...")
    
    # Extract username from URL
    if "://" in mongodb_url:
        creds = mongodb_url.split("://")[1].split("@")[0]
        username = creds.split(":")[0]
        print(f"   Username: {username}")
else:
    print(f"❌ MONGODB_URL not set in .env")

# Test 5: MongoDB Atlas Credentials
print("\n[TEST 5] MongoDB Atlas Credentials")
print("-" * 70)
if "Purushotham:Purushotham123" in (mongodb_url or ""):
    print(f"[SUCCESS] Credentials format: Purushotham (username exists)")
    print(f"   These are the credentials in your connection string")
    print(f"   [WARN]️  Verify these match your MongoDB Atlas user account")
else:
    print(f"❌ Credentials not found in connection string")

print("\n" + "=" * 70)
print("Diagnostic Summary")
print("=" * 70)
print("\nIf tests show:")
print("[SUCCESS] DNS Resolved: Your system can reach MongoDB Atlas")
print("[SUCCESS] Ping successful: Network path is working")
print("[SUCCESS] Port 27017 OPEN: IP is whitelisted and cluster is accessible")
print("   → Move to Python connection test")
print()
print("If tests show:")
print("❌ DNS Failed: Check internet connection")
print("❌ Ping failed: Check network/firewall")
print("❌ Port 27017 REFUSED: Check MongoDB Atlas IP whitelist settings")
print()
print("=" * 70)
