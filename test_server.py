#!/usr/bin/env python3
"""
Test script for the steganographic server
Demonstrates how to use the /encode and /decode endpoints
"""

import requests
import json
import time

# Server configuration
SERVER_URL = "http://localhost:3000"

def test_health():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{SERVER_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_encode():
    """Test the encode endpoint"""
    print("\nTesting encode endpoint...")
    
    # Test data
    test_data = {
        "ciphertext": "48656c6c6f20576f726c64",  # "Hello World" in hex
        "start_text": "The weather today is quite nice and ",
        "temp": 1.2,
        "precision": 16,
        "topk": 50000
    }
    
    try:
        print(f"Sending request to {SERVER_URL}/encode")
        print(f"Request data: {json.dumps(test_data, indent=2)}")
        
        response = requests.post(f"{SERVER_URL}/encode", json=test_data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result['success']}")
            print(f"Stego text: {result['stego_text']}")
            print(f"Starter length: {result['starter_length']}")
            print(f"Stats: {json.dumps(result['stats'], indent=2)}")
            return result['stego_text'], result['starter_length']
        else:
            print(f"Error: {response.json()}")
            return None
            
    except Exception as e:
        print(f"Encode test failed: {e}")
        return None

def test_decode(stego_text, starter_length):
    """Test the decode endpoint"""
    print("\nTesting decode endpoint...")
    
    # Test data
    test_data = {
        "stego_text": stego_text,
        "starter_length": starter_length,
        "temp": 1.2,
        "precision": 16,
        "topk": 50000
    }
    
    try:
        print(f"Sending request to {SERVER_URL}/decode")
        print(f"Request data (truncated): {json.dumps({**test_data, 'stego_text': test_data['stego_text'][:50] + '...'}, indent=2)}")
        
        response = requests.post(f"{SERVER_URL}/decode", json=test_data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result['success']}")
            print(f"Recovered ciphertext: {result['ciphertext']}")
            print(f"Stats: {json.dumps(result['stats'], indent=2)}")
            
            # Verify the recovered ciphertext matches the original
            original_hex = "48656c6c6f20576f726c64"
            recovered_hex = result['ciphertext']
            
            # Compare the beginning (may have extra bits at the end)
            if recovered_hex.startswith(original_hex):
                print("✓ Decoding successful - recovered ciphertext matches original!")
                return True
            else:
                print(f"✗ Decoding mismatch:")
                print(f"  Original:  {original_hex}")
                print(f"  Recovered: {recovered_hex}")
                return False
        else:
            print(f"Error: {response.json()}")
            return False
            
    except Exception as e:
        print(f"Decode test failed: {e}")
        return False

def test_api_documentation():
    """Test the root endpoint (API documentation)"""
    print("\nTesting API documentation endpoint...")
    try:
        response = requests.get(f"{SERVER_URL}/")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"API Name: {result['name']}")
            print(f"Version: {result['version']}")
            print("Available endpoints:")
            for endpoint, description in result['endpoints'].items():
                print(f"  {endpoint}: {description}")
        else:
            print(f"Error: {response.json()}")
    except Exception as e:
        print(f"Documentation test failed: {e}")

def main():
    """Run all tests"""
    print("=" * 80)
    print("STEGANOGRAPHIC SERVER TEST SUITE")
    print("=" * 80)
    
    # Test API documentation
    test_api_documentation()
    
    # Test health check
    if not test_health():
        print("Server is not healthy. Please start the server first.")
        print("Run: python stego_server.py")
        return
    
    # Test encoding
    encode_result = test_encode()
    if not encode_result:
        print("Encoding test failed. Cannot proceed with decoding test.")
        return
    
    stego_text, starter_length = encode_result
    
    # Test decoding
    decode_success = test_decode(stego_text, starter_length)
    
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    if decode_success:
        print("✓ All tests passed! The steganographic server is working correctly.")
    else:
        print("✗ Some tests failed. Please check the server logs.")
    print("=" * 80)

if __name__ == "__main__":
    main() 