#!/usr/bin/env python3
import socket
import pickle
import sys

class InferenceClient:
    def __init__(self, server_host='localhost', server_port=6000):
        self.server_host = server_host
        self.server_port = server_port
    
    def infer(self, text):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.server_host, self.server_port))
        
        sock.send(f"INFERENCE:{text}".encode())
        
        size_bytes = b''
        while len(size_bytes) < 8:
            chunk = sock.recv(8 - len(size_bytes))
            if not chunk:
                break
            size_bytes += chunk
        
        if len(size_bytes) < 8:
            print("Error: Failed to receive data size")
            sock.close()
            return None
        
        size = int.from_bytes(size_bytes, 'big')
        data = b''
        while len(data) < size:
            chunk = sock.recv(min(4096, size - len(data)))
            if not chunk:
                break
            data += chunk
        
        result = pickle.loads(data)
        sock.close()
        
        return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client.py <text>")
        print('Example: python client.py "Once upon a time"')
        sys.exit(1)
    
    text = " ".join(sys.argv[1:])
    
    client = InferenceClient()
    print(f">>> Sending: {text}")
    result = client.infer(text)
    
    if result:
        print(f"\n>>> Input: {result['input']}")
        print(f">>> Generated: {result['generated']}")
