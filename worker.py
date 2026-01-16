# DEMO WORKER
import socket
import pickle
import torch
import torch.nn.functional as F

class Worker:
    def __init__(self, server_host='localhost', server_port=6000, worker_id=0):
        self.server_host = server_host
        self.server_port = server_port
        self.worker_id = worker_id
        self.shards = {}
    
    def send_large_data(self, conn, data):
        pickled = pickle.dumps(data)
        size = len(pickled)
        conn.send(size.to_bytes(8, 'big'))
        conn.sendall(pickled)
    
    def recv_large_data(self, conn):
        size_bytes = b''
        while len(size_bytes) < 8:
            chunk = conn.recv(8 - len(size_bytes))
            if not chunk:
                return None
            size_bytes += chunk
        
        size = int.from_bytes(size_bytes, 'big')
        data = b''
        while len(data) < size:
            chunk = conn.recv(min(size - len(data), 1048576))
            if not chunk:
                raise ConnectionError(f"Connection closed")
            data += chunk
        return pickle.loads(data)
        
    def register(self):
        import time
        max_retries = 10
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.server_host, self.server_port))
                sock.send(b"REGISTER_WORKER")
                
                response = sock.recv(1024).decode()
                if response == "OK":
                    print(f">>> Worker {self.worker_id} registered")
                    self.connection = sock
                    return True
                return False
            except ConnectionRefusedError:
                print(f">>> Connection refused, retrying in {retry_delay}s... ({attempt+1}/{max_retries})")
                time.sleep(retry_delay)
            except Exception as e:
                print(f">>> Connection error: {e}")
                return False
        
        print(f">>> Failed to connect after {max_retries} attempts")
        return False
    
    def wait_for_work(self):
        print(f">>> Worker {self.worker_id} waiting...")
        while True:
            try:
                task = self.recv_large_data(self.connection)
                if not task:
                    print(f">>> Connection closed, reconnecting...")
                    self.connection.close()
                    if self.register():
                        continue
                    else:
                        break
                    
                if task['cmd'] == 'LOAD_SHARDS':
                    self.load_shards(task['shards'])
                elif task['cmd'] == 'COMPUTE':
                    result = self.compute_layer(task)
                    self.send_large_data(self.connection, result)
            except ConnectionError as e:
                print(f">>> Connection error: {e}, reconnecting...")
                try:
                    self.connection.close()
                except:
                    pass
                if self.register():
                    continue
                else:
                    break
            except Exception as e:
                print(f">>> Error: {e}")
                import traceback
                traceback.print_exc()
                break
    
    def load_shards(self, shards):
        print(f">>> Loading {len(shards)} shards...")
        # Clear existing shards before loading new ones
        import gc
        self.shards = {}
        gc.collect()
        
        for shard in shards:
            # Detach tensors to prevent gradient tracking and reduce memory
            detached_shard = {'layer': shard['layer']}
            for key, value in shard.items():
                if key != 'layer' and value is not None:
                    if isinstance(value, torch.Tensor):
                        detached_shard[key] = value.detach().clone()
                    else:
                        detached_shard[key] = value
                elif key != 'layer':
                    detached_shard[key] = None
            self.shards[shard['layer']] = detached_shard
        
        gc.collect()
        print(f">>> Worker {self.worker_id} ready!")
    
    def compute_layer(self, task):
        layer_idx = task['layer']
        x = task['input'].detach()  # Detach input to break gradient chain
        shard = self.shards[layer_idx]
        
        batch, seq_len, dim = x.shape
        
        weights = {k: v for k, v in shard.items() if k.endswith('_weight')}
        biases = {k: v for k, v in shard.items() if k.endswith('_bias')}
        
        with torch.no_grad():  # Ensure no gradient computation
            if task.get('ln_1_weight') is not None:
                x = F.layer_norm(x, (dim,), task['ln_1_weight'], task['ln_1_bias'])
            
            hidden = x
            for weight_name, weight in weights.items():
                bias_name = weight_name.replace('_weight', '_bias')
                bias = biases.get(bias_name)
                
                if weight.dim() == 2:
                    hidden = torch.matmul(hidden, weight.T)
                else:
                    hidden = torch.matmul(hidden, weight)
                
                if bias is not None:
                    hidden = hidden + bias
                
                # Apply activation if not last layer
                # Heuristic: if output grows, it's an expansion layer (apply GELU)
                if hidden.shape[-1] > dim:
                    hidden = F.gelu(hidden)
            
            if task.get('ln_2_weight') is not None:
                hidden = F.layer_norm(hidden, (dim,), task['ln_2_weight'], task['ln_2_bias'])
        
        return hidden.detach()  # Detach result before returning

if __name__ == "__main__":
    import sys
    worker_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    worker = Worker(worker_id=worker_id)
    if worker.register():
        worker.wait_for_work()
