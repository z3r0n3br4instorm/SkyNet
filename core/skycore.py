import socket
import pickle
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
)
from core.logger import SkyLog

class SkyNet:
    def __init__(self, host='localhost', port=6000):
        self.modelname = "HuggingFaceTB/SmolLM-135M"
        self.host = host
        self.port = port
        self.tokenizer = None
        self.model = None
        self.model_type = None
        self.config = None
        self.workers = []
        self.shards = {}
        self.pipeline_stages = []
        self.stage_layers = []
        self.model_ready = False
        self.current_worker_count = 0
        self.logger = SkyLog()

    def load_model(self):
        self.logger.info(f"Loading model: {self.modelname}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.modelname)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            config = AutoConfig.from_pretrained(self.modelname)
            
            if config.is_encoder_decoder:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.modelname)
                self.model_type = 'seq2seq'
            elif hasattr(config, 'model_type'):
                if 'gpt' in config.model_type or 'opt' in config.model_type or 'llama' in config.model_type:
                    self.model = AutoModelForCausalLM.from_pretrained(self.modelname)
                    self.model_type = 'causal'
                elif 'bert' in config.model_type or 'roberta' in config.model_type:
                    self.model = AutoModel.from_pretrained(self.modelname)
                    self.model_type = 'encoder'
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(self.modelname)
                    self.model_type = 'causal'
            else:
                self.model = AutoModel.from_pretrained(self.modelname)
                self.model_type = 'encoder'
            
            self.model.eval()
            self.config = config
            self.logger.info(f"✓ Loaded {self.model_type} model: {config.model_type}")
            self.logger.info(f"✓ Layers: {config.num_hidden_layers}, Hidden: {config.hidden_size}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def send_large_data(self, conn, data):
        try:
            serialized = pickle.dumps(data)
            size = len(serialized)
            conn.sendall(size.to_bytes(8, 'big'))
            conn.sendall(serialized)
            return True
        except Exception as e:
            self.logger.error(f"Send failed: {e}")
            return False
    
    def recv_large_data(self, conn):
        try:
            size_bytes = b''
            while len(size_bytes) < 8:
                chunk = conn.recv(8 - len(size_bytes))
                if not chunk:
                    return None
                size_bytes += chunk
            
            size = int.from_bytes(size_bytes, 'big')
            data = b''
            while len(data) < size:
                chunk = conn.recv(min(4096, size - len(data)))
                if not chunk:
                    return None
                data += chunk
            
            return pickle.loads(data)
        except Exception as e:
            self.logger.error(f"Receive failed: {e}")
            return None
    
    def get_model_layers(self):
        if self.model_type == 'causal':
            if hasattr(self.model, 'transformer'):
                return self.model.transformer.h  # GPT-2
            elif hasattr(self.model, 'model'):
                return self.model.model.layers  # LLaMA
        elif self.model_type == 'seq2seq':
            return self.model.model.encoder.layer
        elif self.model_type == 'encoder':
            return self.model.encoder.layer
        raise ValueError(f"Cannot extract layers from {self.model_type}")
    
    def get_linear_layers(self, layer):
        linear_weights = {}
        
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.Linear):
                linear_weights[name] = {
                    'weight': module.weight,
                    'bias': module.bias if module.bias is not None else None
                }
        
        return linear_weights

    def skysplit(self):
        self.logger.info("Starting distributed model splitting...")
        
        num_workers = len(self.workers)
        if num_workers == 0:
            self.logger.error("No workers available")
            return
        
        self.shards = {}
        self.logger.info("Cleared existing shards")
        
        layers = self.get_model_layers()
        num_layers = len(layers)
        hidden_size = self.config.hidden_size
        
        max_tensor_splits = hidden_size
        
        if num_workers <= max_tensor_splits:
            num_pipeline_stages = min(num_layers, max(1, num_workers // 64))
            workers_per_stage = num_workers // num_pipeline_stages
        else:
            num_pipeline_stages = num_layers
            workers_per_stage = num_workers // num_layers
        
        self.logger.info(f"Strategy: {num_pipeline_stages} stages, {workers_per_stage} workers/stage")
        
        self.pipeline_stages = []
        self.stage_layers = []
        layers_per_stage = num_layers // num_pipeline_stages
        
        for stage_idx in range(num_pipeline_stages):
            start_worker = stage_idx * workers_per_stage
            end_worker = start_worker + workers_per_stage
            stage_workers = list(range(start_worker, min(end_worker, num_workers)))
            
            start_layer = stage_idx * layers_per_stage
            end_layer = start_layer + layers_per_stage if stage_idx < num_pipeline_stages - 1 else num_layers
            stage_layers = list(range(start_layer, end_layer))
            
            self.pipeline_stages.append(stage_workers)
            self.stage_layers.append(stage_layers)
            self.logger.info(f"  Stage {stage_idx}: Workers {stage_workers}, Layers {stage_layers}")
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os
        
        num_threads = min(os.cpu_count() or 4, num_layers)
        self.logger.info(f"Using {num_threads} threads for parallel splitting")
        
        def split_layer(stage_workers, layer_idx):
            layer = layers[layer_idx]
            num_stage_workers = len(stage_workers)
            
            linear_layers = self.get_linear_layers(layer)
            layer_shards = {}
            
            for local_idx, global_worker_id in enumerate(stage_workers):
                layer_shards[global_worker_id] = {}
                
                for layer_name, weights in linear_layers.items():
                    weight = weights['weight']
                    bias = weights['bias']
                    
                    weight_shards = torch.chunk(weight, num_stage_workers, dim=0)
                    layer_shards[global_worker_id][f'{layer_name}_weight'] = weight_shards[local_idx]
                    
                    if bias is not None:
                        bias_shards = torch.chunk(bias, num_stage_workers, dim=0)
                        layer_shards[global_worker_id][f'{layer_name}_bias'] = bias_shards[local_idx]
                    else:
                        layer_shards[global_worker_id][f'{layer_name}_bias'] = None
            
            return layer_idx, layer_shards
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for stage_workers, stage_layers in zip(self.pipeline_stages, self.stage_layers):
                for layer_idx in stage_layers:
                    future = executor.submit(split_layer, stage_workers, layer_idx)
                    futures.append(future)
            
            for future in as_completed(futures):
                layer_idx, layer_shards = future.result()
                self.shards[layer_idx] = layer_shards
        
        for worker_id, conn in enumerate(self.workers):
            worker_stage = None
            for stage_idx, stage_workers in enumerate(self.pipeline_stages):
                if worker_id in stage_workers:
                    worker_stage = stage_idx
                    break
            
            if worker_stage is None:
                continue
            
            assigned_layers = self.stage_layers[worker_stage]
            worker_shards = []
            for layer_idx in assigned_layers:
                if layer_idx in self.shards and worker_id in self.shards[layer_idx]:
                    worker_shards.append({'layer': layer_idx, **self.shards[layer_idx][worker_id]})
            
            self.send_large_data(conn, {'cmd': 'LOAD_SHARDS', 'shards': worker_shards})
            self.logger.info(f"  [+] Worker {worker_id} loaded {len(worker_shards)} layer shards")
        
        self.model_ready = True
        self.logger.info("✓ Model splitting completed")

    def register_worker(self):
        self.current_worker_count += 1
        self.logger.info(f"Worker registered. Total workers: {self.current_worker_count}")

    def deregister_worker(self):
        if self.current_worker_count > 0:
            self.current_worker_count -= 1
        self.logger.info(f"Worker deregistered. Total workers: {self.current_worker_count}")