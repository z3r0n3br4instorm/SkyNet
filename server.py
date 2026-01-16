# DEMO SERVER
import socket
import sys
from core.skycore import SkyNet

class SkyServer(SkyNet):
    def __init__(self, host='0.0.0.0', port=6000):
        super().__init__(host, port)
        self.logger.info("Zerone Laboratories\nSkyNet")
        self.logger.info(f"Initializing SkyServer on {host}:{port}")
    
    def start(self):
        self.logger.info("Loading model...")
        self.load_model()
        
        self.logger.info("Waiting for workers to connect...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(10)
        self.logger.info(f"✓ Listening on {self.host}:{self.port}")
        
        try:
            while True:
                conn, addr = sock.accept()
                data = conn.recv(1024).decode()
                
                if data == "REGISTER_WORKER":
                    self.workers.append(conn)
                    worker_id = len(self.workers) - 1
                    self.logger.info(f"✓ Worker {worker_id} registered from {addr}")
                    conn.send(b"OK")
                    
                    worker_count = len(self.workers)
                    
                    if not self.model_ready and worker_count >= 6:
                        self.logger.info(f"Minimum workers reached ({worker_count}) - splitting model...")
                        self.skysplit()
                        self.current_worker_count = worker_count
                    elif self.model_ready and worker_count != self.current_worker_count:
                        self.logger.info(f"Rebalancing: {self.current_worker_count} -> {worker_count} workers (inference continues)")
                        import threading
                        def rebalance():
                            self.skysplit()
                            self.current_worker_count = worker_count
                            self.logger.info("✓ Rebalancing complete - workers reset")
                        threading.Thread(target=rebalance, daemon=True).start()
                    elif not self.model_ready:
                        self.logger.info(f"Waiting for minimum 6 workers ({worker_count}/6)")
                
                elif data.startswith("INFERENCE:"):
                    if not self.model_ready:
                        self.logger.error("Model not ready - no workers available")
                        conn.send(b"ERROR:NO_WORKERS")
                        conn.close()
                        continue
                    
                    text = data.split(":", 1)[1]
                    self.logger.info(f">>> Inference request: '{text[:50]}...'")
                    
                    try:
                        result = self.run_inference(text)
                        self.send_large_data(conn, result)
                        self.logger.info("✓ Inference complete")
                    except Exception as e:
                        self.logger.error(f"Inference failed: {e}")
                        conn.send(b"ERROR:INFERENCE_FAILED")
                    
                    conn.close()
                
                else:
                    self.logger.warning(f"Unknown command from {addr}: {data[:50]}")
                    conn.close()
        
        except KeyboardInterrupt:
            self.logger.info("\n>>> Shutting down server...")
            sock.close()
            sys.exit(0)
    
    def run_inference(self, text, max_tokens=20):
        import torch
        
        self.logger.info(f"Tokenizing input...")
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids
        
        layers = self.get_model_layers()
        
        if self.model_type == 'causal':
            if hasattr(self.model, 'transformer'):
                wte = self.model.transformer.wte
                wpe = self.model.transformer.wpe
                ln_f = self.model.transformer.ln_f
            else:
                wte = self.model.model.embed_tokens
                wpe = None
                ln_f = self.model.model.norm
            lm_head = self.model.lm_head
        elif self.model_type == 'encoder':
            self.logger.error("Encoder-only models don't support text generation")
            return {'input': text, 'generated': text}
        
        # Disable gradient tracking for inference - saves memory and speeds up computation
        with torch.inference_mode():
            for token_idx in range(max_tokens):
                self.logger.info(f"  Generating token {token_idx + 1}/{max_tokens}...")
                
                hidden = wte(input_ids)
                if wpe is not None:
                    positions = torch.arange(input_ids.size(1))
                    hidden = hidden + wpe(positions)
                
                for stage_idx, (stage_workers, stage_layers) in enumerate(zip(self.pipeline_stages, self.stage_layers)):
                    for layer_idx in stage_layers:
                        layer = layers[layer_idx]
                        
                        ln_1_weight = None
                        ln_1_bias = None
                        ln_2_weight = None
                        ln_2_bias = None
                        
                        for name, module in layer.named_modules():
                            if 'ln_1' in name or 'input_layernorm' in name:
                                if hasattr(module, 'weight'):
                                    ln_1_weight = module.weight
                                    ln_1_bias = module.bias if hasattr(module, 'bias') else None
                            elif 'ln_2' in name or 'post_attention_layernorm' in name:
                                if hasattr(module, 'weight'):
                                    ln_2_weight = module.weight
                                    ln_2_bias = module.bias if hasattr(module, 'bias') else None
                        
                        # Send input to all workers in this stage
                        active_workers = []
                        for worker_id in stage_workers:
                            conn = self.workers[worker_id]
                            success = self.send_large_data(conn, {
                                'cmd': 'COMPUTE',
                                'layer': layer_idx,
                                'input': hidden,
                                'ln_1_weight': ln_1_weight,
                                'ln_1_bias': ln_1_bias,
                                'ln_2_weight': ln_2_weight,
                                'ln_2_bias': ln_2_bias
                            })
                            if success:
                                active_workers.append(worker_id)
                        
                        # Collect all partial results from workers
                        all_worker_results = []
                        for worker_id in active_workers:
                            conn = self.workers[worker_id]
                            worker_partial_results = self.recv_large_data(conn)
                            if worker_partial_results is not None:
                                all_worker_results.append(worker_partial_results)
                        
                        if not all_worker_results:
                            self.logger.warning(f"No active workers for stage {stage_idx}")
                            continue
                        
                        # Group results by operation name and combine
                        # All workers return results for the same operations in the same order
                        num_operations = len(all_worker_results[0])
                        
                        for op_idx in range(num_operations):
                            # Collect this operation's results from all workers
                            op_results = [worker_results[op_idx] for worker_results in all_worker_results]
                            parallel_type = op_results[0]['parallel_type']
                            
                            if parallel_type == 'column':
                                # Column-parallel: concatenate along feature dimension
                                partial_outputs = [r['output'] for r in op_results]
                                combined = torch.cat(partial_outputs, dim=-1)
                            elif parallel_type == 'row':
                                # Row-parallel: sum (all-reduce)
                                partial_outputs = [r['output'] for r in op_results]
                                combined = sum(partial_outputs)
                            else:
                                combined = op_results[0]['output']
                            
                            # Apply activation if this is an expansion layer (rough heuristic)
                            if combined.shape[-1] > hidden.shape[-1]:
                                combined = torch.nn.functional.gelu(combined)
                            
                            # Update hidden state  
                            hidden = combined
                
                hidden = ln_f(hidden)
                logits = lm_head(hidden)
                
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                if self.tokenizer.eos_token_id and next_token.item() == self.tokenizer.eos_token_id:
                    self.logger.info("EOS token reached")
                    break
            
            generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        return {
            'input': text,
            'generated': generated_text
        }

if __name__ == "__main__":
    server = SkyServer()
    server.start()
