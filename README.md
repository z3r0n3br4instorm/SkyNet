# SkyNet - Distributed LLM Inference System

A hybrid tensor + pipeline parallel framework for running large language models across multiple workers.

## What is SkyNet?

SkyNet is a distributed inference system that splits large language models across multiple worker nodes, enabling you to run models that might not fit on a single device. Unlike traditional model parallelism approaches, SkyNet implements **hybrid tensor and pipeline parallelism** for optimal memory efficiency.

## Architecture Overview

### Traditional Approach (Pipeline Parallelism Only)
```
Layer 1 → Device 1
Layer 2 → Device 2  
Layer 3 → Device 3
```
One complete layer per device.

### SkyNet Approach (Hybrid Tensor + Pipeline Parallelism)
```
Layer 1 → [Worker 0: 1/4 weights] + [Worker 1: 1/4 weights] + [Worker 2: 1/4 weights] + [Worker 3: 1/4 weights]
Layer 2 → [Worker 4: 1/4 weights] + [Worker 5: 1/4 weights] + [Worker 6: 1/4 weights] + [Worker 7: 1/4 weights]
```
Weight matrices are split across workers within each pipeline stage, then partial results are aggregated.

## How It Works

### 1. Model Splitting (`skysplit`)

When workers connect, the server splits the model:

**Weight Tensor Splitting** (core/skycore.py:194-202):
```python
weight_shards = torch.chunk(weight, num_stage_workers, dim=0)
```

For a linear layer with weight shape `[3072, 768]` and 4 workers:
- Worker 0 gets `[768, 768]`
- Worker 1 gets `[768, 768]`
- Worker 2 gets `[768, 768]`
- Worker 3 gets `[768, 768]`

Each worker receives a **fraction of the weight matrices** but processes the **full hidden state**.

### 2. Inference Flow

During inference (server.py:130-154):

1. **Full hidden states sent to all workers** in a stage
2. Each worker computes **partial outputs** using their weight shard
3. Server **aggregates results** by summing: `hidden = hidden + sum(results)`

This works because splitting along the output dimension allows partial matrix multiplications to be summed for the final result.

### 3. Dynamic Rebalancing

When workers join or leave:
- Server automatically **re-splits** the model
- Pipeline stages are reconfigured
- Workers receive updated weight shards

## Example: Loading Qwen3-0.6B

### Step 1: Server Startup
```
Server starts → load_model() → downloads Qwen3-0.6B (600M params)
- 24 transformer layers
- hidden_size = 1024
- Full model loaded into server memory
```

### Step 2: Workers Connect
```
Worker 1 → sends "REGISTER_WORKER" → server: "OK"
Worker 2 → sends "REGISTER_WORKER" → server: "OK"
Worker 3 → sends "REGISTER_WORKER" → server: "OK"
Worker 4 → sends "REGISTER_WORKER" → server: "OK"
```

### Step 3: Model Splitting (4 workers)
```
Strategy: 1 pipeline stage, 4 tensor-parallel workers

For each of 24 layers:
  - Extract Linear layer weights (attention Q/K/V/O, MLP projections)
  - Split each weight matrix into 4 chunks along dim=0
  - Send 1/4 to each worker

Memory saved: 600MB × 4 = 2.4GB → distributed across workers!
```

### Step 4: Inference
```
Client: "INFERENCE:Hello world"

For each token (max 20):
  1. Embed token → hidden [1, seq_len, 1024]
  2. For each layer (0-23):
     - Send FULL hidden state to all 4 workers
     - Each worker: computes partial attention + MLP
     - Server: sums 4 partial results
  3. Layer norm → LM head → argmax → next token
  4. Append token, repeat
```

## Key Features

- **Hybrid Parallelism**: Combines tensor and pipeline parallelism
- **Dynamic Rebalancing**: Automatically adjusts when workers join/leave
- **Model Agnostic**: Supports GPT-2, LLaMA, Qwen, and other transformer architectures
- **Fault Tolerance**: Handles worker failures gracefully
- **Flexible Deployment**: Works across machines on a network

## Architecture Components

- **SkyServer** (server.py): Orchestrates inference, manages workers, coordinates computation
- **SkyCore** (core/skycore.py): Model loading, splitting logic, communication protocols
- **Worker** (core/prototyping/worker.py): Receives weight shards, computes partial results
- **Client** (client.py): Sends inference requests to server

## Performance Considerations

### Advantages
- **Memory Efficiency**: Distributes model weights across workers
- **Scalability**: Add more workers to handle larger models
- **Flexibility**: Dynamic worker management

### Trade-offs
- **Network Bandwidth**: Full hidden states sent to all workers in a stage
- **Latency**: Network communication overhead per layer
- **Synchronous Generation**: All workers wait for slowest in each stage

## Project Structure

```
skynet/
├── server.py              # Main server orchestrator
├── client.py              # Inference client
├── core/
│   ├── skycore.py         # Core model splitting & communication
│   ├── logger.py          # Logging utilities
│   └── prototyping/
│       ├── worker.py      # Worker implementation
│       ├── spawn_workers.sh
│       └── ...
└── skylog.txt            # Runtime logs
```

## Usage

### Start Server
```bash
python server.py
```

### Start Workers
```bash
python core/prototyping/worker.py 0
python core/prototyping/worker.py 1
python core/prototyping/worker.py 2
python core/prototyping/worker.py 3
```

### Run Inference
```bash
python client.py "Your prompt here"
```

## Technical Details

- **Model Type Detection**: Automatically identifies causal LM, encoder-only, or seq2seq models
- **Layer Norm Handling**: Explicitly passes layer norm parameters to workers for correctness
- **Tensor Serialization**: Uses pickle with size headers for reliable large data transfer
- **Communication Protocol**: TCP sockets with custom framing
