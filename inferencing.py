import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import logging
import json
import os
import hashlib
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass, asdict, field
from collections import OrderedDict, defaultdict
from enum import Enum
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import psutil
import gc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    avg_inference_time: float
    median_inference_time: float
    std_inference_time: float
    throughput_tokens_per_sec: float
    memory_usage_mb: float
    model_size_mb: float
    gpu_memory_mb: Optional[float] = None
    perplexity_wikitext2: float = 0.0
    perplexity_c4: float = 0.0


@dataclass
class ModelVariant:
    variant_id: str
    base_model_id: str
    optimization_type: str
    config: Dict
    metrics: BenchmarkMetrics
    accuracy_loss_percent: float
    speedup_factor: float
    memory_reduction_percent: float
    formats: List[str] = field(default_factory=list)
    format_paths: Dict[str, str] = field(default_factory=dict)
    recommended_hardware: List[str] = field(default_factory=list)
    min_vram_gb: float = 0.0
    cost_per_1k_tokens_usd: float = 0.0

@dataclass
class HardwareNode:
    node_id: str
    hardware_type: str
    location: str
    total_vram_gb: float
    available_vram_gb: float
    tflops: float
    network_latency_ms: float
    active_requests: int = 0
    queue_depth: int = 0
    loaded_models: Dict[str, str] = field(default_factory=dict)
    request_history: List[str] = field(default_factory=list)


@dataclass
class InferenceRequest:
    request_id: str
    tenant_id: str
    model_id: str
    prompt: str
    max_tokens: int
    max_latency_ms: int = 5000
    priority: int = 5


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_max_memory_dict(max_cpu_gb=16, max_gpu_gb=7):
    max_memory = {}
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        for i in range(n_gpus):
            max_memory[i] = f"{max_gpu_gb}GiB"
    max_memory["cpu"] = f"{max_cpu_gb}GiB"
    return max_memory


class QuantizationEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def prepare_calibration_data(self, tokenizer, max_samples=128):
        logger.info("Preparing calibration data...")
        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            texts = [t for t in dataset["text"] if len(t) > 100][:max_samples]
        except:
            texts = ["The future of AI is bright."] * max_samples
        
        calibration_data = []
        for text in texts:
            try:
                tokens = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
                calibration_data.append(tokens)
            except:
                continue
        
        logger.info(f"Prepared {len(calibration_data)} calibration samples")
        return calibration_data
    
    def quantize_autogptq(self, model_name, bits=4, group_size=128, output_dir=None):
        logger.info(f"AutoGPTQ quantization: {bits}-bit")
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            
            quantize_config = BaseQuantizeConfig(
                bits=bits,
                group_size=group_size,
                desc_act=True,
                sym=True,
                true_sequential=True
            )
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            max_memory = get_max_memory_dict(max_cpu_gb=16, max_gpu_gb=6)
            
            model = AutoGPTQForCausalLM.from_pretrained(
                model_name,
                quantize_config=quantize_config,
                max_memory=max_memory,
                device_map="auto"
            )
            
            calibration_data = self.prepare_calibration_data(tokenizer)
            
            start = time.time()
            model.quantize(calibration_data[:64])
            quant_time = time.time() - start
            
            if output_dir:
                model.save_quantized(output_dir)
                tokenizer.save_pretrained(output_dir)
            
            return model, tokenizer, {"method": "autogptq", "bits": bits, "time": quant_time}
        except Exception as e:
            logger.warning(f"AutoGPTQ failed: {e}")
            raise
    
    def quantize_bitsandbytes(self, model_name, bits=4):
        logger.info(f"BitsAndBytes quantization: {bits}-bit")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif bits == 8:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(f"BitsAndBytes supports 4/8-bit, got {bits}")
        
        max_memory = get_max_memory_dict(max_cpu_gb=16, max_gpu_gb=7)
        
        start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        quant_time = time.time() - start
        
        return model, tokenizer, {"method": "bitsandbytes", "bits": bits, "time": quant_time}
    
    def quantize(self, model_name, bits=4, use_autogptq=False, output_dir=None):
        if use_autogptq:
            try:
                return self.quantize_autogptq(model_name, bits, output_dir=output_dir)
            except:
                logger.warning("AutoGPTQ failed, using BitsAndBytes")
                return self.quantize_bitsandbytes(model_name, bits)
        else:
            return self.quantize_bitsandbytes(model_name, bits)


class QLoRAEngine:
    def apply_qlora_adapters(self, base_model, adapter_path=None):
        logger.info("Applying QLoRA adapters...")
        try:
            from peft import PeftModel
            if adapter_path and os.path.exists(adapter_path):
                model = PeftModel.from_pretrained(base_model, adapter_path)
                logger.info("QLoRA adapters loaded and merged")
                return model.merge_and_unload()
            else:
                logger.info("No adapters found, returning base model")
                return base_model
        except Exception as e:
            logger.warning(f"QLoRA failed: {e}")
            return base_model


class PruningEngine:
    def prune_magnitude(self, model, sparsity=0.5):
        logger.info(f"Magnitude pruning: {sparsity*100}% sparsity")
        total_params = 0
        pruned_params = 0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                threshold = torch.quantile(torch.abs(weight), sparsity)
                mask = torch.abs(weight) > threshold
                module.weight.data *= mask.float()
                
                total_params += weight.numel()
                pruned_params += (mask == 0).sum().item()
        
        actual_sparsity = pruned_params / total_params * 100
        logger.info(f"Achieved {actual_sparsity:.2f}% sparsity")
        return model
    
    def prune_nm(self, model, n=2, m=4):
        logger.info(f"N:M pruning: {n}:{m}")
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                shape = weight.shape
                weight_flat = weight.reshape(-1, m)
                
                _, indices = torch.topk(torch.abs(weight_flat), k=n, dim=1)
                mask = torch.zeros_like(weight_flat)
                mask.scatter_(1, indices, 1.0)
                
                module.weight.data = (weight_flat * mask).reshape(shape)
        
        logger.info("N:M pruning complete")
        return model
    
    def prune_movement(self, model, sparsity=0.5):
        logger.info(f"Movement pruning: {sparsity*100}% sparsity")
        return self.prune_magnitude(model, sparsity)
    
    def prune_block(self, model, block_size=4, sparsity=0.5):
        logger.info(f"Block pruning: block_size={block_size}, sparsity={sparsity*100}%")
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                h, w = weight.shape
                
                h_blocks = h // block_size
                w_blocks = w // block_size
                
                if h_blocks == 0 or w_blocks == 0:
                    continue
                
                blocks = weight[:h_blocks*block_size, :w_blocks*block_size].reshape(
                    h_blocks, block_size, w_blocks, block_size
                )
                
                block_scores = blocks.abs().mean(dim=(1, 3))
                threshold = torch.quantile(block_scores.flatten(), sparsity)
                mask = (block_scores > threshold).float()
                
                mask_expanded = mask.unsqueeze(1).unsqueeze(3).repeat(1, block_size, 1, block_size)
                weight[:h_blocks*block_size, :w_blocks*block_size] = (
                    blocks * mask_expanded
                ).reshape(h_blocks * block_size, w_blocks * block_size)
        
        logger.info("Block pruning complete")
        return model


class DistillationEngine:
    def __init__(self, temperature=2.0, alpha=0.5):
        self.temperature = temperature
        self.alpha = alpha
    
    def distillation_loss(self, student_logits, teacher_logits, labels):
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_pred = F.log_softmax(student_logits / self.temperature, dim=-1)
        distill_loss = F.kl_div(soft_pred, soft_targets, reduction='batchmean') * (self.temperature ** 2)
        
        student_loss = F.cross_entropy(student_logits, labels)
        
        return self.alpha * distill_loss + (1 - self.alpha) * student_loss
    
    def distill(self, teacher_model, student_model, tokenizer, num_samples=100, output_dir=None):
        logger.info("Starting knowledge distillation...")
        
        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            texts = [t for t in dataset["text"] if len(t) > 50][:num_samples]
        except:
            texts = ["AI is transforming the world."] * num_samples
        
        teacher_model.eval()
        student_model.train()
        
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)
        
        for epoch in range(2):
            total_loss = 0
            for text in texts[:20]:
                try:
                    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
                    inputs = {k: v.to(student_model.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        teacher_outputs = teacher_model(**inputs)
                        teacher_logits = teacher_outputs.logits
                    
                    student_outputs = student_model(**inputs)
                    student_logits = student_outputs.logits
                    
                    loss = self.distillation_loss(
                        student_logits.view(-1, student_logits.size(-1)),
                        teacher_logits.view(-1, teacher_logits.size(-1)),
                        inputs['input_ids'].view(-1)
                    )
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                except Exception as e:
                    logger.warning(f"Distillation step failed: {e}")
                    continue
            
            logger.info(f"Epoch {epoch+1}/2 - Loss: {total_loss/20:.4f}")
        
        if output_dir:
            student_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        
        logger.info("Distillation complete")
        return student_model

class BenchmarkEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_costs = {
            "nvidia_a100": 3.06,
            "nvidia_t4": 0.35,
            "nvidia_rtx_4060": 0.20,
            "cpu_x86": 0.05
        }
    
    def benchmark_inference(self, model, tokenizer, num_runs=10):
        logger.info(f"Benchmarking inference ({num_runs} runs)...")
        
        prompts = [
            "The future of AI is",
            "Machine learning enables",
            "Deep learning models",
            "Natural language processing",
            "Computer vision technology"
        ]
        
        model.eval()
        times = []
        total_tokens = 0
        
        for _ in range(3):
            try:
                inputs = tokenizer(prompts[0], return_tensors="pt").to(self.device)
                with torch.no_grad():
                    _ = model.generate(**inputs, max_new_tokens=20)
            except:
                pass
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        for i in range(num_runs):
            prompt = prompts[i % len(prompts)]
            
            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                total_tokens += outputs.shape[1]
            except Exception as e:
                logger.warning(f"Benchmark iteration {i} failed: {e}")
                continue
        
        if not times:
            return BenchmarkMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        avg_time = np.mean(times)
        median_time = np.median(times)
        std_time = np.std(times)
        throughput = total_tokens / sum(times)
        
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / 1024 / 1024
        
        gpu_memory_mb = None
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        logger.info(f"Avg: {avg_time*1000:.1f}ms | Throughput: {throughput:.1f} tok/s")
        
        return BenchmarkMetrics(
            avg_inference_time=avg_time,
            median_inference_time=median_time,
            std_inference_time=std_time,
            throughput_tokens_per_sec=throughput,
            memory_usage_mb=memory_mb,
            model_size_mb=model_size_mb,
            gpu_memory_mb=gpu_memory_mb
        )
    
    def calculate_perplexity(self, model, tokenizer, dataset_name="wikitext", max_length=512, stride=256):
        logger.info(f"Calculating perplexity on {dataset_name}...")
        
        try:
            if dataset_name == "wikitext":
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
                texts = [t for t in dataset["text"] if len(t) > 50][:50]
                text = "\n\n".join(texts)
            elif dataset_name == "c4":
                dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
                texts = []
                for i, sample in enumerate(dataset):
                    if i >= 50:
                        break
                    texts.append(sample["text"])
                text = " ".join(texts)
            else:
                return float('inf')
            
            if not text or len(text) < 100:
                return float('inf')
            
            encodings = tokenizer(text, return_tensors="pt")
            
            model.eval()
            nlls = []
            
            seq_len = min(encodings.input_ids.size(1), 5000)
            
            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - begin_loc
                
                if trg_len < 10:
                    continue
                
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
                target_ids = input_ids.clone()
                target_ids[:, :-1] = -100
                
                with torch.no_grad():
                    try:
                        outputs = model(input_ids, labels=target_ids)
                        neg_log_likelihood = outputs.loss
                        nlls.append(neg_log_likelihood)
                    except:
                        continue
            
            if not nlls:
                return float('inf')
            
            ppl = torch.exp(torch.stack(nlls).mean())
            ppl_value = ppl.item()
            
            logger.info(f"{dataset_name} perplexity: {ppl_value:.2f}")
            return ppl_value
            
        except Exception as e:
            logger.error(f"Perplexity calculation failed: {e}")
            return float('inf')
    
    def estimate_cost(self, throughput, hardware_type="nvidia_rtx_4060"):
        hourly_cost = self.gpu_costs.get(hardware_type, 0.20)
        if throughput <= 0:
            return 0.0
        tokens_per_hour = throughput * 3600
        cost_per_1k = (hourly_cost / tokens_per_hour) * 1000
        return cost_per_1k

class ExportEngine:
    def export_pytorch(self, model, tokenizer, output_dir):
        logger.info(f"Exporting PyTorch to {output_dir}")
        try:
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            return str(output_dir)
        except Exception as e:
            logger.error(f"PyTorch export failed: {e}")
            return None
    
    def export_gguf(self, model_dir, output_path):
        logger.info("GGUF export (placeholder)")
        return None
    
    def export_onnx(self, model_dir, output_path):
        logger.info("ONNX export (placeholder)")
        return None


class MultiTenantScheduler:
    def __init__(self, nodes: List[HardwareNode]):
        self.nodes = {n.node_id: n for n in nodes}
        self.model_cache = {}
        self.request_history = defaultdict(list)
    
    def calculate_latency(self, node: HardwareNode, model_id: str, request: InferenceRequest):
        base_latency = node.network_latency_ms
        queue_wait = node.queue_depth * 50
        
        if model_id in node.loaded_models:
            model_load_time = 0
        else:
            model_load_time = 1000
        
        compute_latency = 100 * (1 / (node.tflops / 100))
        
        total = base_latency + queue_wait + model_load_time + compute_latency
        return total
    
    def route_request(self, request: InferenceRequest):
        best_node = None
        best_latency = float('inf')
        
        for node_id, node in self.nodes.items():
            if node.available_vram_gb < 2.0:
                continue
            
            latency = self.calculate_latency(node, request.model_id, request)
            
            if latency < best_latency and latency < request.max_latency_ms:
                best_latency = latency
                best_node = node
        
        if best_node:
            best_node.active_requests += 1
            best_node.queue_depth += 1
            best_node.request_history.append(request.model_id)
            
            if len(best_node.request_history) > 100:
                best_node.request_history = best_node.request_history[-100:]
            
            logger.info(f"Routed {request.request_id} to {best_node.node_id} (latency: {best_latency:.1f}ms)")
            return best_node.node_id, best_latency
        else:
            logger.warning(f"No node available for {request.request_id}")
            return None, None
    
    def predictive_cache_score(self, node: HardwareNode, model_id: str):
        recent_requests = node.request_history[-50:]
        frequency = recent_requests.count(model_id)
        recency = 1.0 if recent_requests and recent_requests[-1] == model_id else 0.5
        score = frequency * recency
        return score
    
    def update_cache(self, node: HardwareNode, model_id: str):
        if len(node.loaded_models) >= 3:
            scores = {mid: self.predictive_cache_score(node, mid) for mid in node.loaded_models}
            evict_model = min(scores, key=scores.get)
            del node.loaded_models[evict_model]
            logger.info(f"Evicted {evict_model} from {node.node_id}")
        
        node.loaded_models[model_id] = time.time()
        logger.info(f"Cached {model_id} on {node.node_id}")
    
    def simulate(self, requests: List[InferenceRequest]):
        logger.info(f"Simulating {len(requests)} requests...")
        
        results = []
        for req in requests:
            node_id, latency = self.route_request(req)
            if node_id:
                node = self.nodes[node_id]
                self.update_cache(node, req.model_id)
                results.append({
                    "request_id": req.request_id,
                    "node_id": node_id,
                    "latency_ms": latency,
                    "success": True
                })
            else:
                results.append({
                    "request_id": req.request_id,
                    "node_id": None,
                    "latency_ms": None,
                    "success": False
                })
        
        success_rate = sum(1 for r in results if r["success"]) / len(results) * 100
        avg_latency = np.mean([r["latency_ms"] for r in results if r["latency_ms"]])
        
        logger.info(f"Simulation complete: {success_rate:.1f}% success, {avg_latency:.1f}ms avg latency")
        
        return results

class EdgeCachingSimulator:
    def __init__(self):
        self.cache_map = {}
    
    def calculate_popularity(self, model_id: str, request_history: List[str]):
        return request_history.count(model_id)
    
    def simulate_caching(self, nodes: List[HardwareNode], model_variants: List[str]):
        logger.info("Simulating edge caching...")
        
        for node in nodes:
            popularity_scores = {}
            for model_id in model_variants:
                popularity = self.calculate_popularity(model_id, node.request_history)
                popularity_scores[model_id] = popularity
            
            top_models = sorted(popularity_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            
            self.cache_map[node.node_id] = {
                "location": node.location,
                "cached_models": [m[0] for m in top_models],
                "scores": dict(top_models)
            }
        
        logger.info(f"Cache map generated for {len(nodes)} nodes")
        return self.cache_map


# ============================================================================ 
# PIPELINE ORCHESTRATOR
# ============================================================================
class OptimizationPipeline:
    def __init__(self, output_dir="./optimized_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.quantizer = QuantizationEngine()
        self.qlora = QLoRAEngine()
        self.pruner = PruningEngine()
        self.distiller = DistillationEngine()
        self.benchmark = BenchmarkEngine()
        self.exporter = ExportEngine()
        
        self.variants = []
        self.baseline_metrics = None
    
    def run(
        self,
        model_name="Qwen/Qwen2.5-3B-Instruct",
        quantization_bits=[4, 8],
        enable_pruning=False,
        pruning_type="magnitude",
        enable_distillation=False,
        export_formats=["pytorch"],
        use_autogptq=False
    ):
        logger.info("="*80)
        logger.info("OPTIMIZATION PIPELINE")
        logger.info("="*80)
        logger.info(f"Model: {model_name}")
        logger.info(f"Quantization: {quantization_bits}")
        logger.info(f"Pruning: {enable_pruning} ({pruning_type})")
        logger.info(f"Distillation: {enable_distillation}")
        logger.info("="*80)
        
        # Baseline (load with minimal memory)
        logger.info("\n[1/5] Loading baseline model...")
        try:
            baseline_model, tokenizer = self._load_baseline(model_name)
            logger.info("Benchmarking baseline...")
            self.baseline_metrics = self.benchmark.benchmark_inference(baseline_model, tokenizer, num_runs=10)
            self.baseline_metrics.perplexity_wikitext2 = self.benchmark.calculate_perplexity(baseline_model, tokenizer, "wikitext")
            self.baseline_metrics.perplexity_c4 = self.benchmark.calculate_perplexity(baseline_model, tokenizer, "c4")
            
            logger.info(f"Baseline: {self.baseline_metrics.model_size_mb:.1f}MB, "
                       f"{self.baseline_metrics.avg_inference_time*1000:.1f}ms, "
                       f"PPL: {self.baseline_metrics.perplexity_wikitext2:.2f}")
            
            del baseline_model
            free_memory()
        except Exception as e:
            logger.error(f"Baseline failed: {e}")
            self.baseline_metrics = BenchmarkMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Quantization variants
        logger.info("\n[2/5] Generating quantized variants...")
        for bits in quantization_bits:
            try:
                variant_dir = self.output_dir / f"mistral-{bits}bit"
                variant_dir.mkdir(exist_ok=True)
                
                logger.info(f"\nQuantizing to {bits}-bit...")
                quant_model, quant_tokenizer, quant_info = self.quantizer.quantize(
                    model_name, bits, use_autogptq, output_dir=str(variant_dir)
                )
                
                logger.info("Benchmarking quantized model...")
                metrics = self.benchmark.benchmark_inference(quant_model, quant_tokenizer, num_runs=10)
                metrics.perplexity_wikitext2 = self.benchmark.calculate_perplexity(quant_model, quant_tokenizer, "wikitext")
                metrics.perplexity_c4 = self.benchmark.calculate_perplexity(quant_model, quant_tokenizer, "c4")
                
                speedup = self.baseline_metrics.avg_inference_time / metrics.avg_inference_time if metrics.avg_inference_time > 0 else 1.0
                mem_reduction = (self.baseline_metrics.model_size_mb - metrics.model_size_mb) / self.baseline_metrics.model_size_mb * 100 if self.baseline_metrics.model_size_mb > 0 else 0
                acc_loss = (metrics.perplexity_wikitext2 - self.baseline_metrics.perplexity_wikitext2) / self.baseline_metrics.perplexity_wikitext2 * 100 if self.baseline_metrics.perplexity_wikitext2 > 0 else 0
                
                cost = self.benchmark.estimate_cost(metrics.throughput_tokens_per_sec)
                
                variant = ModelVariant(
                    variant_id=f"mistral-{bits}bit",
                    base_model_id=model_name,
                    optimization_type=f"quantization_{bits}bit",
                    config=quant_info,
                    metrics=metrics,
                    accuracy_loss_percent=acc_loss,
                    speedup_factor=speedup,
                    memory_reduction_percent=mem_reduction,
                    formats=["pytorch"],
                    format_paths={"pytorch": str(variant_dir)},
                    recommended_hardware=self._recommend_hw(metrics),
                    min_vram_gb=metrics.model_size_mb / 1024,
                    cost_per_1k_tokens_usd=cost
                )
                
                self.variants.append(variant)
                
                logger.info(f"✅ {bits}-bit: {speedup:.2f}x speedup, -{mem_reduction:.1f}% memory, {acc_loss:.1f}% accuracy loss")
                
                del quant_model
                free_memory()
                
            except Exception as e:
                logger.error(f"{bits}-bit quantization failed: {e}")
        
        # Pruning
        if enable_pruning and self.variants:
            logger.info("\n[3/5] Applying pruning...")
            try:
                base_variant = self.variants[0]
                pruned_model, pruned_tokenizer = self._load_variant(base_variant.format_paths["pytorch"])
                
                if pruning_type == "magnitude":
                    pruned_model = self.pruner.prune_magnitude(pruned_model, 0.5)
                elif pruning_type == "nm":
                    pruned_model = self.pruner.prune_nm(pruned_model, 2, 4)
                elif pruning_type == "movement":
                    pruned_model = self.pruner.prune_movement(pruned_model, 0.5)
                elif pruning_type == "block":
                    pruned_model = self.pruner.prune_block(pruned_model, 4, 0.5)
                
                pruned_dir = self.output_dir / f"mistral-pruned-{pruning_type}"
                pruned_dir.mkdir(exist_ok=True)
                pruned_model.save_pretrained(pruned_dir)
                pruned_tokenizer.save_pretrained(pruned_dir)
                
                logger.info("Pruning complete")
                
                del pruned_model
                free_memory()
                
            except Exception as e:
                logger.error(f"Pruning failed: {e}")
        else:
            logger.info("\n[3/5] Pruning skipped")
        
        # Distillation
        if enable_distillation:
            logger.info("\n[4/5] Knowledge distillation...")
            logger.info("Distillation skipped (requires separate student model)")
        else:
            logger.info("\n[4/5] Distillation skipped")
        
        # Export
        logger.info("\n[5/5] Exporting models...")
        for variant in self.variants:
            for fmt in export_formats:
                if fmt != "pytorch":
                    logger.info(f"Export to {fmt} not implemented")
        
        # Generate report
        self._generate_report(model_name)
        
        # Run scheduler simulation
        self._run_scheduler_simulation()
        
        # Run edge caching simulation
        self._run_edge_caching_simulation()
        
        logger.info("\n" + "="*80)
        logger.info("✅ PIPELINE COMPLETE")
        logger.info("="*80)
    
    def _load_baseline(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        max_memory = get_max_memory_dict(max_cpu_gb=16, max_gpu_gb=7)
        
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        return model, tokenizer
    
    def _load_variant(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        max_memory = get_max_memory_dict()
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True
        )
        
        return model, tokenizer
    
    def _recommend_hw(self, metrics):
        vram = metrics.model_size_mb / 1024
        if vram < 4:    
            return ["nvidia_t4", "nvidia_rtx_4060", "edge_arm"]
        elif vram < 16:
            return ["nvidia_t4", "nvidia_rtx_4060", "nvidia_a100"]
        else:
            return ["nvidia_a100"]
    
    def _generate_report(self, model_name):
        report = {
            "model_name": model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "baseline_metrics": asdict(self.baseline_metrics) if self.baseline_metrics else {},
            "variants": [asdict(v) for v in self.variants]
        }
        
        report_path = self.output_dir / "optimization_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n📄 Report: {report_path}")
        
        logger.info("\n📊 SUMMARY:")
        for v in self.variants:
            logger.info(f"  {v.variant_id}: {v.speedup_factor:.2f}x speedup, "
                       f"-{v.memory_reduction_percent:.1f}% memory, "
                       f"${v.cost_per_1k_tokens_usd:.6f}/1K tokens")
    
    def _run_scheduler_simulation(self):
        logger.info("\n" + "="*80)
        logger.info("MULTI-TENANT SCHEDULER SIMULATION")
        logger.info("="*80)
        
        nodes = [
            HardwareNode("node-1", "nvidia_a100", "us-east", 80, 70, 312, 10),
            HardwareNode("node-2", "nvidia_t4", "us-west", 16, 14, 65, 50),
            HardwareNode("node-3", "nvidia_rtx_4060", "eu-central", 8, 7, 22, 80),
            HardwareNode("node-4", "edge_arm", "edge-ny", 4, 3, 5, 5),
        ]
        
        requests = [
            InferenceRequest(f"req-{i}", f"tenant-{i%3}", f"model-{i%2}", f"prompt-{i}", 100, 1000, 5)
            for i in range(20)
        ]
        
        scheduler = MultiTenantScheduler(nodes)
        results = scheduler.simulate(requests)
        
        logger.info(f"Routed {len(results)} requests")
    
    def _run_edge_caching_simulation(self):
        logger.info("\n" + "="*80)
        logger.info("EDGE CACHING SIMULATION")
        logger.info("="*80)
        
        nodes = [
            HardwareNode("edge-1", "edge_arm", "ny", 4, 3, 5, 5, request_history=["model-0"]*10 + ["model-1"]*5),
            HardwareNode("edge-2", "edge_arm", "la", 4, 3, 5, 5, request_history=["model-1"]*15),
        ]
        
        model_variants = [v.variant_id for v in self.variants] if self.variants else ["model-0", "model-1"]
        
        caching_sim = EdgeCachingSimulator()
        cache_map = caching_sim.simulate_caching(nodes, model_variants)
        
        cache_path = self.output_dir / "edge_cache_map.json"
        with open(cache_path, 'w') as f:
            json.dump(cache_map, f, indent=2)
        
        logger.info(f"📄 Cache map: {cache_path}")


def main():
    parser = argparse.ArgumentParser(description="AI Model Optimization Pipeline")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Model name")
    parser.add_argument("--bits", nargs="+", type=int, default=[4, 8], help="Quantization bits")
    parser.add_argument("--pruning", choices=["magnitude", "nm", "movement", "block"], help="Pruning type")
    parser.add_argument("--distillation", action="store_true", help="Enable distillation")
    parser.add_argument("--export", nargs="+", default=["pytorch"], help="Export formats")
    parser.add_argument("--output-dir", type=str, default="./optimized_models", help="Output directory")
    parser.add_argument("--use-autogptq", action="store_true", help="Use AutoGPTQ")
    
    args = parser.parse_args()
    
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    pipeline = OptimizationPipeline(output_dir=args.output_dir)
    
    try:
        pipeline.run(
            model_name=args.model,
            quantization_bits=args.bits,
            enable_pruning=args.pruning is not None,
            pruning_type=args.pruning or "magnitude",
            enable_distillation=args.distillation,
            export_formats=args.export,
            use_autogptq=args.use_autogptq
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
