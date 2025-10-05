# miniGPT - GPT2(124 Million parameters) model implementation from scratch
This repository contains a comprehensive implementation of attention mechanisms for Large Language Models (LLMs), built from scratch using PyTorch. The project demonstrates the evolution from basic attention to multi-head attention and finally to the complete GPT architecture.

## 📁 Repository Structure

```
.gitignore
ch1_Data_Preprocessing.ipynb          # Data preprocessing pipeline
ch2_AttentionMechanism.ipynb         # Core attention mechanism implementation  
ch3_GPTArchitecture.ipynb            # Complete GPT model architecture
ch4_LLM_Pretraining.ipynb           # Language model pretraining
requirements.txt                      # Python dependencies
the-verdict.txt / Verdict.txt         # Sample text data
images/                              # Visualization assets
myenv/                              # Python virtual environment
```

## 🎯 Learning Path

### Chapter 1: Data Preprocessing
Data preparation and tokenization for language model training.

### Chapter 2: Attention Mechanism (`ch2_AttentionMechanism.ipynb`)
**Core concepts implemented:**

#### 1. **Basic Self-Attention**
- Simple attention calculation using dot products
- Query, Key, Value vector concepts
- Context vector computation

```python
# Basic attention calculation
attn_scores = inputs @ inputs.T
attn_weights = torch.softmax(attn_scores, dim=-1)
context_vectors = attn_weights @ inputs
```

#### 2. **Trainable Self-Attention (`SelfAttention_V1`)**
- Learnable weight matrices for Q, K, V projections
- Scaled dot-product attention with temperature scaling

```python
class SelfAttention_V1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
```

#### 3. **Causal (Masked) Attention**
- Future token masking for autoregressive generation
- Upper triangular masking using `torch.triu()`
- Efficient implementation with `-torch.inf` masking

```python
# Causal masking implementation
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
attn_scores.masked_fill_(mask.bool(), -torch.inf)
```

#### 4. **Multi-Head Attention (`MultiHeadAttention`)**
- Parallel attention computation across multiple heads
- Efficient batch matrix multiplication
- Output projection for dimension consistency

### Chapter 3: GPT Architecture (`ch3_GPTArchitecture.ipynb`)
**Complete transformer implementation:**

#### Key Components:
- **Token & Position Embeddings**: Learnable lookup tables
- **Layer Normalization**: Pre-norm transformer architecture  
- **Feed-Forward Networks**: Position-wise fully connected layers
- **Transformer Blocks**: Complete attention + FFN + residual connections
- **Output Head**: Final linear layer for next-token prediction

#### Model Configuration (GPT-2 124M):
```python
GPT_config_124M = {
    "vocab_size": 50257,      # Vocabulary size
    "context_length": 1024,   # Maximum sequence length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of transformer blocks
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query/Key/Value bias
}
```

### Chapter 4: LLM Pretraining
Model training procedures and optimization strategies.

## 🔧 Technical Implementation Details

### Attention Mechanism Evolution

1. **Basic Attention**: Simple dot-product similarity
2. **Learnable Attention**: Trainable weight matrices
3. **Scaled Attention**: Temperature scaling for stability
4. **Causal Attention**: Future masking for autoregressive tasks
5. **Multi-Head Attention**: Parallel processing for richer representations

### Key Features

- **Efficient Masking**: Uses `torch.triu()` with `-torch.inf` for numerical stability
- **Batch Processing**: Supports batched input for efficient training
- **Dropout Regularization**: Attention weight dropout for overfitting prevention
- **Modular Design**: Clean separation of components for easy understanding

### Performance Optimizations

- **Parallel Head Computation**: All attention heads computed simultaneously
- **Contiguous Memory**: `.contiguous()` for efficient tensor operations
- **Device Compatibility**: Automatic GPU/CPU device handling

## 📊 Architecture Overview

### Multi-Head Attention Pipeline
```
Input [batch, seq_len, emb_dim]
    ↓
Query/Key/Value Projections
    ↓
Reshape to [batch, n_heads, seq_len, head_dim]
    ↓
Scaled Dot-Product Attention with Causal Masking
    ↓
Concatenate Heads [batch, seq_len, emb_dim]
    ↓
Output Projection
    ↓
Output [batch, seq_len, emb_dim]
```

### Complete GPT Model Flow
```
Token IDs [batch, seq_len]
    ↓
Token Embeddings + Position Embeddings
    ↓
Dropout
    ↓
12x Transformer Blocks:
  - Multi-Head Attention
  - Residual Connection + Layer Norm
  - Feed Forward Network
  - Residual Connection + Layer Norm
    ↓
Final Layer Normalization
    ↓
Output Head (Linear)
    ↓
Logits [batch, seq_len, vocab_size]
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- tiktoken (for tokenization)
- matplotlib (for visualizations)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ShubhamBaghel309/MiniGPT.git
   cd MiniGPT
   ```

2. **Create virtual environment**
   ```bash
   python -m venv myenv
   myenv\Scripts\activate  # Windows
   # or
   source myenv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Notebooks

**Run the notebooks in order:**

1. **Data Preprocessing**: `ch1_Data_Preprocessing.ipynb`
   - Text tokenization using tiktoken
   - Data loading and preparation

2. **Attention Mechanisms**: `ch2_AttentionMechanism.ipynb`
   - Basic to advanced attention implementations
   - Visualization of attention patterns

3. **GPT Architecture**: `ch3_GPTArchitecture.ipynb`
   - Complete transformer model
   - Text generation capabilities

4. **Model Training**: `ch4_LLM_Pretraining.ipynb`
   - Training procedures and optimization

## 🧠 Key Learning Outcomes

After completing this project, you'll understand:

- **Attention Mechanisms**: From basic similarity to sophisticated multi-head attention
- **Transformer Architecture**: Complete understanding of GPT-style models
- **Causal Modeling**: Autoregressive language model design
- **Efficient Implementation**: PyTorch best practices for transformer models
- **Scalable Design**: Modular components for easy experimentation

## 📚 Core Classes Reference

### Attention Classes
- **`SelfAttention_V1`**: Basic trainable self-attention with parameter matrices
- **`SelfAttention_v2`**: Improved version using `nn.Linear` layers  
- **`CausalAttention`**: Masked attention for autoregressive tasks
- **`MultiHeadAttention`**: Parallel multi-head implementation

### Architecture Classes
- **`LayerNorm`**: Custom layer normalization implementation
- **`GELU`**: Gaussian Error Linear Unit activation
- **`FeedForward`**: Position-wise feed-forward network
- **`TransformerBlock`**: Complete transformer layer
- **`GPTModel`**: Full GPT architecture (124M parameters)

## 🎯 Technical Highlights

### Attention Implementation Features
```python
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        
        # Register causal mask as buffer for device compatibility
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
```

### Multi-Head Parallel Processing
```python
# Efficient parallel computation of all attention heads
queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

# Batch matrix multiplication for all heads simultaneously
attn_scores = queries @ keys.transpose(2, 3)
```

## 📈 Model Specifications

### GPT-2 124M Configuration
- **Parameters**: ~124 million
- **Layers**: 12 transformer blocks
- **Attention Heads**: 12 per layer
- **Embedding Dimension**: 768
- **Vocabulary Size**: 50,257 tokens
- **Context Length**: 1,024 tokens
- **Feed-Forward Hidden Size**: 3,072 (4x embedding dimension)

### Performance Metrics
- **Multi-Head Attention Parameters**: ~2.36M per layer
- **Feed-Forward Parameters**: ~4.72M per layer
- **Total Attention Parameters**: ~28.3M
- **Total Feed-Forward Parameters**: ~56.6M
- **Embedding Parameters**: ~38.6M

## 🔍 Visualizations & Analysis

The notebooks include comprehensive visualizations:
- **Attention Weight Heatmaps**: Understanding what tokens attend to each other
- **Architecture Diagrams**: Visual representation of data flow
- **Token Processing Flow**: Step-by-step tensor transformations
- **Multi-Head Attention Patterns**: Different heads learning different relationships

## 🎓 Educational Value

This implementation serves as:
- **Learning Resource**: Step-by-step building of transformer components
- **Reference Implementation**: Clean, well-documented PyTorch code
- **Experimentation Platform**: Modular design for testing modifications
- **Interview Preparation**: Demonstrates deep understanding of LLM architectures

## 🚀 Future Extensions

Potential improvements and extensions:
- **Training Pipeline**: Complete training loop with loss computation
- **Inference Optimization**: KV-caching for efficient generation
- **Model Variants**: Different transformer architectures (BERT, T5, etc.)
- **Fine-tuning**: Task-specific model adaptation
- **Quantization**: Model compression techniques

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📞 Contact

- **Author**: Shubham Baghel
- **GitHub**: [@ShubhamBaghel309](https://github.com/ShubhamBaghel309)
- **Repository**: [MiniGPT](https://github.com/ShubhamBaghel309/MiniGPT)

---

This implementation provides a solid foundation for understanding modern transformer-based language models and serves as a stepping stone for more advanced LLM architectures. The modular design and comprehensive documentation make it an excellent resource for both learning and practical implementation.
