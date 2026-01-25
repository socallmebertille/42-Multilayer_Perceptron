<div align="center" class="text-center">
  <h1>42-Multilayer_Perceptron</h1>
  
  <img alt="last-commit" src="https://img.shields.io/github/last-commit/socallmebertille/42-Multilayer_Perceptron?style=flat&amp;logo=git&amp;logoColor=white&amp;color=0080ff" class="inline-block mx-1" style="margin: 0px 2px;">
  <img alt="repo-top-language" src="https://img.shields.io/github/languages/top/socallmebertille/42-Multilayer_Perceptron?style=flat&amp;color=0080ff" class="inline-block mx-1" style="margin: 0px 2px;">
  <img alt="repo-language-count" src="https://img.shields.io/github/languages/count/socallmebertille/42-Multilayer_Perceptron?style=flat&amp;color=0080ff" class="inline-block mx-1" style="margin: 0px 2px;">
  <p><em>Built with the tools and technologies:</em></p>
  <img alt="Markdown" src="https://img.shields.io/badge/Markdown-000000.svg?style=flat&amp;logo=Markdown&amp;logoColor=white" class="inline-block mx-1" style="margin: 0px 2px;">
  <img alt="GNU%20Bash" src="https://img.shields.io/badge/GNU%20Bash-4EAA25.svg?style=flat&amp;logo=GNU-Bash&amp;logoColor=white" class="inline-block mx-1" style="margin: 0px 2px;">
  <img alt="Python" src="https://img.shields.io/badge/python-2496ED.svg?style=flat&amp;logo=python&amp;logoColor=white" class="inline-block mx-1" style="margin: 0px 2px;">
</div>

<h2>Table of Contents</h2>
<ul class="list-disc pl-4 my-0">
  <li class="my-0"><a href="#overview">Overview</a></li>
  <ul class="list-disc pl-4 my-0">
    <li class="my-0"><a href="#install-a-great-virtual-environment">Install a great virtual environment</a></li>
    <li class="my-0"><a href="#mathematical-concept">Mathematical concept</a></li>
  </ul>
  <li class="my-0"><a href="#build-a-multilayer-perceptron-mlp">Build a Multilayer Perceptron (MLP)</a>
  <ul class="list-disc pl-4 my-0">
    <li class="my-0"><a href="#usage">Usage</a></li>
  </ul>
  </li>
</ul>

<h2>Overview</h2>
<h3>Install a great virtual environment</h3>

```bash
uv venv                                         # creation
source .venv/bin/activate                       # activation

uv pip install numpy                            # installation of dependencies
```

<h3>Mathematical concept</h3>

#### Multilayer Perceptron

> **Definition:** feedforward (information flows from the input layer to the output layer only) neural network model with at least 1-2 hidden layers.

MLP stacks several perceptrons organized in **layers**, each of which:
- takes the output of the previous layer
- transforms the space
- simplifies the problem

> **Universal approximation theorem:** a combination of simple functions can approximate any complex function.

#### Standard ML Pipeline

```mermaid
graph LR
    A[📊 Raw Data] --> B[📏 Normalization]
    B --> C[✂️ Split train/validation/test]
    C --> D[✂️ Batching]
    D --> E[🎯 Forward pass = prediction]
    E --> F[📈 Loss function]
    F --> G{Threshold or Early stopping ?}
    G --> |Yes| H[✅ Best Model]
    G --> |No| I[🔄 Backpropagation = gradient]
    I --> J[🔄 Gradient descent = MAJ gradients]
```

#### Formulas

Each layer applies a weighted sum + activation :

**1. Forward Pass**

For each batch, for each layer $l$ :

$$a^{(l)} = f(z^{(l)}) = f(W^{(l)} a^{(l-1)} + b^{(l)})$$

Where:
- $l$ = layer
- $W^{(l)}$ = weight matrix
- $a^{(0)}$ = $x$ (input)
- $a^{(l-1)}$ = output of previous layer
- $b^{(l)}$ = bias
- $f$ = activation function (ReLU, sigmoid…)

> Each layer applies a weighted sum + activation

**Activation Functions:**

| Components | Sigmoid | Softmax | Linear (ReLU) |
| --- | --- | --- | --- |
| Output range | (0, 1) | (0, 1), sum to 1 | [0, +∞) |
| Use case | Binary / multi-label independent | Multi-class (mutually exclusive) | Hidden layers |
| Output structure | Single probability per neuron | Probability distribution | Non-linear activation |
| Advantages | Interpretable | Probabilistic | Simple, fast |
| Disadvantages | Saturation | Coupled to CE | Dead neurons |
| **Formula** | $\sigma(z) = \frac{1}{1 + e^{-z}}$ | $\text{softmax}(z_K) = \frac{e^{z_K}}{\sum_{j=1}^K e^{z_j}}$ | $\text{ReLU}(z) = \max(0, z)$ |

**Usage Rules:**
- **Hidden layers**: ReLU (standard), sometimes Sigmoid/Tanh
- **Output layer**:
  - Regression: Linear (no activation)
  - Binary classification: Sigmoid
  - Multi-class classification: Softmax

**2. Loss Function**

Measures the error: $L(\hat{y}, y)$

| Problem type | Loss function | Formula |
| --- | --- | --- |
| Binary classification | Binary Cross-Entropy | $\text{BCE} = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$ |
| Multi-class classification | Categorical Cross-Entropy | $\text{CCE} = -\sum_i y_i \log(\hat{y}_i)$ |
| Regression | Mean Squared Error | $\text{MSE} = \frac{1}{2m} \sum_i (\hat{y}_i - y_i)^2$ |

**3. Backpropagation**

Computes how each weight contributed to the error, by propagating gradients from output to input.

One layer: $a^{(l)} = f(z^{(l)})$

Layer error:
$$\delta^{(l)} = \frac{\partial L}{\partial z^{(l)}}$$

**Gradient Computation:**

| Element | Gradient |
| --- | --- |
| **Output layer** | $\delta^{(L)} = \hat{y} - y$ (Softmax+CE or Sigmoid+BCE) |
| **Hidden layer** | $\delta^{(l)} = (W^{(l+1)T} \delta^{(l+1)}) \odot f'(z^{(l)})$ |

**4. Gradient Descent**

| Element | Formula |
| --- | --- |
| Weight gradient | $\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} a^{(l-1)T}$ |
| Bias gradient | $\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}$ |
| Update | $W^{(l)} \leftarrow W^{(l)} - \eta \frac{\partial L}{\partial W^{(l)}}$ |

New weights = old weights - (learning rate × weight gradient)

**5. Stopping Criteria**

- Maximum epochs reached
- Loss < minimal threshold (e.g., loss < 0.001)
- **Early stopping** (validation loss plateaus)
- Gradient ≈ 0 (convergence reached)


<h2>Build a Multilayer Perceptron (MLP)</h2>
<h3>Usage</h3>

```bash
python mlp.py --help
python mlp.py --dataset data.csv --split 0.6,0.3
python mlp.py --dataset datasets/train_set.csv
python mlp.py --dataset datasets/test_set.csv --predict saved_model.npy
```
