# Code Citations

## License: Apache-2.0
https://github.com/dmlc/xgboost/blob/a99bb38bd2762e35e6a1673a0c11e09eddd8e723/doc/tutorials/model.rst

```
# Mathematical Equations & Formulas Used in SmartAgri-AI Project

## 1. RANDOM FOREST CLASSIFIER (Used for: Fertilizer, Stress, Crop Recommendation, Best Time Prediction)

### Core Formula - Decision Tree (Foundation of Random Forest):

**Information Gain (Entropy-based):**
$$H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$

Where:
- $H(S)$ = Entropy of set S
- $c$ = number of classes
- $p_i$ = proportion of class i in S

**Information Gain:**
$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$

Where:
- $IG(S, A)$ = Information gain for attribute A
- $S_v$ = subset of S where attribute A has value v

**Gini Index (Alternative splitting criterion):**
$$Gini(S) = 1 - \sum_{i=1}^{c} p_i^2$$

**Random Forest Prediction (Ensemble voting):**
$$\hat{y} = \text{argmax}_k \sum_{t=1}^{T} \mathbb{1}(f_t(x) = k)$$

Where:
- $T$ = number of trees
- $f_t(x)$ = prediction of tree t
- $k$ = class label

---

## 2. XGBOOST REGRESSOR (Used for: Yield Prediction)

### Gradient Boosting Framework:

**Objective Function:**
$$\mathcal{L}(t) = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t)}) + \sum_{k=1}^{t} \Omega(f_k)$$

Where:
- $l$ = differentiable loss function
- $y_i$ = actual value
- $\hat{y}_i^{(t)}$ = prediction at iteration t
- $\Omega(f_k)$ = regularization of tree k

**Prediction Update (Boosting):**
$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta f_t(x_i)$$

Where:
- $\eta$ = learning rate (shrinkage)
- $f_t(x_i)$ = new tree prediction

**Taylor Expansion of Loss:**
$$\mathcal{L}(t) \approx \sum_{i=1}^{n} [l(y_i, \hat{y}^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)] + \Omega(f_t)$$

Where:
- $g_i = \frac{\partial l}{\partial \hat{y}^{(t-1)}}$ = first-order gradient
- $h_i = \frac{\partial^2 l}{\partial (\hat{y}^{(t-1)})^2}$ = second-order gradient

**Gain Splitting Criterion:**
$$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$

Where:
- $G_L, G_R$ = sum of gradients in left/right child
- $H_L, H_R$ = sum of hessians in left/right child
- $\lambda$ = L2 regularization
- $\gamma$ = minimum loss reduction

---

## 3. EFFICIENTNET-B0 (Deep Learning - Fruit Disease Detection)

### Convolutional Neural Network (CNN) Architecture:

**Convolution Operation:**
$$y[m,n] = \sum_{i=0}^{k_h-1} \sum_{j=0}^{k_w-1} x[m+i, n+j] \cdot w[i,j] + b$$

Where:
- $x$ = input feature map
- $w$ = convolutional kernel
- $b$ = bias
- $k_h, k_w$ = kernel height and width

**ReLU Activation:**
$$f(x) = \max(0, x)$$

**Softmax (Output Layer - 17 classes for fruit diseases):**
$$\sigma(z)_j = \frac{e^{z_j}}{\sum_{k=1}^{17} e^{z_k}}$$

Where:
- $z$ = logits from final layer
- $j$ = class index (1 to 17)

**Categorical Cross-Entropy Loss:**
$$L = -\sum_{i=1}^{17} y_i \log(\hat{y}_i)$$

Where:
- $y_i$ = true one-hot encoded label
- $\hat{y}_i$ = predicted probability for class i

**Batch Normalization:**
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y = \gamma \hat{x} + \beta$$

Where:
- $\mu_B$ = batch mean
- $\sigma_B^2$ = batch variance
- $\gamma, \beta$ = learnable parameters
- $\epsilon$ = small constant for numerical stability

**EfficientNet Scaling (Compound Scaling):**
$$\text{depth} = \alpha^\phi, \quad \text{width} = \beta^\phi, \quad \text{resolution} = \gamma^\phi$$

Where $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ (for B0: $\phi=0$, so all = 1)

---

## 4. KERAS/TENSORFLOW MODEL (Plant Disease Detection)

### Multi-Layer Perceptron (Dense Neural Network):

**Forward Pass (Fully Connected Layer):**
$$z_j^{(l)} = \sum_{i=1}^{n^{(l-1)}} w_{ij}^{(l)} a_i^{(l-1)} + b_j^{(l)}$$
$$a_j^{(l)} = \sigma(z_j^{(l)})$$

Where:
- $z_j^{(l)}$ = pre-activation of neuron j at layer l
- $w_{ij}^{(l)}$ = weight from neuron i in layer l-1 to j in layer l
- $a_j^{(l)}$ = activation (output) of neuron j
- $\sigma$ = activation function
- $b_j^{(l)}$ = bias

**Backpropagation (Gradient Descent):**
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

$$w := w - \eta \frac{\partial L}{\partial w}$$

Where:
- $\eta$ = learning rate
- $L$ = loss function

**Adam Optimizer (Used in training):**
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$w := w - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Where:
- $m_t$ = first moment estimate (mean)
- $v_t$ = second moment estimate (uncentered variance)
- $\beta_1 = 0.9, \beta_2 = 0.999$ (typical defaults)
- $g_t$ = gradient at timestep t

---

## 5. STRESS LEVEL PREDICTION (Heuristic-based + RandomForest)

### Stress Score Calculation (Rule-based):

$$\text{Stress Score} = \sum_{i=1}^{n} w_i \cdot f_i(x_i)$$

Where factors include:

**Temperature Impact:**
$$f_{\text{temp}}(T) = \begin{cases} 2 & \text{if } T > 38°C \text{ or } T < 18°C \\ 1 & \text{if } T > 35°C \text{ or } T < 20°C \\ 0 & \text{otherwise} \end{cases}$$

**Soil Moisture Impact:**
$$f_{\text{moisture}}(M) = \begin{cases} 2 & \text{if } M < 30\% \\ 1 & \text{if } M < 40\% \\ 0 & \text{otherwise} \end{cases}$$

**Rainfall Impact:**
$$f_{\text{rainfall}}(R) = \begin{cases} 2 & \text{if } R < 20 \text{ mm} \\ 1 & \text{if } R < 40 \text{ mm} \\ 0 & \text{otherwise} \end{cases}$$

**Pest Damage Impact:**
$$f_{\text{pest}}(P) = \begin{cases} 2 & \text{if } P > 30\% \\ 1 & \text{if } P > 15\% \\ 0 & \text{otherwise} \end{cases}$$

**Wind Speed Impact:**
$$f_{\text{wind}}(W) = \begin{cases} 1 & \text{if } W > 25 \text{ km/h} \\ 0 & \text{otherwise} \end{cases}$$

**Drainage Impact:**
$$f_{\text{drainage}}(D) = \begin{cases} 1 & \text{if } D < 40 \\ 0 & \text{otherwise} \end{cases}$$

**Final Classification:**
$$\text{Stress Level} = \begin{cases} \text{High} & \text{if Score} \geq 6 \\ \text{Moderate} & \text{if } 3 \leq \text{Score} < 6 \\ \text{Low} & \text{if Score} < 3 \end{cases}$$

---

## 6. BEST TIME TO SPRAY PREDICTION (Logistic Regression via RandomForest)

### Spray Recommendation Logic:

$$\text{Spray} = \begin{cases} 1 & \text{if } (R < 2mm) \land (W < 4 \text{ km/h}) \land (H > 40\%) \land (O < 70) \\ 0 & \text{otherwise} \end{cases}$$

Where:
- $R$ = rainfall
- $W$ = wind speed
- $H$ = humidity
- $O$ = ozone level

**Logistic Function (Binary Classification):**
$$P(\text{Spray}=1|x) = \frac{1}{1 + e^{-(\beta_0 + \sum_{i} \beta_i x_i)}}$$

---

## 7. CROP
```


## License: Apache-2.0
https://github.com/dmlc/xgboost/blob/a99bb38bd2762e35e6a1673a0c11e09eddd8e723/doc/tutorials/model.rst

```
# Mathematical Equations & Formulas Used in SmartAgri-AI Project

## 1. RANDOM FOREST CLASSIFIER (Used for: Fertilizer, Stress, Crop Recommendation, Best Time Prediction)

### Core Formula - Decision Tree (Foundation of Random Forest):

**Information Gain (Entropy-based):**
$$H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$

Where:
- $H(S)$ = Entropy of set S
- $c$ = number of classes
- $p_i$ = proportion of class i in S

**Information Gain:**
$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$

Where:
- $IG(S, A)$ = Information gain for attribute A
- $S_v$ = subset of S where attribute A has value v

**Gini Index (Alternative splitting criterion):**
$$Gini(S) = 1 - \sum_{i=1}^{c} p_i^2$$

**Random Forest Prediction (Ensemble voting):**
$$\hat{y} = \text{argmax}_k \sum_{t=1}^{T} \mathbb{1}(f_t(x) = k)$$

Where:
- $T$ = number of trees
- $f_t(x)$ = prediction of tree t
- $k$ = class label

---

## 2. XGBOOST REGRESSOR (Used for: Yield Prediction)

### Gradient Boosting Framework:

**Objective Function:**
$$\mathcal{L}(t) = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t)}) + \sum_{k=1}^{t} \Omega(f_k)$$

Where:
- $l$ = differentiable loss function
- $y_i$ = actual value
- $\hat{y}_i^{(t)}$ = prediction at iteration t
- $\Omega(f_k)$ = regularization of tree k

**Prediction Update (Boosting):**
$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta f_t(x_i)$$

Where:
- $\eta$ = learning rate (shrinkage)
- $f_t(x_i)$ = new tree prediction

**Taylor Expansion of Loss:**
$$\mathcal{L}(t) \approx \sum_{i=1}^{n} [l(y_i, \hat{y}^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)] + \Omega(f_t)$$

Where:
- $g_i = \frac{\partial l}{\partial \hat{y}^{(t-1)}}$ = first-order gradient
- $h_i = \frac{\partial^2 l}{\partial (\hat{y}^{(t-1)})^2}$ = second-order gradient

**Gain Splitting Criterion:**
$$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$

Where:
- $G_L, G_R$ = sum of gradients in left/right child
- $H_L, H_R$ = sum of hessians in left/right child
- $\lambda$ = L2 regularization
- $\gamma$ = minimum loss reduction

---

## 3. EFFICIENTNET-B0 (Deep Learning - Fruit Disease Detection)

### Convolutional Neural Network (CNN) Architecture:

**Convolution Operation:**
$$y[m,n] = \sum_{i=0}^{k_h-1} \sum_{j=0}^{k_w-1} x[m+i, n+j] \cdot w[i,j] + b$$

Where:
- $x$ = input feature map
- $w$ = convolutional kernel
- $b$ = bias
- $k_h, k_w$ = kernel height and width

**ReLU Activation:**
$$f(x) = \max(0, x)$$

**Softmax (Output Layer - 17 classes for fruit diseases):**
$$\sigma(z)_j = \frac{e^{z_j}}{\sum_{k=1}^{17} e^{z_k}}$$

Where:
- $z$ = logits from final layer
- $j$ = class index (1 to 17)

**Categorical Cross-Entropy Loss:**
$$L = -\sum_{i=1}^{17} y_i \log(\hat{y}_i)$$

Where:
- $y_i$ = true one-hot encoded label
- $\hat{y}_i$ = predicted probability for class i

**Batch Normalization:**
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y = \gamma \hat{x} + \beta$$

Where:
- $\mu_B$ = batch mean
- $\sigma_B^2$ = batch variance
- $\gamma, \beta$ = learnable parameters
- $\epsilon$ = small constant for numerical stability

**EfficientNet Scaling (Compound Scaling):**
$$\text{depth} = \alpha^\phi, \quad \text{width} = \beta^\phi, \quad \text{resolution} = \gamma^\phi$$

Where $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ (for B0: $\phi=0$, so all = 1)

---

## 4. KERAS/TENSORFLOW MODEL (Plant Disease Detection)

### Multi-Layer Perceptron (Dense Neural Network):

**Forward Pass (Fully Connected Layer):**
$$z_j^{(l)} = \sum_{i=1}^{n^{(l-1)}} w_{ij}^{(l)} a_i^{(l-1)} + b_j^{(l)}$$
$$a_j^{(l)} = \sigma(z_j^{(l)})$$

Where:
- $z_j^{(l)}$ = pre-activation of neuron j at layer l
- $w_{ij}^{(l)}$ = weight from neuron i in layer l-1 to j in layer l
- $a_j^{(l)}$ = activation (output) of neuron j
- $\sigma$ = activation function
- $b_j^{(l)}$ = bias

**Backpropagation (Gradient Descent):**
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

$$w := w - \eta \frac{\partial L}{\partial w}$$

Where:
- $\eta$ = learning rate
- $L$ = loss function

**Adam Optimizer (Used in training):**
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$w := w - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Where:
- $m_t$ = first moment estimate (mean)
- $v_t$ = second moment estimate (uncentered variance)
- $\beta_1 = 0.9, \beta_2 = 0.999$ (typical defaults)
- $g_t$ = gradient at timestep t

---

## 5. STRESS LEVEL PREDICTION (Heuristic-based + RandomForest)

### Stress Score Calculation (Rule-based):

$$\text{Stress Score} = \sum_{i=1}^{n} w_i \cdot f_i(x_i)$$

Where factors include:

**Temperature Impact:**
$$f_{\text{temp}}(T) = \begin{cases} 2 & \text{if } T > 38°C \text{ or } T < 18°C \\ 1 & \text{if } T > 35°C \text{ or } T < 20°C \\ 0 & \text{otherwise} \end{cases}$$

**Soil Moisture Impact:**
$$f_{\text{moisture}}(M) = \begin{cases} 2 & \text{if } M < 30\% \\ 1 & \text{if } M < 40\% \\ 0 & \text{otherwise} \end{cases}$$

**Rainfall Impact:**
$$f_{\text{rainfall}}(R) = \begin{cases} 2 & \text{if } R < 20 \text{ mm} \\ 1 & \text{if } R < 40 \text{ mm} \\ 0 & \text{otherwise} \end{cases}$$

**Pest Damage Impact:**
$$f_{\text{pest}}(P) = \begin{cases} 2 & \text{if } P > 30\% \\ 1 & \text{if } P > 15\% \\ 0 & \text{otherwise} \end{cases}$$

**Wind Speed Impact:**
$$f_{\text{wind}}(W) = \begin{cases} 1 & \text{if } W > 25 \text{ km/h} \\ 0 & \text{otherwise} \end{cases}$$

**Drainage Impact:**
$$f_{\text{drainage}}(D) = \begin{cases} 1 & \text{if } D < 40 \\ 0 & \text{otherwise} \end{cases}$$

**Final Classification:**
$$\text{Stress Level} = \begin{cases} \text{High} & \text{if Score} \geq 6 \\ \text{Moderate} & \text{if } 3 \leq \text{Score} < 6 \\ \text{Low} & \text{if Score} < 3 \end{cases}$$

---

## 6. BEST TIME TO SPRAY PREDICTION (Logistic Regression via RandomForest)

### Spray Recommendation Logic:

$$\text{Spray} = \begin{cases} 1 & \text{if } (R < 2mm) \land (W < 4 \text{ km/h}) \land (H > 40\%) \land (O < 70) \\ 0 & \text{otherwise} \end{cases}$$

Where:
- $R$ = rainfall
- $W$ = wind speed
- $H$ = humidity
- $O$ = ozone level

**Logistic Function (Binary Classification):**
$$P(\text{Spray}=1|x) = \frac{1}{1 + e^{-(\beta_0 + \sum_{i} \beta_i x_i)}}$$

---

## 7. CROP
```


## License: unknown
https://github.com/Anirudh257/Solutions-to-Machine-Learning-Interviews-Book-By-Chip-Huyen/blob/2a8e44805953c2c19aea6714b998f40f5067ad38/Chapter5/Calculus%20and%20convex%20optimization.md

```
# Mathematical Equations & Formulas Used in SmartAgri-AI Project

## 1. RANDOM FOREST CLASSIFIER (Used for: Fertilizer, Stress, Crop Recommendation, Best Time Prediction)

### Core Formula - Decision Tree (Foundation of Random Forest):

**Information Gain (Entropy-based):**
$$H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$

Where:
- $H(S)$ = Entropy of set S
- $c$ = number of classes
- $p_i$ = proportion of class i in S

**Information Gain:**
$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$

Where:
- $IG(S, A)$ = Information gain for attribute A
- $S_v$ = subset of S where attribute A has value v

**Gini Index (Alternative splitting criterion):**
$$Gini(S) = 1 - \sum_{i=1}^{c} p_i^2$$

**Random Forest Prediction (Ensemble voting):**
$$\hat{y} = \text{argmax}_k \sum_{t=1}^{T} \mathbb{1}(f_t(x) = k)$$

Where:
- $T$ = number of trees
- $f_t(x)$ = prediction of tree t
- $k$ = class label

---

## 2. XGBOOST REGRESSOR (Used for: Yield Prediction)

### Gradient Boosting Framework:

**Objective Function:**
$$\mathcal{L}(t) = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t)}) + \sum_{k=1}^{t} \Omega(f_k)$$

Where:
- $l$ = differentiable loss function
- $y_i$ = actual value
- $\hat{y}_i^{(t)}$ = prediction at iteration t
- $\Omega(f_k)$ = regularization of tree k

**Prediction Update (Boosting):**
$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta f_t(x_i)$$

Where:
- $\eta$ = learning rate (shrinkage)
- $f_t(x_i)$ = new tree prediction

**Taylor Expansion of Loss:**
$$\mathcal{L}(t) \approx \sum_{i=1}^{n} [l(y_i, \hat{y}^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)] + \Omega(f_t)$$

Where:
- $g_i = \frac{\partial l}{\partial \hat{y}^{(t-1)}}$ = first-order gradient
- $h_i = \frac{\partial^2 l}{\partial (\hat{y}^{(t-1)})^2}$ = second-order gradient

**Gain Splitting Criterion:**
$$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$

Where:
- $G_L, G_R$ = sum of gradients in left/right child
- $H_L, H_R$ = sum of hessians in left/right child
- $\lambda$ = L2 regularization
- $\gamma$ = minimum loss reduction

---

## 3. EFFICIENTNET-B0 (Deep Learning - Fruit Disease Detection)

### Convolutional Neural Network (CNN) Architecture:

**Convolution Operation:**
$$y[m,n] = \sum_{i=0}^{k_h-1} \sum_{j=0}^{k_w-1} x[m+i, n+j] \cdot w[i,j] + b$$

Where:
- $x$ = input feature map
- $w$ = convolutional kernel
- $b$ = bias
- $k_h, k_w$ = kernel height and width

**ReLU Activation:**
$$f(x) = \max(0, x)$$

**Softmax (Output Layer - 17 classes for fruit diseases):**
$$\sigma(z)_j = \frac{e^{z_j}}{\sum_{k=1}^{17} e^{z_k}}$$

Where:
- $z$ = logits from final layer
- $j$ = class index (1 to 17)

**Categorical Cross-Entropy Loss:**
$$L = -\sum_{i=1}^{17} y_i \log(\hat{y}_i)$$

Where:
- $y_i$ = true one-hot encoded label
- $\hat{y}_i$ = predicted probability for class i

**Batch Normalization:**
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y = \gamma \hat{x} + \beta$$

Where:
- $\mu_B$ = batch mean
- $\sigma_B^2$ = batch variance
- $\gamma, \beta$ = learnable parameters
- $\epsilon$ = small constant for numerical stability

**EfficientNet Scaling (Compound Scaling):**
$$\text{depth} = \alpha^\phi, \quad \text{width} = \beta^\phi, \quad \text{resolution} = \gamma^\phi$$

Where $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ (for B0: $\phi=0$, so all = 1)

---

## 4. KERAS/TENSORFLOW MODEL (Plant Disease Detection)

### Multi-Layer Perceptron (Dense Neural Network):

**Forward Pass (Fully Connected Layer):**
$$z_j^{(l)} = \sum_{i=1}^{n^{(l-1)}} w_{ij}^{(l)} a_i^{(l-1)} + b_j^{(l)}$$
$$a_j^{(l)} = \sigma(z_j^{(l)})$$

Where:
- $z_j^{(l)}$ = pre-activation of neuron j at layer l
- $w_{ij}^{(l)}$ = weight from neuron i in layer l-1 to j in layer l
- $a_j^{(l)}$ = activation (output) of neuron j
- $\sigma$ = activation function
- $b_j^{(l)}$ = bias

**Backpropagation (Gradient Descent):**
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

$$w := w - \eta \frac{\partial L}{\partial w}$$

Where:
- $\eta$ = learning rate
- $L$ = loss function

**Adam Optimizer (Used in training):**
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$w := w - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Where:
- $m_t$ = first moment estimate (mean)
- $v_t$ = second moment estimate (uncentered variance)
- $\beta_1 = 0.9, \beta_2 = 0.999$ (typical defaults)
- $g_t$ = gradient at timestep t

---

## 5. STRESS LEVEL PREDICTION (Heuristic-based + RandomForest)

### Stress Score Calculation (Rule-based):

$$\text{Stress Score} = \sum_{i=1}^{n} w_i \cdot f_i(x_i)$$

Where factors include:

**Temperature Impact:**
$$f_{\text{temp}}(T) = \begin{cases} 2 & \text{if } T > 38°C \text{ or } T < 18°C \\ 1 & \text{if } T > 35°C \text{ or } T < 20°C \\ 0 & \text{otherwise} \end{cases}$$

**Soil Moisture Impact:**
$$f_{\text{moisture}}(M) = \begin{cases} 2 & \text{if } M < 30\% \\ 1 & \text{if } M < 40\% \\ 0 & \text{otherwise} \end{cases}$$

**Rainfall Impact:**
$$f_{\text{rainfall}}(R) = \begin{cases} 2 & \text{if } R < 20 \text{ mm} \\ 1 & \text{if } R < 40 \text{ mm} \\ 0 & \text{otherwise} \end{cases}$$

**Pest Damage Impact:**
$$f_{\text{pest}}(P) = \begin{cases} 2 & \text{if } P > 30\% \\ 1 & \text{if } P > 15\% \\ 0 & \text{otherwise} \end{cases}$$

**Wind Speed Impact:**
$$f_{\text{wind}}(W) = \begin{cases} 1 & \text{if } W > 25 \text{ km/h} \\ 0 & \text{otherwise} \end{cases}$$

**Drainage Impact:**
$$f_{\text{drainage}}(D) = \begin{cases} 1 & \text{if } D < 40 \\ 0 & \text{otherwise} \end{cases}$$

**Final Classification:**
$$\text{Stress Level} = \begin{cases} \text{High} & \text{if Score} \geq 6 \\ \text{Moderate} & \text{if } 3 \leq \text{Score} < 6 \\ \text{Low} & \text{if Score} < 3 \end{cases}$$

---

## 6. BEST TIME TO SPRAY PREDICTION (Logistic Regression via RandomForest)

### Spray Recommendation Logic:

$$\text{Spray} = \begin{cases} 1 & \text{if } (R < 2mm) \land (W < 4 \text{ km/h}) \land (H > 40\%) \land (O < 70) \\ 0 & \text{otherwise} \end{cases}$$

Where:
- $R$ = rainfall
- $W$ = wind speed
- $H$ = humidity
- $O$ = ozone level

**Logistic Function (Binary Classification):**
$$P(\text{Spray}=1|x) = \frac{1}{1 + e^{-(\beta_0 + \sum_{i} \beta_i x_i)}}$$

---

## 7. CROP RECOMMENDATION (RandomForest Multi-class)

### Multinomial Classification:

**For each crop candidate:**
$$P(\text{Crop}_j | N, P, K, T, H, pH, R, O) = \frac{\text{number of trees voting for crop}_j}{T}$$

Where input features are:
- $N$ = Nitrogen level
- $P$ = Phosphorus level
- $K$ = Potassium level
- $T$ = Temperature
- $H$ = Humidity
- $pH$ = Soil pH
- $R$ = Rainfall
- $O$ = Ozone

**Final Prediction:**
$$\hat{y} = \underset{j}{\text{argmax }} P(\text{Crop}_j | \text{features})$$

---

## 8. FERTILIZER RECOMMENDATION (RandomForest Multi-class - 7 classes)

### Feature Space:

Input features processed with StandardScaler:
$$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$$

**7 Fertilizer Classes** predicted via ensemble voting

---

## 9. YIELD PREDICTION (XGBoost Regression)

### Mean Squared Error (MSE) Loss:

$$L_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### Root Mean Squared Error:

$$\text{RMSE} = \sqrt{MSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

### Mean Absolute Error:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### R² Score (Coefficient of Determination):

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

Where $\bar{y}$ = mean of actual values

---

## 10. MODEL EVALUATION METRICS (All Models)

### Accuracy (Classification):

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

### Precision:

$$\text{Precision} = \frac{TP}{TP + FP}$$

### Recall (Sensitivity):

$$\text{Recall} = \frac{TP}{TP + FN}$$

### F1-Score:

$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Confusion Matrix:

$$CM = \begin{bmatrix} TN & FP \\ FN & TP \end{bmatrix}$$

### ROC-AUC (Area Under Curve):

$$\text{AUC} = \int_0^1 \text{TPR}(t) \, d(\text{FPR}(t))$$

Where:
- $\text{TPR} = \frac{TP}{TP + FN}$ (True Positive Rate)
- $\text{FPR} = \frac{FP}{FP + TN}$ (False Positive Rate)

### Precision-Recall AUC:

$$\text{PR-AUC} = \int_0^1 \text{Precision}(r) \, dr$$

---

## 11. TRANSFER LEARNING (EfficientNet-B0)

### Fine-tuning Loss:

$$L_{\text{total}} = L_{\text{classification}} + \lambda \sum_{\text{pretrained}} \|w - w_0\|^2$$

Where:
- $w$ = new weights
- $w_0$ = pretrained ImageNet weights
- $\lambda$ = regularization strength

---

## 12. DATA PREPROCESSING FORMULAS

### Min-Max Normalization:

$$x_{\text{norm}} = \frac{x - \min(x)}{\max(x) - \min(x)}$$

### Z-Score Standardization:

$$z = \frac{x - \mu}{\sigma}$$

### One-Hot Encoding:

$$e_i = \begin{cases} 1 & \text{if } \text{category} = i \\ 0 & \text{otherwise} \end{cases}$$

### Label Encoding:

$$\text{class}_i \rightarrow \text{integer}_i \quad (0, 1, 2, \ldots, C-1)$$

---

## Summary Table:

| Algorithm | Type | Task | Loss Function | Key Metric |
|-----------|------|------|---------------|-----------|
| RandomForest | Classification | Fertilizer, Stress, Crop, Best Time | Entropy/Gini | Accuracy |
| XGBoost | Regression | Yield | MSE | RMSE, MAE, R² |
| EfficientNet-B0 | Deep Learning |
```


## License: unknown
https://github.com/Anirudh257/Solutions-to-Machine-Learning-Interviews-Book-By-Chip-Huyen/blob/2a8e44805953c2c19aea6714b998f40f5067ad38/Chapter5/Calculus%20and%20convex%20optimization.md

```
# Mathematical Equations & Formulas Used in SmartAgri-AI Project

## 1. RANDOM FOREST CLASSIFIER (Used for: Fertilizer, Stress, Crop Recommendation, Best Time Prediction)

### Core Formula - Decision Tree (Foundation of Random Forest):

**Information Gain (Entropy-based):**
$$H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$

Where:
- $H(S)$ = Entropy of set S
- $c$ = number of classes
- $p_i$ = proportion of class i in S

**Information Gain:**
$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$

Where:
- $IG(S, A)$ = Information gain for attribute A
- $S_v$ = subset of S where attribute A has value v

**Gini Index (Alternative splitting criterion):**
$$Gini(S) = 1 - \sum_{i=1}^{c} p_i^2$$

**Random Forest Prediction (Ensemble voting):**
$$\hat{y} = \text{argmax}_k \sum_{t=1}^{T} \mathbb{1}(f_t(x) = k)$$

Where:
- $T$ = number of trees
- $f_t(x)$ = prediction of tree t
- $k$ = class label

---

## 2. XGBOOST REGRESSOR (Used for: Yield Prediction)

### Gradient Boosting Framework:

**Objective Function:**
$$\mathcal{L}(t) = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t)}) + \sum_{k=1}^{t} \Omega(f_k)$$

Where:
- $l$ = differentiable loss function
- $y_i$ = actual value
- $\hat{y}_i^{(t)}$ = prediction at iteration t
- $\Omega(f_k)$ = regularization of tree k

**Prediction Update (Boosting):**
$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta f_t(x_i)$$

Where:
- $\eta$ = learning rate (shrinkage)
- $f_t(x_i)$ = new tree prediction

**Taylor Expansion of Loss:**
$$\mathcal{L}(t) \approx \sum_{i=1}^{n} [l(y_i, \hat{y}^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)] + \Omega(f_t)$$

Where:
- $g_i = \frac{\partial l}{\partial \hat{y}^{(t-1)}}$ = first-order gradient
- $h_i = \frac{\partial^2 l}{\partial (\hat{y}^{(t-1)})^2}$ = second-order gradient

**Gain Splitting Criterion:**
$$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$

Where:
- $G_L, G_R$ = sum of gradients in left/right child
- $H_L, H_R$ = sum of hessians in left/right child
- $\lambda$ = L2 regularization
- $\gamma$ = minimum loss reduction

---

## 3. EFFICIENTNET-B0 (Deep Learning - Fruit Disease Detection)

### Convolutional Neural Network (CNN) Architecture:

**Convolution Operation:**
$$y[m,n] = \sum_{i=0}^{k_h-1} \sum_{j=0}^{k_w-1} x[m+i, n+j] \cdot w[i,j] + b$$

Where:
- $x$ = input feature map
- $w$ = convolutional kernel
- $b$ = bias
- $k_h, k_w$ = kernel height and width

**ReLU Activation:**
$$f(x) = \max(0, x)$$

**Softmax (Output Layer - 17 classes for fruit diseases):**
$$\sigma(z)_j = \frac{e^{z_j}}{\sum_{k=1}^{17} e^{z_k}}$$

Where:
- $z$ = logits from final layer
- $j$ = class index (1 to 17)

**Categorical Cross-Entropy Loss:**
$$L = -\sum_{i=1}^{17} y_i \log(\hat{y}_i)$$

Where:
- $y_i$ = true one-hot encoded label
- $\hat{y}_i$ = predicted probability for class i

**Batch Normalization:**
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y = \gamma \hat{x} + \beta$$

Where:
- $\mu_B$ = batch mean
- $\sigma_B^2$ = batch variance
- $\gamma, \beta$ = learnable parameters
- $\epsilon$ = small constant for numerical stability

**EfficientNet Scaling (Compound Scaling):**
$$\text{depth} = \alpha^\phi, \quad \text{width} = \beta^\phi, \quad \text{resolution} = \gamma^\phi$$

Where $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ (for B0: $\phi=0$, so all = 1)

---

## 4. KERAS/TENSORFLOW MODEL (Plant Disease Detection)

### Multi-Layer Perceptron (Dense Neural Network):

**Forward Pass (Fully Connected Layer):**
$$z_j^{(l)} = \sum_{i=1}^{n^{(l-1)}} w_{ij}^{(l)} a_i^{(l-1)} + b_j^{(l)}$$
$$a_j^{(l)} = \sigma(z_j^{(l)})$$

Where:
- $z_j^{(l)}$ = pre-activation of neuron j at layer l
- $w_{ij}^{(l)}$ = weight from neuron i in layer l-1 to j in layer l
- $a_j^{(l)}$ = activation (output) of neuron j
- $\sigma$ = activation function
- $b_j^{(l)}$ = bias

**Backpropagation (Gradient Descent):**
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

$$w := w - \eta \frac{\partial L}{\partial w}$$

Where:
- $\eta$ = learning rate
- $L$ = loss function

**Adam Optimizer (Used in training):**
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$w := w - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Where:
- $m_t$ = first moment estimate (mean)
- $v_t$ = second moment estimate (uncentered variance)
- $\beta_1 = 0.9, \beta_2 = 0.999$ (typical defaults)
- $g_t$ = gradient at timestep t

---

## 5. STRESS LEVEL PREDICTION (Heuristic-based + RandomForest)

### Stress Score Calculation (Rule-based):

$$\text{Stress Score} = \sum_{i=1}^{n} w_i \cdot f_i(x_i)$$

Where factors include:

**Temperature Impact:**
$$f_{\text{temp}}(T) = \begin{cases} 2 & \text{if } T > 38°C \text{ or } T < 18°C \\ 1 & \text{if } T > 35°C \text{ or } T < 20°C \\ 0 & \text{otherwise} \end{cases}$$

**Soil Moisture Impact:**
$$f_{\text{moisture}}(M) = \begin{cases} 2 & \text{if } M < 30\% \\ 1 & \text{if } M < 40\% \\ 0 & \text{otherwise} \end{cases}$$

**Rainfall Impact:**
$$f_{\text{rainfall}}(R) = \begin{cases} 2 & \text{if } R < 20 \text{ mm} \\ 1 & \text{if } R < 40 \text{ mm} \\ 0 & \text{otherwise} \end{cases}$$

**Pest Damage Impact:**
$$f_{\text{pest}}(P) = \begin{cases} 2 & \text{if } P > 30\% \\ 1 & \text{if } P > 15\% \\ 0 & \text{otherwise} \end{cases}$$

**Wind Speed Impact:**
$$f_{\text{wind}}(W) = \begin{cases} 1 & \text{if } W > 25 \text{ km/h} \\ 0 & \text{otherwise} \end{cases}$$

**Drainage Impact:**
$$f_{\text{drainage}}(D) = \begin{cases} 1 & \text{if } D < 40 \\ 0 & \text{otherwise} \end{cases}$$

**Final Classification:**
$$\text{Stress Level} = \begin{cases} \text{High} & \text{if Score} \geq 6 \\ \text{Moderate} & \text{if } 3 \leq \text{Score} < 6 \\ \text{Low} & \text{if Score} < 3 \end{cases}$$

---

## 6. BEST TIME TO SPRAY PREDICTION (Logistic Regression via RandomForest)

### Spray Recommendation Logic:

$$\text{Spray} = \begin{cases} 1 & \text{if } (R < 2mm) \land (W < 4 \text{ km/h}) \land (H > 40\%) \land (O < 70) \\ 0 & \text{otherwise} \end{cases}$$

Where:
- $R$ = rainfall
- $W$ = wind speed
- $H$ = humidity
- $O$ = ozone level

**Logistic Function (Binary Classification):**
$$P(\text{Spray}=1|x) = \frac{1}{1 + e^{-(\beta_0 + \sum_{i} \beta_i x_i)}}$$

---

## 7. CROP RECOMMENDATION (RandomForest Multi-class)

### Multinomial Classification:

**For each crop candidate:**
$$P(\text{Crop}_j | N, P, K, T, H, pH, R, O) = \frac{\text{number of trees voting for crop}_j}{T}$$

Where input features are:
- $N$ = Nitrogen level
- $P$ = Phosphorus level
- $K$ = Potassium level
- $T$ = Temperature
- $H$ = Humidity
- $pH$ = Soil pH
- $R$ = Rainfall
- $O$ = Ozone

**Final Prediction:**
$$\hat{y} = \underset{j}{\text{argmax }} P(\text{Crop}_j | \text{features})$$

---

## 8. FERTILIZER RECOMMENDATION (RandomForest Multi-class - 7 classes)

### Feature Space:

Input features processed with StandardScaler:
$$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$$

**7 Fertilizer Classes** predicted via ensemble voting

---

## 9. YIELD PREDICTION (XGBoost Regression)

### Mean Squared Error (MSE) Loss:

$$L_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### Root Mean Squared Error:

$$\text{RMSE} = \sqrt{MSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

### Mean Absolute Error:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### R² Score (Coefficient of Determination):

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

Where $\bar{y}$ = mean of actual values

---

## 10. MODEL EVALUATION METRICS (All Models)

### Accuracy (Classification):

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

### Precision:

$$\text{Precision} = \frac{TP}{TP + FP}$$

### Recall (Sensitivity):

$$\text{Recall} = \frac{TP}{TP + FN}$$

### F1-Score:

$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Confusion Matrix:

$$CM = \begin{bmatrix} TN & FP \\ FN & TP \end{bmatrix}$$

### ROC-AUC (Area Under Curve):

$$\text{AUC} = \int_0^1 \text{TPR}(t) \, d(\text{FPR}(t))$$

Where:
- $\text{TPR} = \frac{TP}{TP + FN}$ (True Positive Rate)
- $\text{FPR} = \frac{FP}{FP + TN}$ (False Positive Rate)

### Precision-Recall AUC:

$$\text{PR-AUC} = \int_0^1 \text{Precision}(r) \, dr$$

---

## 11. TRANSFER LEARNING (EfficientNet-B0)

### Fine-tuning Loss:

$$L_{\text{total}} = L_{\text{classification}} + \lambda \sum_{\text{pretrained}} \|w - w_0\|^2$$

Where:
- $w$ = new weights
- $w_0$ = pretrained ImageNet weights
- $\lambda$ = regularization strength

---

## 12. DATA PREPROCESSING FORMULAS

### Min-Max Normalization:

$$x_{\text{norm}} = \frac{x - \min(x)}{\max(x) - \min(x)}$$

### Z-Score Standardization:

$$z = \frac{x - \mu}{\sigma}$$

### One-Hot Encoding:

$$e_i = \begin{cases} 1 & \text{if } \text{category} = i \\ 0 & \text{otherwise} \end{cases}$$

### Label Encoding:

$$\text{class}_i \rightarrow \text{integer}_i \quad (0, 1, 2, \ldots, C-1)$$

---

## Summary Table:

| Algorithm | Type | Task | Loss Function | Key Metric |
|-----------|------|------|---------------|-----------|
| RandomForest | Classification | Fertilizer, Stress, Crop, Best Time | Entropy/Gini | Accuracy |
| XGBoost | Regression | Yield | MSE | RMSE, MAE, R² |
| EfficientNet-B0 | Deep Learning |
```

