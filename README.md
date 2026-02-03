# penalized_ll_and_weight_optims_in_finance
### Preliminaries
This rep is a personal project, for an implementation of a penalized log likelihood and weight optimizations in finance and diffusion models. The main idea is inspired by Zhang,  Leung, and Aravkin (2020). The complete top-to-bottom code is via the the .ipynb  [Penalized_LL_and_Diffusion_Models_for_Finance.ipynb](Penalized_LL_and_Diffusion_Models_for_Finance.ipynb) file of We start with an asset price series stack as $S \in \mathbb{R}^{(T+1)\times m}$, and a linear portofolio of $x_t = S_t^\top w$.The original paper uses an Ornstein-Uhlenbeck process (OU) in which the equation is:
```math
dx_t = \mu(\theta - x_t)\,dt + \sigma\,dB_t ,
```
where $\mu$ is the mean reversion speed, $\theta$ is long-run equilibirum, $\sigma$ is diffusion volatility. The discrete version has for
```math
x_t = c\,x_{t-1} + (1-c)\theta + \varepsilon_t \quad,\quad \varepsilon_t \sim \mathcal{N}(0, a),
```
with reparameterization according to the paper:
```math
c = e^{-\mu\Delta t}, \qquad a = \frac{\sigma^2(1-e^{-2\mu\Delta t})}{2\mu} \approx \sigma^2\Delta t
```
The paper defines:
```math
A(c) = S_{1:T} - c S_{0:T-1}
```
So that:
```math
\mathcal{L}(w,a,c,\theta) = \frac{1}{2}\ln a + \frac{1}{2Ta}\left\| A(c)w - \theta(1-c)\mathbf{1} \right\|^2
```
The portofolio is normalized with $\|w\|_1 = 1$, by which enforces self-financing upon long/short portofolios. The sparsity or $l_0$ constraint has also condition of $\|w\|_0 \le \eta$. In the paper, a penalty constant is added via $\gamma c$, since when $c$ decreases, $\mu$ increases as it explicitly trades the likelihood fit for speed of mean reversions. The final optimization problem becomes:
```math
\min_{a,c,\theta,w} \; \frac{1}{2} \ln a + \frac{\|A(c)w - \theta(1-c)\mathbf{1}\|^2}{2Ta} + \gamma c \quad \text{s.t} \quad \|w\|_1 = 1, \|w\|_0 \le \eta
```
The paper also eliminates nuisance parameters via partial minimization:
```math
f_3(w) = \min_{a,c,\theta} \mathcal{L}(w,a,c,\theta)
```
Basically the flow in a nuthsell is like:
```math
\arg \min_\theta f(\cdot) \rightarrow \arg \min_{c, \theta} f(\cdot) \rightarrow \arg \min_{a, c, \theta} f(\cdot) \rightarrow f_3(w)
```
The result is thus $\min_{w \in \mathcal{W}} f_3(w)$ where $\mathcal{W}= \left\\{ \|w\|_1=1,\|w\|_0\le\eta \right\\}$. The paper's algorithm core is via:
```math
\Pi_{\mathcal{W}}(x) = \text{sign}(x) \odot \Pi_{\Delta_1 \cap \|·\|_0 \le \eta}(|x|)
```
The algorithm starts by sorting the weights , keep the top $\eta$, project into the simplex, and restore the signs. The clean code is in [test_with_yahoo_finance](clean_code_in_pycharm/test_with_yahoo_finance.py), where i applied it to Indonesian stocks. The figure below shows the diagnostics <img src="clean_code_in_pycharm/Figure_1.png" alt="Project Screenshot" width="1000" />
### Extension - other diffusion models and interest rate
Interest rate model diffusions are also included, in which it is the Cox-Ingersoll-Ross, Black-Kariniski, Black-Derman-Toy, and Hull-White (abbreviated usually as CIR, BK, BDT, and HW). 
