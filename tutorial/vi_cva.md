# VI-CVA
---
VI-CVA here denotes the CVA model trained by VI, and the whole training process is mainly established according to the reference:
Yu J, Ye L, Zhou L, et al. Dynamic process monitoring based on variational Bayesian canonical variate analysis[J]. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2021, 52(4): 2412-2422.

Wishart distribution $\mathcal{W}$ is used as the prior of the precision in Multivariate Gaussian.

---
## Model Structure
**input：**

$\bold{x}_{t}\in \R^{M\times 1}$

$\bold{P}^t = [\bold{x}_{t-1}^T,\bold{x}_{t-2}^T,...,\bold{x}_{t-l}^T]^T\in \R^{Ml \times 1}$

$\bold{F}^t = [\bold{x}_{t}^T,\bold{x}_{t+1}^T,...,\bold{x}_{t+s}^T]^T\in \R^{M(s+1) \times 1}$

$\bold{P}^t = [p_{1}^t,p_{2}^t,...,p_{Ml}^t]^T\in \R^{Ml \times 1}$

$\bold{F}^t = [f_{1}^t,f_{2}^t,...,f_{M(s+1)}^t]^T\in \R^{M(s+1)\times 1}$


**model mapping：**

$\bold{P}=\bold{W}^T\bold{z}+\epsilon$

$\bold{F}=\bold{H}^T\bold{z}+\delta$

$\bold{z}_{t}\in \R^{D\times 1}$

$\mathbf{W}^T\in \R^{Ml\times D}$

$\mathbf{H}^T \in \R^{M(s+1)\times D}$



$
\mathbf{W}^T= \begin{pmatrix}
-\mathbf{W}_1^T- \\
-\mathbf{W}_2^T-  \\
\vdots \\
-\mathbf{W}_{Ml}^T- 
\end{pmatrix}  =
\begin{pmatrix}
\mathbf{W}_1,
\mathbf{W}_2,
\dots,
\mathbf{W}_{Ml}
\end{pmatrix}, \mathbf{W}_m\in \R^{D \times 1}
$


$
\mathbf{H}^T = \begin{pmatrix}
-\mathbf{H}_1^T- \\
-\mathbf{H}_2^T-  \\
\vdots \\
-\mathbf{H}_{M(s+1)}^T- 
\end{pmatrix} =
\begin{pmatrix}
\mathbf{H}_1,
\mathbf{H}_2,
\dots,
\mathbf{H}_{M(s+1)}
\end{pmatrix} , \mathbf{H}_m\in \R^{D \times 1}
$


**prior distributions：**

$\mathcal{P}(\bold{z})=\prod \limits_{t=1}^N \mathcal{N}(\bold{z}_t|0,\bold{I})$

$\mathcal{P}(\mathbf{\epsilon})=\prod \limits_{m=1}^{Ml} \mathcal{N}(\epsilon_m|0,\bold{\tau}_m^{-1})$; $\mathcal{P}(\mathbf{\delta})=\prod \limits_{m=1}^{M(s+1)} \mathcal{N}(\delta_m|0,\bold{\psi}_m^{-1})$

$\mathcal{P}(\mathbf{\tau})=\prod \limits_{m=1}^{Ml} \mathcal{G}(\tau_m|j_m^\tau,k_m^\tau)$; $\mathcal{P}(\mathbf{\psi})=\prod \limits_{m=1}^{M(s+1)} \mathcal{G}(\psi_m|j_m^\psi,k_m^\psi)$

$\mathcal{P}(\mathbf{\alpha})=\prod \limits_{m=1}^{Ml}  \mathcal{W}(\alpha_m|a_m^\alpha,\mathbf{B}_m^\alpha)$; $\mathcal{P}(\mathbf{\beta})=\prod \limits_{m=1}^{M(s+1)} \mathcal{W}(\beta_m|a_m^\beta,\mathbf{B}_m^\beta)$

$\mathcal{P}(\mathbf{W}|\mathbf{\alpha})=\prod \limits_{m=1}^{Ml} \mathcal{N}(\mathbf{W}_m|0,\alpha_m^{-1})$; $\mathcal{P}(\mathbf{H}|\mathbf{\beta})=\prod \limits_{m=1}^{M(s+1)} \mathcal{N}(\mathbf{H}_m|0,\beta_m^{-1})$

$\mathcal{P}(\mathbf{P}|\mathbf{z}, \mathbf{W}, \mathbf{\alpha}, \mathbf{\tau})=\prod \limits_{t=1}^{N} \prod \limits_{m=1}^{Ml} \mathcal{N}(p_m^t|\mathbf{W}_m^T\mathbf{z}_t,\tau_m^{-1})$

$\mathcal{P}(\mathbf{F}|\mathbf{z}, \mathbf{H}, \mathbf{\beta}, \mathbf{\psi})=\prod \limits_{t=1}^{N} \prod \limits_{m=1}^{M(s+1)} \mathcal{N}(f_m^t|\mathbf{H}_m^T\mathbf{z}_t,\psi_m^{-1})$


**variational distributions：**

$\mathcal{Q}(\mathbf{z}) = \prod \limits_{t=1}^{N} \mathcal{N}(\mathbf{z}_t|\mu_t^z, (\Lambda_t^z)^{-1} )$

$\mathcal{Q}(\mathbf{W}) = \prod \limits_{m=1}^{Ml} \mathcal{N}(\mathbf{W}_m|\mu_m^W,  (\Lambda_m^W)^{-1} )$; $\mathcal{Q}(\mathbf{H}) = \prod \limits_{m=1}^{M(s+1)} \mathcal{N}(\mathbf{H}_m|\mu_m^H,  (\Lambda_m^H)^{-1} )$

$\mathcal{Q}(\mathbf{\tau}) = \prod \limits_{m=1}^{Ml} \mathcal{G}(\mathbf{\tau}_m|\lambda_m^{\tau}, \nu_m^{\tau})$; $\mathcal{Q}(\mathbf{\psi}) = \prod \limits_{m=1}^{M(s+1)} \mathcal{G}(\mathbf{\psi}_m|\lambda_m^{\psi}, \nu_m^{\psi})$

$\mathcal{Q}(\mathbf{\alpha}) = \prod \limits_{m=1}^{Ml} \mathcal{W}(\mathbf{\alpha}_m|\nu_m^{\alpha}, \mathbf{V}_m^{\alpha})$; $\mathcal{Q}(\mathbf{\beta}) = \prod \limits_{m=1}^{M(s+1)} \mathcal{W}(\mathbf{\beta}_m| \nu_m^{\beta},\mathbf{V}_m^{\beta})$

---
## Upate Strategy

### joint distribution

$$
\mathcal{P}(\mathbf{P}, \mathbf{F}, \mathbf{H}, \mathbf{W}, \mathbf{z}, \mathbf{\tau}, \mathbf{\psi}, \mathbf{\alpha}, \mathbf{\beta}) \\
= \mathcal{P}(\mathbf{P}| \mathbf{W}, \mathbf{z}, \mathbf{\tau})\mathcal{P}(\mathbf{W}| \mathbf{\alpha})\mathcal{P}(\mathbf{\alpha})\mathcal{P}(\mathbf{\tau})\\
\mathcal{P}(\mathbf{F}| \mathbf{H}, \mathbf{z}, \mathbf{\psi}) 
\mathcal{P}(\mathbf{H}| \mathbf{\beta}) \mathcal{P}(\mathbf{\beta}) \mathcal{P}(\mathbf{\psi}) \mathcal{P}(\mathbf{z})
$$

### mean-field distribution

$$
\mathcal{Q}( \mathbf{H}, \mathbf{W}, \mathbf{z}, \mathbf{\tau}, \mathbf{\psi}, \mathbf{\alpha}, \mathbf{\beta}) \\
= \mathcal{Q}(\mathbf{W})\mathcal{Q}(\mathbf{\alpha})\mathcal{Q}(\mathbf{\tau})\mathcal{Q}(\mathbf{H}) \mathcal{Q}(\mathbf{\beta}) \mathcal{Q}(\mathbf{\psi}) \mathcal{Q}(\mathbf{z})
$$


<!-- 
### ELBO计算

$$ 
ELBO = \int\mathcal{Q}(...)ln{\frac{\mathcal{P}(...)}{\mathcal{Q}(...)}}\\=\int\mathcal{Q}(...)ln{{\mathcal{P}(...)}}-{\mathcal{Q}(...)}ln{\mathcal{Q}(...)}
$$

$
{\int\mathcal{Q}(...)}ln{\mathcal{Q}(...)}=\sum_{\xi_i\in \Psi}\int{\mathcal{Q}(\xi_i)}ln{\mathcal{Q}(\xi_i) }
$

$
{\int\mathcal{Q}(...)}ln{\mathcal{P}(...)}\\
=\int{\mathcal{Q}(...)}ln\mathcal{P}(\mathbf{P}| \mathbf{W}, \mathbf{z}, \mathbf{\tau})  ->L_0\\
+\int{\mathcal{Q}(...)}ln\mathcal{P}(\mathbf{F}| \mathbf{H}, \mathbf{z}, \mathbf{\psi})  ->L_1\\
+\int{\mathcal{Q}(...)}ln\mathcal{P}(\mathbf{W}| \mathbf{\alpha})  ->L_2\\
+\int{\mathcal{Q}(...)}ln\mathcal{P}(\mathbf{H}| \mathbf{\beta})  ->L_3\\
+\int{\mathcal{Q}(\alpha)}ln\mathcal{P}( \mathbf{\alpha})  ->L_4\\
+\int{\mathcal{Q}(\beta)}ln\mathcal{P}( \mathbf{\beta})  ->L_5\\
+\int{\mathcal{Q}(\mathbf{\tau})}ln\mathcal{P}(\mathbf{\tau})  ->L_6\\
+\int{\mathcal{Q}(\mathbf{\psi})}ln\mathcal{P}(\mathbf{\psi})  ->L_7\\
$

无视了一些常数项

$
L_0=\sum_{t=1}^N\sum_{m=1}^{Ml}\{\mathbb{E}_{\tau_m}(ln\tau_m)-\mathbb{E}_{\tau_m}(\tau_m)\mathbb{E}_{\mathbf{W}_m}(\mathbf{W}_m^T\mathbf{W}_m)\mathbb{E}_{\mathbf{z}_t}(\mathbf{z}_t^T\mathbf{z}_t)-(p_m^t)^T(p_m^t)\mathbb{E}_{\tau_m}(\tau_m)+2(p_m^t)^T\mathbb{E}_{\tau_m}(\tau_m)\mathbb{E}_{\mathbf{W}_m}(\mathbf{W}_m^T)\mathbb{E}_{\mathbf{z}_t}(\mathbf{z}_t)\}
$

$
L_1=\sum_{t=1}^N\sum_{m=1}^{M(s+1)}\{\mathbb{E}_{\psi_m}(ln\psi_m)-\mathbb{E}_{\psi_m}(\psi_m)\mathbb{E}_{\mathbf{H}_m}(\mathbf{H}_m^T\mathbf{H}_m)\mathbb{E}_{\mathbf{z}_t}(\mathbf{z}_t^T\mathbf{z}_t)-(f_m^t)^T(f_m^t)\mathbb{E}_{\psi_m}(\psi_m)+2(p_m^t)^T\mathbb{E}_{\psi_m}(\psi_m)\mathbb{E}_{\mathbf{H}_m}(\mathbf{H}_m^T)\mathbb{E}_{\mathbf{z}_t}(\mathbf{z}_t)\}\\
$

$
L_2=\sum_{m=1}^{Ml}\{\mathbb{E}_{\mathcal{Q}(\alpha_m)}(ln|\alpha_m|)-\mathbb{E}_{\mathcal{Q}(\mathbf{W_m})\mathcal{Q}(\mathbf{\alpha_m})}(\mathbf{W}_m^T\alpha_m\mathbf{W}_m)\}\\
$

$
L_3=\sum_{m=1}^{M(s+1)}\{\mathbb{E}_{\mathcal{Q}(\beta_m)}(ln|\beta_m|)-\mathbb{E}_{\mathcal{Q}(\mathbf{H_m})\mathcal{Q}(\mathbf{\beta_m})}(\mathbf{H}_m^T\beta_m\mathbf{H}_m)\}\\
$

$
L_4=\sum_{m=1}^{Ml}\{\mathbb{E}_{\mathcal{Q}(\alpha_m)}(ln|\alpha_m|)-\mathbb{E}_{\mathcal{Q}(\mathbf{W_m})\mathcal{Q}(\mathbf{\alpha_m})}(\mathbf{W}_m^T\alpha_m\mathbf{W}_m)\}\\
$

$
L_5=\sum_{m=1}^{M(s+1)}\{\mathbb{E}_{\mathcal{Q}(\beta_m)}(ln|\beta_m|)-\mathbb{E}_{\mathcal{Q}(\mathbf{H_m})\mathcal{Q}(\mathbf{\beta_m})}(\mathbf{H}_m^T\beta_m\mathbf{H}_m)\}\\
$
不好意思~ELBO实在不知道咋算
 -->



### Update $\mathcal{Q}(\mathbf{z}_t)$

$\Lambda_t^z =\sum_{m=1}^{Ml}\mathbb{E}_{Q(\tau_m)}(\tau_m)\mathbb{E}_{Q(\mathbf{W}_m)}(\mathbf{W}_m\mathbf{W}_m^T) + \sum_{m=1}^{M(s+1)}\mathbb{E}_{Q(\psi_m)}(\psi_m)\mathbb{E}_{Q(\mathbf{H}_m)}(\mathbf{H}_m\mathbf{H}_m^T)+\mathbf{I}$

$\mu_t^z=(\Lambda_t^z )^{-1}\{\sum_{m=1}^{Ml}\mathbb{E}_{Q(\tau_m)}(\tau_m)\mathbb{E}_{Q(\mathbf{W}_m)}(\mathbf{W}_m)p_m^t + \sum_{m=1}^{M(s+1)}\mathbb{E}_{Q(\psi_m)}(\psi_m)\mathbb{E}_{Q(\mathbf{H}_m)}(\mathbf{H}_m)f_m^t\}$

$\mathbb{E}_{Q(\tau_m)}(\tau_m) = \frac{\lambda_m^\tau}{\nu_m^\tau}$

$\mathbb{E}_{Q(\mathbf{W}_m)}(\mathbf{W}_m\mathbf{W}_m^T) =  (\Lambda_m^W)^{-1}+\mu_m\mu_m^T$






### update $\mathcal{Q}(\mathbf{H})$, $\mathcal{Q}(\mathbf{W})$

$\Lambda_m^W =\sum_{t=1}^{N}\mathbb{E}_{Q(\tau_m)}(\tau_m)\mathbb{E}_{Q(\mathbf{z}_t)}(\mathbf{z}_t\mathbf{z}_t^T) + \mathbb{E}_{Q(\alpha_m)}(\alpha_m)$

$\mu_m^\mathbf{W}=(\Lambda_m^W)^{-1} \mathbb{E}_{Q(\tau_m)}(\tau_m) \sum_{t=1}^{N}\mathbb{E}_{Q(\mathbf{z}_t)}(\mathbf{z}_t)p_m^t$

$\Lambda_m^H =\sum_{t=1}^{N}\mathbb{E}_{Q(\psi_m)}(\psi_m)\mathbb{E}_{Q(\mathbf{z}_t)}(\mathbf{z}_t\mathbf{z}_t^T) + \mathbb{E}_{Q(\beta_m)}(\beta_m)$

$\mu_m^\mathbf{H}=(\Lambda_m^H)^{-1} \mathbb{E}_{Q(\psi_m)}(\psi_m) \sum_{t=1}^{N}\mathbb{E}_{Q(\mathbf{z}_t)}(\mathbf{z}_t)f_m^t$



$\mathbb{E}_{Q(\mathbf{\alpha_m})}(\alpha_m) = \nu_m^{\alpha}\mathbf{V}_m^{\alpha}$

$\mathbb{E}_{Q(\mathbf{z}_t)}(\mathbf{z}_t\mathbf{z}_t^T) =  (\Lambda_t^z)^{-1}+\mu_t^z(\mu_t^z)^T$


### update $\mathcal{Q}(\tau)$, $\mathcal{Q}(\psi)$

$\lambda_m^\tau=j_m^\tau+\frac{1}{2}N$
$\nu_m^\tau = k_m^\tau+\frac{1}{2}\sum\limits_{t=1}^N\mathbb{E}_{Q(\mathbf{z}_t)Q(\mathbf{W}_m)}\{(p_m^t-\mathbf{W}_m^T\mathbf{z}_t)^T(p_m^t-\mathbf{W}_m^T\mathbf{z}_t)\}$

$\lambda_m^\psi=j_m^\psi+\frac{1}{2}N$
$\nu_m^\psi = k_m^\psi+\frac{1}{2}\sum\limits_{t=1}^N\mathbb{E}_{Q(\mathbf{z}_t)Q(\mathbf{H}_m)}\{(f_m^t-\mathbf{H}_m^T\mathbf{z}_t)^T(f_m^t-\mathbf{H}_m^T\mathbf{z}_t)\}$

$\mathbb{E}_{\mathcal{Q}(\mathbf{z})}(\mathbf{z}^T\mathbf{z})=\text{Tr}(\Lambda_{z}^{-1})+\mu_z^T\mu_z$


### update $\mathcal{Q}(\alpha)$, $\mathcal{Q}(\beta)$

$\nu_m^\alpha=a_m^\alpha+1$
$(\mathbf{V}_m^\alpha )^{-1} = (\mathbf{B}_m^\alpha )^{-1}+\mathbb{E}_{Q(\mathbf{W}_m)}\{\mathbf{W}_m\mathbf{W}_m^T\}$

$\nu_m^\beta=a_m^\beta+1$
$(\mathbf{V}_m^\beta )^{-1}= (\mathbf{B}_m^\beta )^{-1}+\mathbb{E}_{Q(\mathbf{H}_m)}\{\mathbf{H}_m\mathbf{H}_m^T\}$


---

## Monitoring Strategy

Monitoring Strategy is completely established according to the aforementioned reference, where SPE and $T^2$ are constructed. The specific calculations are given as follows:

### index $T^2$  
$
T^2=\mathbf{z}_{new}^T\mathbf{z}_{new}
$
where 
$
\mathbf{z}_{new} = \Lambda^\mathbf{z}_t(\mathbf{z}_{new}^p+\mathbf{z}_{new}^f)\\
\mathbf{z}_{new}^p = (\mu^W)^T\mathbf{\tau}\mathbf{p}_{new}\\
\mathbf{z}_{new}^f = (\mu^H)^T\mathbf{\psi}\mathbf{f}_{new}
$
$
\mathbf{\tau} = diag\{[\mathbb{E}_{Q(\tau_m)}(\tau_m)]_{m=1}^{Ml}\}\\
\mathbf{\psi} = diag\{[\mathbb{E}_{Q(\psi_m)}(\psi_m)]_{m=1}^{M(s+1)}\}
$

### index $SPE$  

$
SPE=\delta_{new}^T\mathbf{\psi}\delta_{new}\\
\delta_{new} = \mathbf{f}_{new}-\mu^H\mathbf{z}_{new}
$












