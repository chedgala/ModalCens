# ModalCens: Parametric Modal Regression with Right Censoring

<!-- badges: start -->
[![CRAN status](https://www.r-pkg.org/badges/version/ModalCens)](https://CRAN.R-project.org/package=ModalCens)
[![R-CMD-check](https://img.shields.io/badge/R--CMD--check-passing-brightgreen)](https://github.com/chedgala/ModalCens)
[![License: GPL-3](https://img.shields.io/badge/License-GPL3-blue.svg)](https://www.r-project.org/Licenses/GPL-3)
<!-- badges: end -->

**ModalCens** is an R package for fitting parametric modal regression models to continuous positive random variables. While traditional Generalized Linear Models (GLMs) target the conditional mean, **ModalCens** directly models the conditional mode $M_i$, providing a more robust and meaningful measure of central tendency for asymmetric and heavy-tailed distributions — particularly under right censoring.

The methodology is grounded in the framework of modal linear regression introduced by [Yao & Li (2014)](#references). It extends beyond traditional exponential family distributions to include highly flexible models such as the Log-Logistic and Birnbaum-Saunders distributions, as detailed in our preprint [Galarza & Lachos (2026)](#references).

---

## Key Features

- **Mode-Link Framework:** Analytical reparameterization of exponential family densities to link the conditional mode $M_i$ directly to a linear predictor via $g(M_i) = \mathbf{X}_i^\top \boldsymbol{\gamma}$.

- **Right-Censoring Support:** Right-censored observations (`cens = 1`) contribute their survival probability $S(y_i)$ to the likelihood, while fully observed values (`cens = 0`) contribute their density $f(y_i)$.

- **Strict Data Validation:** The function stops with an informative error if `cens` or any variable in `data` contains missing values (`NA`). No imputation is performed — the user is responsible for supplying a complete dataset.

- **Supported Distributions:**

  | Family | Link | Domain | Typical Use |
  |---|---|---|---|
  | `"gamma"` | Log | $(0, \infty)$ | Skewed positive data |
  | `"beta"` | Logit | $(0, 1)$ | Proportions and rates |
  | `"weibull"` | Log | $(0, \infty)$ | Survival and reliability |
  | `"invgauss"` | Log | $(0, \infty)$ | Highly skewed data ($\lambda > 3M_i$) * |
  | `"lognormal"` | Log | $(0, \infty)$ | Multiplicative / log-symmetric data |
  | `"loglogistic"` | Log | $(0, \infty)$ | Non-monotonic hazard rates, heavy tails |
  | `"bisa"` | Log | $(0, \infty)$ | Fatigue life, wear-out processes |

  *\* **Note on Inverse Gaussian:** The condition $\lambda > 3M_i$ required to ensure a valid mode parameterization is a strong parametric restriction. This can make the Inverse Gaussian distribution less plausible in many routine modeling contexts compared to the other families available.*

- **Asymptotic Inference:** Standard errors derived from the observed Fisher information matrix (inverse Hessian at the MLE).

- **Randomized Quantile Residuals:** Implements the Dunn–Smyth method for model diagnostics, valid under any continuous distribution and under censoring.

---

## Installation

```r
# Development version from GitHub
# install.packages("devtools")
devtools::install_github("chedgala/ModalCens")
```

---

## Quick Start: Multi-Family Comparison

Censoring is simulated for didactic purposes. Each family is fitted on its natural domain:
long-tail families (Gamma, Weibull, InvGauss) on the original scale, and Beta on the $(0,1)$ scale.

```r
library(ModalCens)
data(trees)
set.seed(007)

# Simulate 15% right-censoring (cens = 1: censored | cens = 0: observed)
q85      <- quantile(trees$Volume, 0.85)
cens_ind <- as.integer(trees$Volume > q85)

# (0,1) rescaling for Beta only
ys     <- (trees$Volume - min(trees$Volume) + 1) / (max(trees$Volume) - min(trees$Volume) + 2)
q85b   <- quantile(ys, 0.85)
cens_b <- as.integer(ys > q85b)

# Original scale for long-tail families; (0,1) scale for Beta
df_orig <- data.frame(y = pmin(trees$Volume, q85), cens = cens_ind,
                      Girth = trees$Girth, Height = trees$Height)
df_beta <- data.frame(y = pmin(ys, q85b), cens = cens_b,
                      Girth = trees$Girth, Height = trees$Height)

# Fit all families
datasets <- list(gamma = df_orig, weibull = df_orig,
                 invgauss = df_orig, lognormal = df_orig, beta = df_beta,
                 loglogistic = df_orig, bisa = df_orig)
models <- list(); aic_values <- list()

for (f in names(datasets)) {
  mod <- try(modal_cens(y ~ Girth + Height, data = datasets[[f]],
                        cens = datasets[[f]]$cens, family = f), silent = TRUE)
  if (!inherits(mod, "try-error")) { models[[f]] <- mod; aic_values[[f]] <- AIC(mod) }
}

# Select best model by AIC
best <- names(which.min(aic_values))
cat("Best family:", best, "| AIC =", round(aic_values[[best]], 3), "\n")

# Full analysis of the winner
fit <- models[[best]]
summary(fit)
plot(fit)
confint(fit)
```

---

## Methodology

### Mode Reparameterization

The package reparameterizes exponential family densities of the form

$$
f(y_i \mid \theta_i, \phi) = \exp\left\lbrace\frac{y_i \theta_i - b(\theta_i)}{a(\phi)} + c(y_i, \phi)\right\rbrace
$$

by solving the first-order condition $\left.\dfrac{\partial \log f(y_i)}{\partial y_i}\right|_{y_i = M_i} = 0$, which expresses the canonical parameter $\theta_i$ as an explicit function of the conditional mode $M_i$ and the dispersion $\phi$. This yields the mode-link models:

| Family | Mode–parameter relation |
|---|---|
| Gamma | $\alpha = \phi^{-1} + 1$, $\;\text{scale} = M_i \cdot \phi$ |
| Beta | $\alpha = \phi + 1.01$, $\;\beta_p = (\alpha - 1)/M_i - \alpha + 2$ |
| Weibull | $k = \phi + 1.01$, $\;\lambda = M_i \cdot (k/(k-1))^{1/k}$ |
| Inv. Gaussian | $\mu = \bigl[M_i^{-2} - 3/(\lambda M_i)\bigr]^{-1/2}$, $\;\lambda > 3M_i$ |
| Log-Normal | $\mu_{LN} = \log(M_i) + \sigma^2$, $\;\sigma = \sqrt{\phi}$ |
| Log-Logistic | $k = \phi + 1.01$, $\;\alpha = M_i \cdot \bigl((k+1)/(k-1)\bigr)^{1/k}$ |
| Birnbaum-Saunders | $\alpha_{bs} = \sqrt{\phi}$, $\;\beta_{bs} = M_i / \bigl(\sqrt{1+\alpha_{bs}^2/4} - \alpha_{bs}/2\bigr)^2$ |

### Censored Log-Likelihood

For a sample of $n$ observations with censoring indicator $\delta_i \in \{0, 1\}$ (where $\delta_i = 1$ denotes right-censored):

$$\ell(\boldsymbol{\gamma}, \phi) = \sum_{i=1}^{n} \left[(1 - \delta_i) \log f(y_i \mid M_i, \phi) + \delta_i \log S(y_i \mid M_i, \phi)\right]$$

where $S(\cdot) = 1 - F(\cdot)$ is the survival function. Maximization is performed via the BFGS algorithm.

### Randomized Quantile Residuals

For each observation, define $u_i = F(y_i \mid \hat{M}_i, \hat{\phi})$. For right-censored observations ($\delta_i = 1$), $u_i$ is drawn uniformly on $[F(y_i), 1]$. The residual is $r_i = \Phi^{-1}(u_i)$, which follows $\mathcal{N}(0,1)$ under the correct model (Dunn & Smyth, 1996).

---

## Main Function

```r
modal_cens(formula, data, cens, family = "gamma")
```

| Argument | Description |
|---|---|
| `formula` | A formula object, e.g., `y ~ x1 + x2` |
| `data` | A data frame with no missing values |
| `cens` | Binary vector: `1` = right-censored, `0` = fully observed. Must have no `NA`s. |
| `family` | One of `"gamma"`, `"beta"`, `"weibull"`, `"invgauss"`, `"lognormal"`, `"loglogistic"`, `"bisa"` |

The function stops with an error if any `NA` is detected in `cens` or in the variables referenced by `formula`. No imputation is performed.

The function returns an object of class `"ModalCens"` with methods for `summary()`, `plot()`, `AIC()`, `BIC()`, `logLik()`, `coef()`, `vcov()`, and `residuals()`.

---

## Authors

- **Christian E. Galarza** — Escuela Superior Politécnica del Litoral, Ecuador · [chedgala@espol.edu.ec](mailto:chedgala@espol.edu.ec)
- **Víctor H. Lachos** — University of Connecticut, USA · [hlachos@uconn.edu](mailto:hlachos@uconn.edu)

---

## References

- **Yao, W. & Li, L. (2014).** A new regression model: modal linear regression. *Scandinavian Journal of Statistics*, 41(3), 656–671. <https://doi.org/10.1111/sjos.12054>

- **Galarza, C. E., & Lachos, V. H. (2026).** Parametric Modal Regression for Positive Distributions. *arXiv preprint*, arXiv:2603.07099. <doi:10.48550/arXiv.2603.07099>

- **Dunn, P. K. & Smyth, G. K. (1996).** Randomized quantile residuals. *Journal of Computational and Graphical Statistics*, 5(3), 236–244.
