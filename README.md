# ModalCens: Parametric Modal Regression with Right Censoring

<!-- badges: start -->
[![R-CMD-check](https://img.shields.io/badge/R--CMD--check-passing-brightgreen)](https://github.com/chedgala/ModalCens)
[![License: GPL-3](https://img.shields.io/badge/License-GPL3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
<!-- badges: end -->

**ModalCens** is an R package for fitting parametric modal regression models to continuous positive random variables within the exponential family. While traditional Generalized Linear Models (GLMs) target the conditional mean, **ModalCens** directly models the conditional mode $M_i$, providing a more robust and meaningful measure of central tendency for asymmetric and heavy-tailed distributions — particularly under right censoring.

The methodology is grounded in the framework of modal linear regression introduced by [Yao & Li (2014)](#references), with connections to recent advances surveyed in [Xiang, Yao & Cui (2026)](#references), extended here to parametric exponential family distributions under right censoring.

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
  | `"invgauss"` | Log | $(0, \infty)$ | Highly skewed data ($\lambda > 3M_i$) |

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

## Quick Start

This example uses the built-in `trees` dataset with simulated censoring **for illustration purposes only**. In practice, supply your own dataset with a real censoring indicator.

```r
library(ModalCens)

# 1. Prepare data: scale Volume to (0, 1) for multi-family compatibility
data(trees)
y_scaled <- (trees$Volume - min(trees$Volume) + 1) /
            (max(trees$Volume) - min(trees$Volume) + 2)

# 2. Simulate a censoring indicator (for illustration only)
#    cens = 1: right-censored  |  cens = 0: fully observed
set.seed(2026)
cut_point <- quantile(y_scaled, 0.85)
cens_ind  <- ifelse(y_scaled > cut_point, 1, 0)
y_obs     <- pmin(y_scaled, cut_point)

df <- data.frame(
  y      = y_obs,
  cens   = cens_ind,
  Girth  = trees$Girth,
  Height = trees$Height
)

# 3. Fit a Gamma modal regression
fit_gamma <- modal_cens(y ~ Girth + Height, data = df,
                        cens = df$cens, family = "gamma")

# 4. Summary and diagnostics
summary(fit_gamma)
plot(fit_gamma)
```

---

## Multi-Family Comparison

```r
families   <- c("gamma", "beta", "weibull", "invgauss")
models     <- list()
aic_values <- numeric()

for (f in families) {
  mod <- try(
    modal_cens(y ~ Girth + Height, data = df, cens = df$cens, family = f),
    silent = TRUE
  )
  if (!inherits(mod, "try-error")) {
    models[[f]]     <- mod
    aic_values[[f]] <- AIC(mod)
  }
}

best_family <- names(which.min(aic_values))
cat("Best model:", best_family, "| AIC =", min(aic_values), "\n")

summary(models[[best_family]])
plot(models[[best_family]])
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
| `family` | One of `"gamma"`, `"beta"`, `"weibull"`, `"invgauss"` |

The function stops with an error if any `NA` is detected in `cens` or in the variables referenced by `formula`. No imputation is performed.

The function returns an object of class `"ModalCens"` with methods for `summary()`, `plot()`, `AIC()`, `BIC()`, `logLik()`, `coef()`, `vcov()`, and `residuals()`.

---

## Authors

- **Christian E. Galarza** — Escuela Superior Politécnica del Litoral, Ecuador · [chedgala@espol.edu.ec](mailto:chedgala@espol.edu.ec)
- **Víctor H. Lachos** — University of Connecticut, USA · [hlachos@uconn.edu](mailto:hlachos@uconn.edu)

---

## References

- **Yao, W. & Li, L. (2014).** A new regression model: modal linear regression. *Scandinavian Journal of Statistics*, 41(3), 656–671. <https://doi.org/10.1111/sjos.12054>

- **Xiang, S., Yao, W. & Cui, X. (2026).** Advances in modal regression: from theoretical foundations to practical implementations. *Wiley Interdisciplinary Reviews: Computational Statistics*, e70014. <https://doi.org/10.1002/wics.70014>

- **Dunn, P. K. & Smyth, G. K. (1996).** Randomized quantile residuals. *Journal of Computational and Graphical Statistics*, 5(3), 236–244.
