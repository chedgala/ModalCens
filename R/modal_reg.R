#' Modal Regression Fit with Right Censoring
#'
#' @description
#' Fits a parametric modal regression model under right censoring by linking
#' the conditional mode \eqn{M_i} to a linear predictor via a family-specific
#' link function \eqn{g(M_i) = \mathbf{x}_i^\top \boldsymbol{\gamma}}.
#'
#' The density of each family is reparameterized so that \eqn{M_i} appears
#' explicitly, obtained by solving
#' \eqn{\partial \log f(y_i) / \partial y_i |_{y_i = M_i} = 0}.
#' The resulting mode-parameter mappings are:
#'
#' \tabular{lll}{
#'   \strong{Family}  \tab \strong{Link} \tab \strong{Mode mapping} \cr
#'   Gamma      \tab log   \tab \eqn{\alpha = \phi^{-1}+1},\ \eqn{\text{scale} = M_i \cdot \phi} \cr
#'   Beta       \tab logit \tab \eqn{\alpha = \phi+1.01},\ \eqn{\beta_p = (\alpha-1)/M_i - \alpha + 2} \cr
#'   Weibull    \tab log   \tab \eqn{k = \phi+1.01},\ \eqn{\lambda = M_i\,(k/(k-1))^{1/k}} \cr
#'   Inv. Gauss \tab log   \tab \eqn{\mu = [(M_i^{-2}) - 3/(\lambda M_i)]^{-1/2}},\ \eqn{\lambda > 3M_i} \cr
#'   Log-Normal \tab log   \tab \eqn{\mu_{LN} = \log(M_i) + \sigma^2},\ \eqn{\sigma = \sqrt{\phi}} \cr
#' }
#'
#' The censored log-likelihood is
#' \deqn{\ell(\boldsymbol{\gamma}, \phi) =
#'   \sum_{i=1}^{n} \bigl[(1-\delta_i)\log f(y_i) + \delta_i \log S(y_i)\bigr],}
#' where \eqn{\delta_i = 1} for right-censored observations and
#' \eqn{S(\cdot) = 1 - F(\cdot)} is the survival function.
#' Maximization is performed via the BFGS algorithm with analytical Hessian
#' for asymptotic standard errors.
#'
#' @param formula An object of class "formula" (e.g., y ~ x1 + x2).
#' @param data A data frame containing the variables in the model. Must not
#'   contain missing values (\code{NA}) in any variable used by \code{formula}.
#'   No imputation is performed; the function stops with an error if \code{NA}s
#'   are detected.
#' @param cens A binary numeric vector indicating censoring status:
#'   \code{1} = right-censored (only a lower bound \eqn{y_i} is observed),
#'   \code{0} = fully observed. Must not contain \code{NA}s and must have the
#'   same length as \code{nrow(data)}.
#' @param family A character string naming the distribution family: \code{"gamma"},
#'   \code{"beta"}, \code{"weibull"}, \code{"invgauss"}, or \code{"lognormal"}.
#'   Default is \code{"gamma"}.
#'
#' @return An object of class \code{"ModalCens"} containing:
#' \describe{
#'   \item{coefficients}{Estimated regression coefficients (beta vector).}
#'   \item{phi}{Estimated dispersion/shape parameter (on original scale).}
#'   \item{vcov}{Full covariance matrix (p+1 x p+1), including log_phi.}
#'   \item{vcov_beta}{Covariance matrix for regression coefficients only (p x p).}
#'   \item{fitted.values}{Estimated conditional modes.}
#'   \item{residuals}{Randomized quantile residuals (Dunn-Smyth).}
#'   \item{loglik}{Maximized log-likelihood value.}
#'   \item{n}{Number of observations used in fitting.}
#'   \item{n_par}{Total number of estimated parameters.}
#'   \item{family}{Distribution family used.}
#'   \item{cens}{Censoring indicator vector as supplied.}
#'   \item{call}{The matched call.}
#'   \item{terms}{The model terms object.}
#'   \item{y}{Response variable.}
#' }
#'
#' @author Christian Galarza and Victor Lachos
#'
#' @importFrom stats dgamma pgamma dbeta pbeta dweibull pweibull dlnorm plnorm pnorm qnorm
#' @importFrom stats optim model.frame model.matrix model.response coef vcov
#' @importFrom stats residuals logLik AIC BIC runif qlogis lm.fit na.fail
#' @export
#'
#' @section Available methods:
#' Objects of class \code{"ModalCens"} support the following S3 methods:
#' \describe{
#'   \item{\code{summary()}}{Coefficient table with standard errors, z-values,
#'     and p-values; dispersion parameter; log-likelihood; AIC; BIC; and
#'     pseudo R-squared.}
#'   \item{\code{plot()}}{Two-panel diagnostic plot: (1) residuals vs.\ fitted
#'     modes, distinguishing observed and censored points; (2) normal Q-Q plot
#'     of randomized quantile residuals.}
#'   \item{\code{coef()}}{Estimated regression coefficients \eqn{\hat{\boldsymbol{\gamma}}}.}
#'   \item{\code{vcov()}}{Full \eqn{(p+1) \times (p+1)} covariance matrix
#'     including \code{log_phi}. Use \code{object$vcov_beta} for the
#'     \eqn{p \times p} submatrix of regression coefficients only.}
#'   \item{\code{residuals()}}{Randomized quantile residuals (Dunn & Smyth,
#'     1996), which follow \eqn{\mathcal{N}(0,1)} under the correct model,
#'     even under censoring.}
#'   \item{\code{logLik()}}{Maximized log-likelihood value.}
#'   \item{\code{AIC()} / \code{BIC()}}{Information criteria computed as
#'     \eqn{-2\ell + k \cdot p} with \eqn{k = 2} or \eqn{k = \log n}.}
#' }
#'
#' @examples
#' data(trees, package = "datasets")
#' set.seed(007)
#'
#' # Simulate 15% right-censoring (for didactic purposes)
#' q85      <- quantile(trees$Volume, 0.85)
#' cens_ind <- as.integer(trees$Volume > q85)  # 1 = censored, 0 = observed
#'
#' # (0,1) rescaling for Beta
#' ys       <- (trees$Volume - min(trees$Volume) + 1) / (max(trees$Volume) - min(trees$Volume) + 2)
#' q85b     <- quantile(ys, 0.85)
#' cens_b   <- as.integer(ys > q85b)
#'
#' # Build datasets
#' df_orig <- data.frame(y = pmin(trees$Volume, q85), cens = cens_ind,
#'                       Girth = trees$Girth, Height = trees$Height)
#' df_beta <- data.frame(y = pmin(ys, q85b), cens = cens_b,
#'                       Girth = trees$Girth, Height = trees$Height)
#'
#' # Fit all families
#' datasets <- list(gamma = df_orig, weibull = df_orig,
#'                  invgauss = df_orig, lognormal = df_orig, beta = df_beta)
#' models <- list(); aic_values <- list()
#'
#' for (f in names(datasets)) {
#'   mod <- try(modal_cens(y ~ Girth + Height, data = datasets[[f]],
#'                         cens = datasets[[f]]$cens, family = f), silent = TRUE)
#'   if (!inherits(mod, "try-error")) { models[[f]] <- mod; aic_values[[f]] <- AIC(mod) }
#' }
#'
#' # Select best model by AIC and analyse
#' best <- names(which.min(aic_values))
#' cat("Best family:", best, "| AIC =", round(aic_values[[best]], 3), "\n")
#'
#' fit <- models[[best]]
#' summary(fit)
#' plot(fit)
#' confint(fit)
#'
modal_cens <- function(formula, data, cens, family = "gamma") {

  # ------------------------------------------------------------------
  # 0. Input validation
  # ------------------------------------------------------------------
  family <- match.arg(family, choices = c("gamma", "beta", "weibull", "invgauss", "lognormal"))

  if (!is.data.frame(data)) {
    stop("'data' must be a data frame.")
  }

  # --- cens validation ---
  if (!is.numeric(cens) || !all(cens %in% c(0, 1))) {
    stop("'cens' must be a binary numeric vector: 1 = right-censored, 0 = fully observed.")
  }

  if (anyNA(cens)) {
    stop("'cens' must not contain missing values (NA).")
  }

  if (length(cens) != nrow(data)) {
    stop(sprintf(
      "'cens' has length %d but 'data' has %d rows. They must match.",
      length(cens), nrow(data)
    ))
  }

  # ------------------------------------------------------------------
  # 1. Build model frame — stop immediately if any NA is found.
  #    na.fail raises an error rather than silently dropping rows.
  # ------------------------------------------------------------------
  mf <- tryCatch(
    model.frame(formula, data = data, na.action = na.fail),
    error = function(e) {
      stop(
        "Missing values (NA) detected in the model variables. ",
        "Please remove or correct them before calling modal_cens(). ",
        "No imputation is performed.\n",
        "Original error: ", conditionMessage(e),
        call. = FALSE
      )
    }
  )

  mt <- attr(mf, "terms")
  y  <- model.response(mf)
  X  <- model.matrix(formula, mf)
  n  <- nrow(mf)
  p  <- ncol(X)

  # ------------------------------------------------------------------
  # 2. Family-specific response validation
  # ------------------------------------------------------------------
  if (family %in% c("gamma", "weibull", "invgauss", "lognormal") && any(y <= 0)) {
    stop(sprintf("All response values must be strictly positive for family = '%s'.", family))
  }
  if (family == "beta" && (any(y <= 0) || any(y >= 1))) {
    stop("All response values must lie in the open interval (0, 1) for family = 'beta'.")
  }

  # ------------------------------------------------------------------
  # 3. Negative log-likelihood
  #
  #    Censoring convention:
  #      cens_i = 1  ->  right-censored  ->  contributes log S(y_i)
  #      cens_i = 0  ->  fully observed  ->  contributes log f(y_i)
  # ------------------------------------------------------------------
  neg_loglik <- function(par) {

    coef_reg <- par[1:p]         # regression coefficients
    phi      <- exp(par[p + 1])  # dispersion/shape (always positive via exp)

    # Conditional mode via link function
    if (family == "beta") {
      eta  <- X %*% coef_reg
      moda <- exp(eta) / (1 + exp(eta))  # logit link -> (0, 1)
    } else {
      moda <- exp(X %*% coef_reg)        # log link   -> (0, Inf)
    }

    suppressWarnings({

      if (family == "gamma") {
        # alpha = 1/phi + 1,  scale = mode * phi  (= mode / (alpha - 1))
        alpha <- (1 / phi) + 1
        scale <- moda * phi
        dens  <- dgamma(y, shape = alpha, scale = scale, log = TRUE)
        prob  <- pgamma(y, shape = alpha, scale = scale,
                        lower.tail = FALSE, log.p = TRUE)

      } else if (family == "beta") {
        # alpha = phi + 1.01 (ensures alpha > 1 so mode is interior)
        alpha  <- phi + 1.01
        beta_p <- ((alpha - 1) / moda) - alpha + 2
        if (any(beta_p <= 0)) return(1e10)  # guard: beta_p must be positive
        dens <- dbeta(y, shape1 = alpha, shape2 = beta_p, log = TRUE)
        prob <- pbeta(y, shape1 = alpha, shape2 = beta_p,
                      lower.tail = FALSE, log.p = TRUE)

      } else if (family == "weibull") {
        # k = phi + 1.01 (ensures k > 1 so mode is interior)
        k      <- phi + 1.01
        lambda <- moda * (k / (k - 1))^(1 / k)
        dens   <- dweibull(y, shape = k, scale = lambda, log = TRUE)
        prob   <- pweibull(y, shape = k, scale = lambda,
                           lower.tail = FALSE, log.p = TRUE)

      } else if (family == "invgauss") {
        # Parametric restriction: lambda > 3 * mode (ensures valid mu)
        lambda_ig  <- phi
        violations <- lambda_ig <= 3 * moda
        if (any(violations)) {
          # Smooth penalty to preserve BFGS gradient continuity
          penalty <- sum(3 * moda[violations] - lambda_ig)
          return(1e10 + penalty)
        }
        mu   <- ((1 / moda^2) - (3 / (lambda_ig * moda)))^(-1 / 2)
        dens <- 0.5 * log(lambda_ig) -
          0.5 * log(2 * pi * y^3) -
          (lambda_ig * (y - mu)^2) / (2 * mu^2 * y)
        z1   <-  sqrt(lambda_ig / y) * (y / mu - 1)
        z2   <- -sqrt(lambda_ig / y) * (y / mu + 1)
        prob <- log(pmax(1e-10,
                         1 - (pnorm(z1) + exp(2 * lambda_ig / mu) * pnorm(z2))))

      } else if (family == "lognormal") {
        # Mode of LogNormal: M = exp(mu_ln - sigma^2)
        # => mu_ln = log(M) + sigma^2, sigma = sqrt(phi)
        sigma_ln <- sqrt(phi)
        mu_ln    <- log(moda) + phi
        dens     <- dlnorm(y, meanlog = mu_ln, sdlog = sigma_ln, log = TRUE)
        prob     <- plnorm(y, meanlog = mu_ln, sdlog = sigma_ln,
                           lower.tail = FALSE, log.p = TRUE)
      }

    }) # end suppressWarnings

    # cens = 1 -> right-censored -> log S(y_i)
    # cens = 0 -> fully observed -> log f(y_i)
    ll <- sum((1 - cens) * dens + cens * prob)
    if (!is.finite(ll)) return(1e10)
    return(-ll)
  }

  # ------------------------------------------------------------------
  # 4. Initial values
  # ------------------------------------------------------------------
  y_start <- switch(family,
                    "beta" = (y * (n - 1) + 0.5) / n,
                    y
  )

  lm_resp <- switch(family,
                    "beta" = qlogis(y_start),
                    log(y_start)
  )

  beta_init <- tryCatch(
    lm.fit(X, lm_resp)$coefficients,
    error = function(e) rep(0, p)
  )

  init_par <- c(beta_init, log(1))  # log(phi) = 0 => phi = 1

  # ------------------------------------------------------------------
  # 5. Optimization (BFGS)
  # ------------------------------------------------------------------

  # 1. Definir init_par estándar (ceros por defecto)
  init_par <- rep(0, ncol(X) + 1)

  # 2. Ajuste condicional para la Gaussiana Inversa
  if (family == "invgauss") {
    # Forzar un intercepto negativo para reducir la Moda inicial (opcional pero recomendado)
    init_par[1] <- -1.0

    # Forzar un phi (lambda) inicial alto para cumplir lambda > 3 * max(Mode)
    # Nota: Si tu paquete parametriza el último valor como log(phi), usa log(5).
    # Si es phi directo, usa 5. Asumo log(phi) basado en tu código Gamma.
    init_par[length(init_par)] <- log(5.0)
  }

  # 3. Optimización
  opt <- optim(
    par     = init_par,
    fn      = neg_loglik,
    hessian = TRUE,
    method  = "BFGS",
    control = list(maxit = 2000, reltol = 1e-10)
  )

  if (opt$convergence != 0) {
    warning(sprintf(
      "Optimization did not converge (code %d). Results may be unreliable.",
      opt$convergence
    ))
  }

  # ------------------------------------------------------------------
  # 6. Asymptotic inference
  # ------------------------------------------------------------------
  vcov_mat <- tryCatch(
    solve(opt$hessian),
    error = function(e) {
      warning("Hessian is singular; standard errors are not available.")
      matrix(NA_real_, nrow = p + 1, ncol = p + 1)
    }
  )

  coef_names         <- colnames(X)
  rownames(vcov_mat) <- colnames(vcov_mat) <- c(coef_names, "log_phi")

  coef_est <- opt$par[1:p]
  names(coef_est) <- coef_names

  phi_est  <- exp(opt$par[p + 1])

  moda_est <- if (family == "beta") {
    exp(X %*% coef_est) / (1 + exp(X %*% coef_est))
  } else {
    exp(X %*% coef_est)
  }

  # ------------------------------------------------------------------
  # 7. Randomized Quantile Residuals (Dunn-Smyth)
  #
  #    For right-censored observations (cens = 1), u_i is drawn
  #    uniformly on [F(y_i), 1] to account for the unknown true value.
  # ------------------------------------------------------------------
  k_est     <- phi_est + 1.01      # shared shape offset for Weibull / Beta
  scale_est <- moda_est * phi_est  # Gamma scale

  u <- suppressWarnings(switch(family,

                               "gamma" = pgamma(
                                 y,
                                 shape = (1 / phi_est) + 1,
                                 scale = scale_est
                               ),

                               "beta" = {
                                 alpha_est  <- k_est
                                 beta_p_est <- ((alpha_est - 1) / moda_est) - alpha_est + 2
                                 pbeta(y, shape1 = alpha_est, shape2 = beta_p_est)
                               },

                               "weibull" = {
                                 lambda_est <- moda_est * (k_est / (k_est - 1))^(1 / k_est)
                                 pweibull(y, shape = k_est, scale = lambda_est)
                               },

                               "invgauss" = {
                                 lambda_ig <- phi_est
                                 mu_ig     <- ((1 / moda_est^2) - (3 / (lambda_ig * moda_est)))^(-1 / 2)
                                 z1 <-  sqrt(lambda_ig / y) * (y / mu_ig - 1)
                                 z2 <- -sqrt(lambda_ig / y) * (y / mu_ig + 1)
                                 pnorm(z1) + exp(2 * lambda_ig / mu_ig) * pnorm(z2)
                               },

                               "lognormal" = {
                                 sigma_ln <- sqrt(phi_est)
                                 mu_ln    <- log(moda_est) + phi_est
                                 plnorm(y, meanlog = mu_ln, sdlog = sigma_ln)
                               }
  ))

  # Randomize right-censored observations (cens = 1) over [F(y_i), 1]
  u <- pmin(pmax(u, 1e-7), 1 - 1e-7)          # clamp ANTES de randomizar
  if (any(cens == 1)) {
    idx   <- cens == 1
    u_low <- pmin(pmax(u[idx], 1e-7), 1 - 1e-7)  # clamp seguro, elimina NAs implícitamente
    u_low[is.na(u_low)] <- 1e-7                   # fallback explícito para NAs residuales
    u[idx] <- runif(sum(idx), min = u_low, max = 1)
  }
  u <- pmin(pmax(u, 1e-7), 1 - 1e-7)          # clamp de seguridad post-runif
  rq_res <- qnorm(u)

  # ------------------------------------------------------------------
  # 8. Return object
  # ------------------------------------------------------------------
  res <- list(
    coefficients  = coef_est,
    phi           = phi_est,
    vcov          = vcov_mat,
    vcov_beta     = vcov_mat[1:p, 1:p, drop = FALSE],
    fitted.values = as.vector(moda_est),
    residuals     = as.vector(rq_res),
    loglik        = -opt$value,
    n             = n,
    n_par         = p + 1L,
    family        = family,
    cens          = cens,
    call          = match.call(),
    terms         = mt,
    y             = y
  )

  class(res) <- "ModalCens"
  return(res)
}
