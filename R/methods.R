#' @export
print.ModalCens <- function(x, ...) {
  cat("\nCall:\n", paste(deparse(x$call), sep = "\n", collapse = "\n"), "\n\n", sep = "")
  cat("Coefficients (Modal Mode):\n")
  print(x$coefficients)
  cat("\n")
}

#' @importFrom stats cor
#' @export
summary.ModalCens <- function(object, ...) {
  cat("\n--- Modal Regression Summary ---\n")
  cat("Family:", object$family, "\n")
  cat("N:", object$n, "(Censored:", sum(object$cens == 1),")\n\n")

  se <- sqrt(diag(object$vcov))
  n_beta <- length(object$coefficients)

  coef_mat <- data.frame(
    Estimate = object$coefficients,
    Std.Error = se[1:n_beta],
    z.value = object$coefficients / se[1:n_beta],
    p.value = 2 * (1 - pnorm(abs(object$coefficients / se[1:n_beta])))
  )

  print(coef_mat)
  cat("\nDispersion Parameter (phi):", round(object$phi, 5), "\n")
  cat("---\n")

  # Cálculo de métricas inferiores
  ll <- object$loglik
  k <- object$n_par
  aic <- -2 * ll + 2 * k
  bic <- -2 * ll + k * log(object$n)
  pseudo_r2 <- cor(object$fitted.values, object$y)^2

  cat("Log-likelihood:", round(ll, 4), "\n")
  cat("AIC:", round(aic, 4), "| BIC:", round(bic, 4), "\n")
  cat("Pseudo R-squared:", round(pseudo_r2, 4), "\n")
}

#' @export
logLik.ModalCens <- function(object, ...) {
  val <- object$loglik
  attr(val, "df") <- object$n_par
  attr(val, "nobs") <- object$n
  class(val) <- "logLik"
  return(val)
}

#' @export
vcov.ModalCens <- function(object, ...) {
  n_beta <- length(object$coefficients)
  return(object$vcov[1:n_beta, 1:n_beta])
}

#' @export
confint.ModalCens <- function(object, parm, level = 0.95, ...) {
  cf <- coef(object)
  pnames <- names(cf)

  if (missing(parm)) parm <- pnames
  else if (is.numeric(parm)) parm <- pnames[parm]

  a <- (1 - level)/2
  a <- c(a, 1 - a)
  se <- sqrt(diag(vcov(object)))[pnames]  # solo los p coeficientes
  pct <- paste(format(100 * a, trim = TRUE, scientific = FALSE, digits = 3), "%")

  ci <- array(NA, dim = c(length(parm), 2), dimnames = list(parm, pct))
  z_crit <- qnorm(a[2])

  ci[, 1] <- cf[parm] - z_crit * se[parm]
  ci[, 2] <- cf[parm] + z_crit * se[parm]

  return(ci)
}

#' @importFrom graphics par plot abline legend points polygon
#' @importFrom stats ppoints rnorm quantile qqnorm qqline
#' @importFrom grDevices adjustcolor
#' @importFrom stats ppoints rnorm quantile
#' @export
plot.ModalCens <- function(x, ...) {
  # Guardamos todos los parámetros originales
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  # Dividimos la pantalla y apagamos explícitamente la pausa (ask = FALSE)
  par(mfrow = c(1, 2), ask = FALSE)

  # Gráfico 1
  plot(x$fitted.values, x$residuals,
       pch = 19, col = ifelse(x$cens == 0, "steelblue", "orange"),
       main = "Residuals vs Fitted", xlab = "Fitted Mode", ylab = "Residuals")
  abline(h = 0, lty = 2, col = "red")
  legend("topright", legend = c("Observed", "Censored"),
         col = c("steelblue", "orange"), pch = 19, cex = 0.7)

  # Gráfico 2: Q-Q con envelope 95%
  n    <- length(x$residuals)
  r    <- sort(x$residuals)
  probs <- ppoints(n)
  theo  <- qnorm(probs)

  # Envelope simulado (999 muestras bajo H0: normalidad)
  sim  <- replicate(999, sort(rnorm(n)))
  env_low  <- apply(sim, 1, quantile, probs = 0.025)
  env_high <- apply(sim, 1, quantile, probs = 0.975)

  qqnorm(x$residuals, main = "Normal Q-Q Plot", pch = 19,
         ylim = range(c(r, env_low, env_high)))
  polygon(c(theo, rev(theo)), c(env_high, rev(env_low)),
          col = adjustcolor("steelblue", alpha.f = 0.15), border = NA)
  qqline(x$residuals, col = "red")
  points(theo, r, pch = 19, cex = 0.8)  # redibujar puntos encima del envelope
}

#' @export
coef.ModalCens <- function(object, ...) object$coefficients

#' @export
residuals.ModalCens <- function(object, ...) object$residuals
