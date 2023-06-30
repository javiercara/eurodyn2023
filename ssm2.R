# modelo con Q,R,S
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

y0 = read.table("simula.txt", sep=",", col.names = "")
y = y0$X # tiene que ser un vector
dt = 0.05
f0 = 4 # Hz
z0 = 0.02

d = list(y=y, nt=length(y), nx = 2, ny = 1, dt = dt, 
         m1 = matrix(0, nrow = 2, ncol = 1),
         P1 = matrix(0, nrow = 2, ncol = 2))

#(m <- rstan::stan(file="ssm1.stan", data=d, iter=5000, seed=123,
#                  control=list(adapt_delta = 0.999)))

# opciones por defecto
# m1 = stan(file="ssm2.stan", data=d, seed=123)

# solo una cadena y 1000 iteraciones
m1 = stan(file="ssm2.stan", data=d, seed=123, cores = 2, chains = 1, warmup = 1000, iter = 5000)
save("m1", file = "ssm2.RData" )

load("ssm2.RData")
(msum = summary(m1))

# comprobaciones
library(bayesplot)

# Muestras posteriori
posterior <- as.matrix(m1)

# Información extra
logPosterior <- log_posterior(m1)
nutsParams <- nuts_params(m1)

# Posterior uncertainty intervals
mcmc_intervals(posterior, pars = c("f"))

mcmc_areas(posterior, 
           pars = c("f"), # parameters
           prob= 0.9, # 80% intervals
           prob_outer = 0.99,
           point_est = "mean")

mcmc_areas(posterior, 
           pars = c("z"), # parameters
           prob= 0.9, # 80% intervals
           prob_outer = 0.99,
           point_est = "mean")

mcmc_trace(posterior, pars = c("f","z"))

# extract samples from a fitted stan model
m_samples = extract(m1, permuted = F, pars = c("f", "z"))
ff = as.vector(m_samples[,,1])

hist(ff, freq = F, xlab = "freq (Hz)", main = "Histogram of f|Y", ylim = c(0,30))
curve(dnorm(x,msum$summary[1,1],msum$summary[1,3]), add = T, lwd = 1, col = "red")

zz = as.vector(m_samples[,,2])
hist(zz)

pdf(file = "posterior.pdf", width = 10, height = 8)
par(mfrow = c(2,1))
hist(ff, freq = F, xlab = "freq (Hz)", main = "Histogram of f|Y", ylim = c(0,30))
curve(dnorm(x,msum$summary[1,1],msum$summary[1,3]), add = T, lwd = 1, col = "red")
#
hist(zz, freq = F, xlab = "z (Hz)", main = "Histogram of z|Y", ylim = c(0,120))
curve(dnorm(x,msum$summary[2,1],msum$summary[2,3]), add = T, lwd = 1, col = "red")
dev.off()

# estimacion maxima verosimilitud CREO QUE SE PUEDE HACER CON STAN DIRECTAMENTE
#source("ACQR_dare.R")
#source("dare.R")
#source("ACQR_kfilter_s.R")
source("ACQR_kfilter.R")
source("ACQRS_kfilter.R")
library(Matrix)

logLikelihood <- function(param,y,dt){
  #
  f = param[1]
  w = 2*pi*f
  z = param[2]
  H = param[3]^2
  E = param[4]^2
  
  # A
  Ac = matrix(c(0,-w^2,1,-2*z*w), nrow = 2)
  A1 = expm(Ac*dt)
  A = matrix(A1@x, nrow = 2)
  
  # C
  C = matrix(c(-w^2,-2*z*w), nrow = 1)
  
  # Q
  Bc = matrix(c(0,1), ncol = 1)
  B = (A - diag(2)) %*% solve(Ac) %*% Bc
  Q = (B %*% t(B))*H
  
  # R
  R = H + E
  
  # S
  S = B*H
  
  # m1 is not estimated
  m1 = rep(0,2)
  P1 = matrix(0, nrow = 2, ncol = 2)
  
  nx = 2
  ny = 1
  #kf = ACQR_kfilter(y,A,C,Q,R,m1,P1,nx,ny)
  kf = ACQRS_kfilter(y,A,C,Q,R,S,m1,P1,nx,ny)
  
  logL = kf$loglik + dnorm(f,4,1,log=T) + dbeta(z,2,50,log=T) + dcauchy(H,0,5,log=T) + dcauchy(E,0,5,log=T)
  
  return(-logL) # minus loglik because optim finds the minimum
}

init_par = c(4,0.02,1,1)
# checking the likelihood
logLikelihood(init_par,y,dt)

mle = optim(par = init_par, fn = logLikelihood, y = y, dt = dt, gr = NULL, 
            method = "BFGS", control = list(trace=1, REPORT = 1, maxit = 100))
mle$par
hess = optimHess(mle$par, fn = logLikelihood, y = y, dt = dt)
hess_inv = solve(hess)
# standar deviations
sqrt(diag(hess_inv))

# comparison
hist(ff, freq = F)
curve(dnorm(x,msum$summary[1,1],msum$summary[1,3]), add = T, lwd = 1, col = "red")
curve(dnorm(x, mean = mle$par[1], sd = sqrt(hess_inv[1,1])), add = T)

hist(zz, freq = F)
curve(dnorm(x, mean = mle$par[2], sd = sqrt(hess_inv[2,2])), add = T)
curve(dnorm(x,msum$summary[2,1],msum$summary[2,3]), add = T, lwd = 1, col = "red")

