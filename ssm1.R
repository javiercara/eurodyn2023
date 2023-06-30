# modelo con Q y R
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
m1 = stan(file="ssm1.stan", data=d, seed=123, cores = 2, chains = 1, iter = 5000)

summary(m1)

# comprobaciones
library(bayesplot)

# Muestras posteriori
posterior <- as.array(m1)

# Información extra
logPosterior <- log_posterior(m1)
nutsParams <- nuts_params(m1)

# Posterior uncertainty intervals
mcmc_intervals(posterior, pars = c("f"))

mcmc_areas(posterior, 
           pars = c("f"), # parameters
           prob= 0.8, # 80% intervals
           prob_outer = 0.99,
           point_est = "mean")

mcmc_areas(posterior, 
           pars = c("z"), # parameters
           prob= 0.8, # 80% intervals
           prob_outer = 0.99,
           point_est = "mean")

mcmc_trace(posterior, pars = c("f","z"))

# extract samples from a fitted stan model
m_samples = extract(m1, permuted = F, pars = c("f", "z"))
ff = as.vector(m_samples[,,1])
hist(ff)

zz = as.vector(m_samples[,,2])
hist(zz)

# estimacion maxima verosimilitud CREO QUE SE PUEDE HACER CON STAN DIRECTAMENTE
source("ACQR_dare.R")
source("dare.R")
source("ACQR_kfilter_s.R")
source("ACQR_kfilter.R")
library(Matrix)

logLikelihood0 <- function(param,y,dt){
  #
  w = 2*pi*param[1]
  z = param[2]
  
  # A
  Ac = matrix(c(0,-w^2,1,-2*z*w), nrow = 2)
  A1 = expm(Ac*dt)
  A = matrix(A1@x, nrow = 2)
  
  # C
  C = matrix(c(-w^2,-2*z*w), nrow = 1)

  # Q
  LQ = matrix(0, nrow = 2, ncol = 2)
  LQ[1,1] = param[3]
  LQ[2,1] = param[4]
  LQ[2,2] = param[5]
  Q = LQ %*% t(LQ)
  
  # R
  R = param[6]^2
  
  # m1 is not estimated
  m1 = rep(0,2)
  P1 = matrix(0, nrow = 2, ncol = 2)
  
  nx = 2
  ny = 1
  #kf = ACQR_kfilter_s(y,A,C,Q,R,m1,nx,ny)
  kf = ACQR_kfilter(y,A,C,Q,R,m1,P1,nx,ny)

  return(-kf$loglik) # minus loglik because optim finds the minimum
}

logLikelihood <- function(param,y,dt){
  #
  w = 2*pi*param[1]
  z = param[2]
  
  # A
  Ac = matrix(c(0,-w^2,1,-2*z*w), nrow = 2)
  A1 = expm(Ac*dt)
  A = matrix(A1@x, nrow = 2)
  
  # C
  C = matrix(c(-w^2,-2*z*w), nrow = 1)
  
  # Q
  B0 = matrix(c(0,1), ncol = 1)
  B = (A - diag(2)) %*% solve(Ac) %*% B0
  Q = (B %*% t(B))*param[3]^2
  
  # R
  R = param[3]^2 + param[4]^2
  
  # m1 is not estimated
  m1 = rep(0,2)
  P1 = matrix(0, nrow = 2, ncol = 2)
  
  nx = 2
  ny = 1
  #kf = ACQR_kfilter_s(y,A,C,Q,R,m1,nx,ny)
  kf = ACQR_kfilter(y,A,C,Q,R,m1,P1,nx,ny)
  
  return(-kf$loglik) # minus loglik because optim finds the minimum
}

init_par = c(4,0.02,1,1)
mle = optim(par = init_par, fn = logLikelihood, y = y, dt = dt, gr = NULL, 
            method = "Nelder-Mead", control = list(trace=1, REPORT = 1, maxit = 2000))
mle$par
hess = optimHess(mle$par, fn = logLikelihood, y = y, dt = dt)
hess_inv = solve(hess)

# comparison
hist(ff, freq = F)
curve(dnorm(x, mean = mle$par[1], sd = sqrt(hess_inv[1,1])), add = T,
      from = 3.95, to = 4.10)

hist(zz, freq = F)
curve(dnorm(x, mean = mle$par[2], sd = sqrt(hess_inv[2,2])), add = T,
      from = 0.01, to = 0.04)

