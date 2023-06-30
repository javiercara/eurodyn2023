//
functions {

real kfilter_logL(vector y, matrix A, matrix C, matrix Q, 
                  real R, matrix S, matrix x10, matrix P10) {
  real logL;
  int nt = num_elements(y);
  int nx = cols(C);
  int ny = rows(C);
  
  matrix[nx,1] xtt;
  matrix[nx,nx] Ptt;
  matrix[nx,1] xtt1;
  matrix[nx,nx] Ptt1;
  real et;
  real St;
  real Stinv;
  matrix[nx,ny] Kt;
  
  logL = - 0.5*ny*nt*log(2*pi());
  xtt1 = x10;
  Ptt1 = P10;
  for (t in 1:nt) {
    // innovations
    et = y[t] - (C*xtt1)[1,1];
    St = quad_form(Ptt1,C')[1,1] + R;
    
    // Kalman gain
    Stinv = 1/St;
    Kt = (A*Ptt1*C'+S)*Stinv;
    
    // one-step ahead prediction
    xtt1 = A*xtt1 + Kt*et;
    Ptt1 = quad_form(Ptt1,A') + Q - Kt*St*Kt';
    
    // likelihood
    logL += -0.5*(log(fabs(St)) + et*Stinv*et);
  }
  
  return logL;
}

real ssm2_lpdf(vector y, real f, real z, real dt, real H, 
               real E, matrix m1, matrix P1){
  
  real w = 2*pi()*f;
  matrix[2,2] Ac;
  matrix[2,2] A;
  matrix[1,2] C;
  matrix[2,1] B0;
  matrix[2,2] I2;
  matrix[2,1] B;
  matrix[2,2] Q;
  real R;
  matrix[2,1] S;
  matrix[2,1] x10 = m1;
  matrix[2,2] P10 = P1;
  
  // A matrix
  Ac[1,1] = 0;
  Ac[1,2] = 1;
  Ac[2,1] = -square(w);
  Ac[2,2] = -2*z*w;
  A = matrix_exp(Ac*dt);
  
  // C matrix
  C[1,1] = -square(w);
  C[1,2] = -2*z*w;
  
  // Q matrix
  B0[1,1] = 0;
  B0[2,1] = 1;
  I2 = diag_matrix(rep_vector(1.0, 2));
  B = (A - I2)*inverse(Ac)*B0;
  Q = (B*B')*H;
  
  // R
  R = H + E;
  
  // S
  S = B*H;
  
  return kfilter_logL(y, A, C, Q, R, S, x10, P10);
}
  
} 
// end funciones

// The input data is a vector 'y' of length 'N'.
data {
  int nt;
  int nx;
  int ny;
  vector[nt] y;
  real dt;
  matrix[nx,1] m1;
  matrix[nx,nx] P1;
}

// The parameters to be estimated
parameters {
  real<lower=0.0, upper=1/(2*dt)> f;
  real<lower=0.0, upper=1> z;
  real<lower=0.0> H;
  real<lower=0.0> E;
}

// transformed parameters

// The model 
model {
  // priors
  f ~ normal(4,1);
  z ~ beta(2,50);
  H ~ cauchy(0,5);
  E ~ cauchy(0,5);
  
  // likelihood
  target += ssm2_lpdf(y | f, z, dt, H, E, m1, P1);
}

