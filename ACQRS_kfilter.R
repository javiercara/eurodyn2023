#'
#' Kalman filter for ACQRS model (m=m, n=n)
#'
#' Kalman filter for ACQRS model (m=m, n=n)
#'
#' @param y: time series
#' @param smm: list defining the state space model
#'
#' @return list(xtt,Ptt,xtt1,Ptt1,et,St,Kt,loglik)
#'
#' @export
#'
ACQRS_kfilter <- function(y,A,C,Q,R,S,x10,P10,nx,ny){

  # data as matrices
  y = as.matrix(y)
  if (nrow(y) != ny){
    y = t(y)
  }
  nt = ncol(y)

  # allocation
  xtt <- array(0,c(nx,nt))
  Ptt <- array(0,c(nx,nx,nt))
  xtt1 <- array(0,c(nx,nt+1))
  Ptt1 <- array(0,c(nx,nx,nt+1))
  et <- array(0,c(ny,nt))
  St <- array(0,c(ny,ny,nt))
  Kt <- array(0,c(nx,ny,nt))
  Kt1 <- array(0,c(nx,ny,nt))
  loglik <- 0.0

  # Filter
  xtt1[,1] <- x10
  Ptt1[,,1] <- P10
  for (t in 1:nt){

    #  innovations
    et[,t] = y[,t] - C %*% matrix(xtt1[,t],ncol=1)
    St[,,t] = C %*% Ptt1[,,t] %*% t(C) + R # et variance

    # Kalman gain
    if (ny==1){ Stinv=1/St[,,t] }
    else{ Stinv = solve(St[,,t]) }
    Kt[,,t] = Ptt1[,,t] %*% t(C) %*% Stinv
    Kt1[,,t] = (A %*% Ptt1[,,t] %*% t(C) + S) %*% Stinv

    # filtered values
    xtt[,t] = xtt1[,t] + Kt[,,t] %*% matrix(et[,t],ncol=1)
    Ptt[,,t] = (diag(nx) - Kt[,,t] %*% C) %*% Ptt1[,,t]

    # one-step ahead prediction
    xtt1[,t+1] = A %*% xtt1[,t] + Kt1[,,t] %*% matrix(et[,t],ncol=1)
    Ptt1[,,t+1] = A %*% Ptt1[,,t] %*% t(A) + Q - Kt1[,,t] %*% matrix(St[,,t],ncol=ny) %*% t(Kt1[,,t])

    # likelihood
    if (ny==1){
      Stdet = St[,,t]
    }
    else{
      Stdet = det(St[,,t])
    }
    loglik = loglik + log(Stdet) + t(et[,t]) %*% Stinv %*% et[,t]
  }

  loglik =  - ny*nx/2*log(2*pi) - 0.5*loglik

  return(list(xtt=xtt,Ptt=Ptt,xtt1=xtt1,Ptt1=Ptt1,et=et,St=St,Kt=Kt,loglik=loglik))

}

