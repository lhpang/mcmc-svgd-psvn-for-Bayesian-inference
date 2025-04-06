library(caret)
library(coda)
library(mcmcplots)
library(ggplot2)
library(forecast)
library(ggmcmc)
library(Hmisc)
library(psych)
library(MCMCvis)
library(GGally)
library(microbenchmark)



###test





###################################logistic model with gaussian prior##################
logisticModel <- function(b, X){
  if(length(X)>0){
    if(nrow(t(X))==1){
      X = matrix(X,nrow=1)
    }
    et = exp(-(cbind(rep(1, dim(X)[1]), X) %*% b))
  }
  else{
    et = 0
  }
  return(log(1 / (1 + et)))
}


logPriorFunction <- function(x, mu, sigma){ # p(b)
  mu = 0
  sigma = 1
  return(sum(-0.5*(x - mu)^2 / sigma^2))
}

logLikelihoodFunction <- function(b, X, y) { # logistic probability p(X, y | b)
  return((sum(logisticModel(b, X[y==1,])) + sum(1-logisticModel(b, X[y!=1,]))))
}

logPosteriorFunction <- function(b, X, y){ # un-normalized posterior pdf
  return(logPriorFunction(b, X, y) + logLikelihoodFunction(b, X, y)) # return scalar p(b | X, y)
}


#####################################function to generate beta##############################

# inputs(model, prior, likelihood, X, y)
# outputs(samples of beta, stationary distribution of beta)
beta.generate = function(model, prior, likelihood, posterior, X, y){
  
  #data preprocessing
  X = scale(X) # centered features
  pca = prcomp(X, center = TRUE,scale. = TRUE) # pca on centered features (not essential, but helpful)
  X = pca$x[,1:(ncol(X)/2)] # top 1/2 principle component features
  d = dim(X)[2] 
  #sample initialization
  n = 100000 # number of mcmc samples
  n_burnin = n / 10 # 10% burn-in
  samples = matrix(0, nrow=(n+n_burnin), ncol=d+1) # allocation
  currentProbability = posterior(samples[1,], X, y) # mcmc p(b | X, y) sample 1
  proposal_cov_chol = chol(diag(d+1))
  sd = 2.4^2 / d # scaling factor, see Haario et al 2001
  
  for(i in 1:(n + n_burnin-1)){ # vanilla MCMC
    
    if(i==n_burnin){
      proposal_cov_chol = chol(cov(samples[1:n_burnin,])*sd)
    }
    deltaProposal = proposal_cov_chol %*% rnorm(d+1)
    
    proposal = matrix(samples[i,] + deltaProposal, nrow=d+1)
    proposalProbability = posterior(proposal, X, y)
    logAlpha = (proposalProbability - currentProbability) # log alpha = log proposalP - log currentP
    
    logU = log(runif(1)) # log u
    acceptance = logAlpha >= logU # bool
    
    if(is.na(acceptance)){ # if nan
      samples[i+1,] = rnorm(d+1, 0, 1) # if nan then start chain at random value
      currentProbability = posterior(samples[i+1,], X, y)
    }
    
    else if(acceptance){ # if accept
      # print(TRUE)
      samples[i+1,] = proposal
      currentProbability = proposalProbability
    }
    
    else if(!acceptance){ # if reject
      # print(FALSE)
      samples[i+1,] = samples[i,]
    }
  }
  sample_stationary = samples[(n_burnin+1):(n+n_burnin),]
  return(sample_stationary)
}



#implementation
breastCancerDataset <- read.csv("breast_cancer_data.csv") #load raw dataset
y = as.double(data.matrix(breastCancerDataset['diagnosis'] == 'M')) # labels
X = data.matrix(breastCancerDataset[, 3:32])
print(microbenchmark(beta.generate(logisticModel, logPriorFunction, logLikelihoodFunction, logPosteriorFunction, X, y)
))
samples = beta.generate(logisticModel, logPriorFunction, logLikelihoodFunction, logPosteriorFunction, X, y)


#visualization
MCMCtrace(samples) #traceplot and density plot 

MCMCplot(samples,
         horiz = FALSE,
         rank = TRUE,
         ref_ovl = FALSE,
         xlab = 'My x-axis label', 
         main = 'MCMCvis plot',
) #caterpillar plots for posterior estimates

for(i in 1:ncol(samples)){
  autplot1(as.mcmc.list(as.mcmc(samples[,i])))
  
} #autocorrelation plots

#ggpairs(data.frame(samples))


#####################################function to get prediction##############################

# inputs(sample, X)
# outputs(prediction for y)
prediction = function(model, likelihood, sample, X, selected.clusters){
  n = nrow(X)
  k = length(selected.clusters)
  prob_mat = matrix(NA,n,k)
  
  #preprocessing
  X = scale(X) # centered features
  pca = prcomp(X, center = TRUE,scale. = TRUE) # pca on centered features (not essential, but helpful)
  X = pca$x[,1:(ncol(X)/2)] # top 1/2 principle component features
  
  for(i in 1:n){
    for(j in 1:k){
      prob_mat[i,j] = likelihood(t(t(sample[i,])),t(X[i,]),selected.clusters[j])
    }
  }
  y_pred = selected.clusters[apply(prob_mat,1,which.max)]
  return(y_pred)
}

#evaluation
print(microbenchmark(prediction(logisticModel, logLikelihoodFunction, samples, X, c(0,1))))
y_pred = prediction(logisticModel, logLikelihoodFunction, samples, X, c(0,1))
dim(samples)

accuracy = mean(near(y_pred,y))

#pairs(samples)
# MCMC diagnostics:
#   Trace, autocorrelation
# 
# SVGD / pSVN diagnostics:
#   Loss vs epoch
#   ||grad|| vs epoch
# 
# Visual Comparison:
#   Lower triangular matrix of sample histograms
# 
#   1vs1 
#   1vs2  2vs2
#   . 
#   .
#   1vs16 2vs16 3vs16 4vs16 ...  16vs16


# gaussian prior fn
# laplace prior fn

# logistic likelihood fn
# gamma likelihood fn
# gaussian likelihood fn

# logistic regression model
# gaussian regression model
# gamma regression model

# 3 main:
#   MCMC (baseline comparison)
#   SVGD (novel implementation)
#   pSVN (novel implementation)