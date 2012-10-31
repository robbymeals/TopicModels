## Perform Inference on Simulated LDA Corpus

library(MCMCpack)
library(ggplot2)
library(tm)
library(topicmodels)
library(plyr)
setwd('/home/rmealey/Dropbox/LaPlacianAmbitions/10-TopicModels')

##################################
### 1. Simulate Data

# Default Values
M <- 1000 # number of documents
nTerms <- 100 # number of terms

## document lengths all identical at 100
docLengths <- rep(100,M)

## document lengths (word counts) distributed according to poisson(100)
#docLengths <- rpois(M,100)

## Set additional hyperparameters to some customary values used in LDA priors
#K <- round(nTerms/M) # Number of Topics
K <- 10 # Number of Topics
alphA <- 1/K # parameter for symmetric Document/Topic dirichlet distribution
betA <- 1/K # parameter for Topic/Term dirichlet distribution
AlphA <- rep(alphA, K) # number-of-topics length vector set to symmetric alpha paramater across all topics
BetA <- rep(betA, nTerms) # number-of-terms length vector set to symmetric beta paramater across all terms 

## generate simulated corpus (See script SimulateCorpus.R)
# Returns corpus list object, default 100 documents, 1000 terms, 1000/100=10 topics
# with documents and "true" values for doc/topic matrix and topic/term matrix
source('SimulateCorpus.R') 
corpus <- simulateCorpus(M, nTerms, docLengths, K, alphA, betA)

							  ####
##################################


##################################
### 2. Inference

# A. Using R Package topicmodels:
# Labels
Terms <- paste("Term",seq(nTerms))
Topics <- paste("Topic", seq(K))
Documents <- paste("Document", seq(M))

LDA(corpus[['termFreqMatrix']], K, control=list(alpha=alphA, beta=betA), method='Gibbs') -> lda1

# "Estimated" Document/Topic distribution matrix
Theta_est <- posterior(lda1,corpus[['termFreqMatrix']])$topics
colnames(Theta_est) <- Topics
rownames(Theta_est) <- Documents

# "Estimated" Topic/Term Distribution Matrix
Phi_est <- posterior(lda1,corpus[['termFreqMatrix']])$terms
colnames(Phi_est) <- Terms
rownames(Phi_est) <- Topics

							  ####
##################################

##################################
### 3. Compare "True" to Estimate

Theta_true <- corpus[['Theta']]
Phi_true <- corpus[['Phi']]


							  ####
##################################
