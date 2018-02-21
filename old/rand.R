#
# Data matrix

n<-1000
p<-5000
lambda<-c(10, 5, 3, rep(1, p-3))
X<-t(matrix(rnorm(n*p), ncol=n) * lambda )


# Random PCA algorithm 
# Without normalization part
# Get rank 10
n.iter<-10
k<-100
R<-matrix(rnorm(n*2*k), ncol=2*k)

Y<-R
for(i in 1:n.iter){
	
	Y<- X %*% (t(X) %*% Y) / sqrt(n)
	
}


qr.out<-qr(Y)
Q = qr.Q(qr.out)
B = t(Q) %*% X
S = B %*% t(B)
out.eigen<-eigen(S)
U = Q %*% out.eigen$vectors

summary(lm(X[,1] ~ U))
cor(X[,1], U[,1])

svd.out <- svd(X)
U.svd <- svd.out$u
d.svd <- svd.out$d
d <- sqrt(out.eigen$values)
plot(U.svd[,1], U[,1])
plot(d.svd[1:length(d)], d)
abline(0,1)
