dir <- "../data"
pref <- "tmp"
p <- "100000"
n <- "1000"
m <- "100"
k <- "2"
s <- "1"
mig <- "10"
dirpref <- paste(dir, pref, pref, sep = "/")
dirall <- paste(dirpref, p, n, m, k, s, mig, sep = "_")

V.ref.std.file <- paste0(dirall, ".rand.RefPC.std.V")
V.ref.rand.file <- paste0(dirall, ".rand.RefPC.rand.V")
d.ref.std.file <- paste0(dirall, ".rand.RefPC.std.d")
d.ref.rand.file <- paste0(dirall, ".rand.RefPC.rand.d")
par.ref.rand.file <- paste0(dirall, ".rand.RefPC.rand.par")
par.ref.rand <- scan(par.ref.rand.file, "integer")
randsvd.k <- par.ref.rand[1]
randsvd.niter <- par.ref.rand[2]
V.ref.std <- as.matrix(read.table(V.ref.std.file))
V.ref.std <- V.ref.std[,1:randsvd.k]
V.ref.rand <- as.matrix(read.table(V.ref.rand.file))
d.ref.std <- scan(d.ref.std.file)
d.ref.std <- d.ref.std[1:randsvd.k]
d.ref.rand <- scan(d.ref.rand.file)


V.diff.norm <- norm(V.ref.rand - V.ref.std, "F")
V.diff.norm <- round(V.diff.norm, 2)
d.diff.norm <- sqrt(sum((d.ref.rand - d.ref.std)^2))
d.diff.norm <- round(d.diff.norm, 2)
d.cor <- cor(d.ref.rand, d.ref.std)
d.cor <- round(d.cor, 2)

ii <- 3
par(mfrow=c(ii,ii))
plot(d.ref.std, d.ref.rand,
     main = paste0("Ref e-vals, Cor = ", d.cor))
abline(0,1)
for(i in 1:(ii*ii-1)){
  pcs.ref.std <- V.ref.std[,i]
  pcs.ref.rand <- V.ref.rand[,i]
  pcs.cor <- cor(pcs.ref.std, pcs.ref.rand)
  pcs.cor <- round(pcs.cor, 2)
  ## if(pcs.cor < 0){
  ##   pcs.ref.rand <- pcs.ref.rand * (-1)
  ## }
  main = paste0("PC ", i, ", Cor = ", abs(pcs.cor))
  plot(pcs.ref.std, pcs.ref.rand, main = main,
       xlab = "standard SVD",
       ylab = "randomized SVD")
  abline(0, 1)
}
title(
  paste0(
    "\n",
    "SVD on Ref data",
    "p = ", p,
    ", n = ", n, 
    ", ggs.k = ", k,
    ", ggs.mig = ", mig,
    ", randsvd.k = ", randsvd.k,
    ", randsvd.niter = ", randsvd.niter,
    ", V.diff.norm = ", V.diff.norm,
    ", d.diff.norm = ", d.diff.norm
  ),
  outer = TRUE
)

## pcs.std.1 <- V.ref.std[,1] * d.ref.std[1]
## pcs.std.2 <- V.ref.std[,2] * d.ref.std[2]
## plot(pcs.std.1, pcs.std.2)
## pcs.rand.1 <- V.ref.rand[,1] * d.ref.rand[1]
## pcs.rand.2 <- V.ref.rand[,2] * d.ref.rand[2]
## plot(pcs.rand.1, pcs.rand.2, col = 2)


## par(mfrow=c(2,2))
## plot(1:1000, pcs.std.1)
## points(1:1000, pcs.rand.1, col = 2)
## points(1:1000, pcs.std.2, col = 3)
## points(1:1000, pcs.rand.2, col = 4)
## plot(pcs.std.1, pcs.std.2)
## points(pcs.rand.1, pcs.rand.2, col = 2)
