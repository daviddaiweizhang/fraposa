rm(list=ls())

plot.m <- function(pref.std, pref.rand, items){
  idxs <- c(items.idxs[, items])
  ## idxs <- c(12:15)
  p <- 1e5
  n <- 600
  ver.std <- "comb"
  # Y.std <- X[X$pref==pref.std & X$ver==ver.std & X$n==n & X$m >= 1000 & X$m <= 3000,]
  Y.std <- X[X$pref==pref.std,]
  Y.std <- Y.std[order(Y.std$m),]
  ver.rand <- "rand"
  # Y.rand <- X[X$pref==pref.rand & X$ver==ver.rand & X$n==n & X$m >= 1000 & X$m <= 3000,]
  Y.rand <- X[X$pref==pref.rand,]
  Y.rand <- Y.rand[order(Y.rand$m),]
  main <- paste0(items, ", p=", toString(p), ", n=", toString(n), ", mig=100")
  pch = c(rep(1,4), rep(2,4))
  lty = 1
  col = c(1:4, 1:4)
  cex = 1.5
  ylab = items
  Y.cbind <- cbind(Y.std[, idxs], Y.rand[, idxs])
  matplot(Y.std$m, Y.cbind, type="b", lty=lty, pch=pch, col=col, xlab="m", ylab=ylab, cex=cex, main=main)
  legend("topleft",
        legend = methods,
          pch = pch,
          col = col,
        cex = cex,
  )
}

plot.n <- function(pref.std, pref.rand, items){
  idxs <- c(items.idxs[, items])
  ## idxs <- c(12:15)
  p <- 1e5
  m <- 200
  ver.std <- "comb"
  # Y.std <- X[X$pref==pref.std & X$ver==ver.std & X$n==n & X$m >= 1000 & X$m <= 3000,]
  Y.std <- X[X$pref==pref.std,]
  Y.std <- Y.std[order(Y.std$n),]
  ver.rand <- "rand"
  # Y.rand <- X[X$pref==pref.rand & X$ver==ver.rand & X$n==n & X$m >= 1000 & X$m <= 3000,]
  Y.rand <- X[X$pref==pref.rand,]
  Y.rand <- Y.rand[order(Y.rand$n),]
  main <- paste0(items, ", p=", toString(p), ", m=", toString(m), ", mig=100")
  pch = c(rep(1,4), rep(2,4))
  lty = 1
  col = c(1:4, 1:4)
  cex = 1.5
  ylab = items
  Y.cbind <- cbind(Y.std[, idxs], Y.rand[, idxs])
  matplot(Y.std$n, Y.cbind, type="b", lty=lty, pch=pch, col=col, xlab="n", ylab=ylab, cex=cex, main=main)
  legend("topleft",
        legend = methods,
          pch = pch,
          col = col,
        cex = cex,
  )
}


methods <- c("trace", "onl", "proj", "hdpca")
methods.std = paste(methods, "std", sep = " ")
methods.rand = paste(methods, "rand", sep = " ")
methods = c(methods.std, methods.rand)
items.idxs <- data.frame(
  runtimes = 12:15,
  err_refcenter = 17:20,
  err_trace = 22:25
)

name <- "pcSummary"
pdf.file <- paste0(name, '.pdf')
X <- read.table("pcSummary", header=T, sep="\t", stringsAsFactors=F)
## X.agg <- aggregate(cbind(X$ver=="rand", X$n!=600), by=list(X$pref), FUN=min)
## colnames(X.agg) <- c("slurmJobID", "randomized", "nChanging")
## freq <- c(unname(table(X$pref)))
## X.agg <- cbind(X.agg, freq)
## X.agg <- X.agg[order(X.agg$randomized, X.agg$nChanging),]
## X.agg
## table(X.agg$randomized, X.agg$nChanging)
## table(X$pref)
# complete.idxs <- (table(X$pref)==5)
# complete.id <-  names(complete.idxs[complete.idxs])
# X <- X[X$pref %in% complete.id, ]
## X <- X[X$pref == "1640156/1640156" | X$pref == "1640156/1640156"]
X$pref <- substr(X$pref, 1, 8)
X[order(X$ver, X$pref), c("ver", "pref", "n", "m")]

pdf("pcSummary.pdf", width = 16, height = 9)
par(mfrow = c(1,2))
for(i in 1:3){
  pref.std.n <- "26746977"
  pref.rand.n <- "26746979"
  pref.std.m <- "26745070"
  pref.rand.m <- "26745314"
  items <- colnames(items.idxs)[i]
  plot.n(pref.std.n, pref.rand.n, items)
  plot.m(pref.std.m, pref.rand.m, items)
}
dev.off()

p <- 1e5
m <- 200
Y <- X[X$ver==ver & X$pref=="ggsim/ggsim" & X$p==p & X$n>=1000 & X$n <=3000 & X$m==m,]
Y <- Y[order(Y$n),]
Y.old <- X.old[X.old$ver==ver.old & X.old$pref=="ggsim/ggsim" & X.old$p==p & X.old$m==m & X.old$n >= 1000 & X.old$n <= 3000,]
Y.old<- Y.old[order(Y.old$m),]
items <- c("runtimes")
idxs <- c(12:15)
idxs.old <- c(10:13)
main <- paste0("p=", toString(p), ", m=", toString(m), ", mig=100")
main <- paste(main, ver, sep = ", ")
ylab = items
Y.cbind <- cbind(Y[, idxs], Y.old[, idxs.old])
matplot(Y$n, Y.cbind, type="b", lty=lty, pch=pch, col=col, xlab="m", ylab=ylab, cex=cex, main=main)
legend("topleft",
        legend = methods,
        pch = pch,
        col = col,
        cex = cex
)
dev.off()


pdf("err_refcenter.pdf", width = 16, height = 9)
methods <- c("trace", "onl", "proj", "hdpca")

par(mfrow = c(1,2))
p <- 1e5
n <- 600
Y <- X[X$ver==ver & X$pref=="ggsim/ggsim" & X$p==p & X$n==n & X$m >= 1000 & X$m <= 3000,]
Y <- Y[order(Y$m),]
Y.old <- X.old[X.old$ver==ver.old & X.old$pref=="ggsim/ggsim" & X.old$p==p & X.old$n==n & X.old$m >= 1000 & X.old$m <= 3000,]
Y.old<- Y.old[order(Y.old$m),]
items <- c("err_refcenter")
idxs <- c(17:20)
idxs.old <- c(15:18)
main <- paste0("p=", toString(p), ", n=", toString(n), ", mig=100")
main <- paste(main, ver, sep = ", ")
pch = c(rep(1,4), rep(2,4))
lty = 1
col = c(1:4, 1:4)
cex = 1.5
methods = paste(methods, "rand", sep = " ")
methods.old = paste(methods, "std", sep = " ")
methods = c(methods, methods.old)
ylab = items
Y.cbind <- cbind(Y[, idxs], Y.old[, idxs.old])
matplot(Y$m, Y.cbind, type="b", lty=lty, pch=pch, col=col, xlab="m", ylab=ylab, cex=cex, main=main)
legend("topleft",
       legend = methods,
        pch = pch,
        col = col,
       cex = cex,
)




p <- 1e5
m <- 200
Y <- X[X$ver==ver & X$pref=="ggsim/ggsim" & X$p==p & X$n>=1000 & X$n <=3000 & X$m==m,]
Y <- Y[order(Y$n),]
Y.old <- X.old[X.old$ver==ver.old & X.old$pref=="ggsim/ggsim" & X.old$p==p & X.old$m==m & X.old$n >= 1000 & X.old$n <= 3000,]
Y.old<- Y.old[order(Y.old$m),]
items <- c("err_refcenter")
main <- paste0("p=", toString(p), ", m=", toString(m), ", mig=100")
main <- paste(main, ver, sep = ", ")
ylab = items
Y.cbind <- cbind(Y[, idxs], Y.old[, idxs.old])
matplot(Y$n, Y.cbind, type="b", lty=lty, pch=pch, col=col, xlab="m", ylab=ylab, cex=cex, main=main)
legend("topleft",
        legend = methods,
        pch = pch,
        col = col,
        cex = cex
)
dev.off()


pdf("err_trace.pdf", width = 16, height = 9)
methods <- c("trace", "onl", "proj", "hdpca")

par(mfrow = c(1,2))
p <- 1e5
n <- 600
Y <- X[X$ver==ver & X$pref=="ggsim/ggsim" & X$p==p & X$n==n & X$m >= 1000 & X$m <= 3000,]
Y <- Y[order(Y$m),]
Y.old <- X.old[X.old$ver==ver.old & X.old$pref=="ggsim/ggsim" & X.old$p==p & X.old$n==n & X.old$m >= 1000 & X.old$m <= 3000,]
Y.old<- Y.old[order(Y.old$m),]
items <- c("err_trace")
idxs <- c(22:25)
idxs.old <- c(20:23)
main <- paste0("p=", toString(p), ", n=", toString(n), ", mig=100")
main <- paste(main, ver, sep = ", ")
pch = c(rep(1,4), rep(2,4))
lty = 1
col = c(1:4, 1:4)
cex = 1.5
methods = paste(methods, "rand", sep = " ")
methods.old = paste(methods, "std", sep = " ")
methods = c(methods, methods.old)
ylab = items
Y.cbind <- cbind(Y[, idxs], Y.old[, idxs.old])
matplot(Y$m, Y.cbind, type="b", lty=lty, pch=pch, col=col, xlab="m", ylab=ylab, cex=cex, main=main)
legend("topleft",
       legend = methods,
        pch = pch,
        col = col,
       cex = cex,
)




p <- 1e5
m <- 200
Y <- X[X$ver==ver & X$pref=="ggsim/ggsim" & X$p==p & X$n>=1000 & X$n <=3000 & X$m==m,]
Y <- Y[order(Y$n),]
Y.old <- X.old[X.old$ver==ver.old & X.old$pref=="ggsim/ggsim" & X.old$p==p & X.old$m==m & X.old$n >= 1000 & X.old$n <= 3000,]
Y.old<- Y.old[order(Y.old$m),]
items <- c("err_refcenter")
main <- paste0("p=", toString(p), ", m=", toString(m), ", mig=100")
main <- paste(main, ver, sep = ", ")
ylab = items
Y.cbind <- cbind(Y[, idxs], Y.old[, idxs.old])
matplot(Y$n, Y.cbind, type="b", lty=lty, pch=pch, col=col, xlab="m", ylab=ylab, cex=cex, main=main)
legend("topleft",
        legend = methods,
        pch = pch,
        col = col,
        cex = cex
)
dev.off()


######################################################
pdf("err_refcenter.pdf", width = 16, height = 9)
par(mfrow = c(1,2))
p <- 1e5
n <- 600
Y <- X[X$ver==ver & X$pref=="ggsim/ggsim" & X$p==p & X$n==n & X$m >= 1000 & X$m <= 3000,]
Y <- Y[order(Y$m),]
items <- c("err.refcenter")
idxs <- c(16:19)
main <- paste0("p=", toString(p), ", n=", toString(n), ", mig=100")
main <- paste(main, ver, sep = ", ")
pch = 1:4
col = 1:4
cex = 1.5
ylab = items
matplot(Y$m, Y[, idxs], type="b", pch=pch, col=col, xlab="m", ylab=ylab, cex=cex, main=main)
legend("topleft",
        legend = methods,
        pch = pch,
        col = col,
        cex = cex
)

p <- 1e5
m <- 200
Y <- X[X$ver==ver & X$pref=="ggsim/ggsim" & X$p==p & X$n>=1000 & X$n <= 3000 & X$m==m,]
Y <- Y[order(Y$n),]
items <- c("err.refcenter")
idxs <- c(16:19)
main <- paste0("p=", toString(p), ", m=", toString(m), ", mig=100")
main <- paste(main, ver, sep = ", ")
pch = 1:4
col = 1:4
cex = 1.5
ylab = items
matplot(Y$n, Y[, idxs], type="b", pch=pch, col=col, xlab="n", ylab=ylab, cex=cex, main=main)
legend("topleft",
        legend = methods,
        pch = pch,
        col = col,
        cex = cex
)
dev.off()


pdf("err_online.pdf", width = 16, height = 9)
par(mfrow = c(1,2))
p <- 1e5
n <- 600
Y <- X[X$ver==ver & X$pref=="ggsim/ggsim" & X$p==p & X$n==n & X$m >= 1000 & X$m <= 3000,]
Y <- Y[order(Y$m),]
items <- c("err.online")
idxs <- c(21:24)
main <- paste0("p=", toString(p), ", n=", toString(n), ", mig=100")
main <- paste(main, ver, sep = ", ")
pch = 1:4
col = 1:4
cex = 1.5
ylab = items
matplot(Y$m, Y[, idxs], type="b", pch=pch, col=col, xlab="m", ylab=ylab, cex=cex, main=main)
legend("topleft",
        legend = methods,
        pch = pch,
        col = col,
        cex = cex
)

p <- 1e5
m <- 200
Y <- X[X$ver==ver & X$pref=="ggsim/ggsim" & X$p==p & X$n>=1000 & X$n <=3000 & X$m==m,]
Y <- Y[order(Y$n),]
items <- c("err.online")
idxs <- c(21:24)
main <- paste0("p=", toString(p), ", m=", toString(m), ", mig=100")
main <- paste(main, ver, sep = ", ")
pch = 1:4
col = 1:4
cex = 1.5
ylab = items
matplot(Y$n, Y[, idxs], type="b", pch=pch, col=col, xlab="n", ylab=ylab, cex=cex, main=main)
legend("topleft",
        legend = methods,
        pch = pch,
        col = col,
        cex = cex
)
dev.off()
## Y <- X[X$ver=="comb" & X$pref=="ggsim/ggsim" & X$p==1e5 & X$n >= 1000 & X$m==200,]
## items <- c("runtimes")
## idxs <- matrix(c(10:13), nrow=1)
## ps <- c(100000)
## for(i in 1:length(ps)){
##     for(j in 1:nrow(idxs)){
##         idx <- idxs[j,]
##         p = ps[i]
##         main = paste0("p=", toString(p), ", m=200")
##         pch = 1:4
##         col = 1:4
##         cex = 1.5
##         ylab = items[j]
##         matplot(Y$n[Y$p==p], Y[Y$p==p, idx], type="b", pch=pch, col=col, xlab="n", ylab=ylab, cex=cex)
##         legend("topleft",
##                title = main,
##                ## legend = names(Y[,idx]),
##                legend = methods,
##                pch = pch,
##                col = col,
##                cex = cex
##         )
##     }
## }


## Y <- X[X$ver=="comb" & X$pref=="ggsim/ggsim" & X$p==1e5 & X$n >= 1000 & X$m==200,]
## items <- c("runtimes")
## idxs <- matrix(c(10:13), nrow=1)
## ps <- c(100000)
## for(i in 1:length(ps)){
##     for(j in 1:nrow(idxs)){
##         idx <- idxs[j,]
##         p = ps[i]
##         main = paste0("p=", toString(p), ", m=200")
##         pch = 1:4
##         col = 1:4
##         cex = 1.5
##         ylab = items[j]
##         matplot(Y$n[Y$p==p], Y[Y$p==p, idx], type="b", pch=pch, col=col, xlab="n", ylab=ylab, cex=cex)
##         legend("topleft",
##                title = main,
##                ## legend = names(Y[,idx]),
##                legend = methods,
##                pch = pch,
##                col = col,
##                cex = cex
##         )
##     }
## }


## pdf("accuracy.pdf", width = 16, height = 9)
## Y <- (X[X$ver=="comb" & X$pref=="ggsim/ggsim" & X$m==200,])
## items <- c("err.refcenter", "err.trace")
## idxs <- c(15:18)
## idxs <- rbind(idxs, c(20:23))
## ps <- 10^(c(0:2) + 3)
## par(mfcol = c(nrow(idxs), length(ps)))
## for(i in 1:length(ps)){
##     for(j in 1:nrow(idxs)){
##         idx <- idxs[j,]
##         p = ps[i]
##         main = paste0("p=", toString(p), ", m=200")
##         pch = 1:4
##         col = 1:4
##         cex = 1.5
##         ylab = items[j]
##         matplot(Y$n[Y$p==p], Y[Y$p==p, idx], type="b", pch=pch, col=col, xlab="n", ylab=ylab, cex=cex)
##         legend("topleft",
##                title = main,
##                ## legend = names(Y[,idx]),
##                legend = methods,
##                pch = pch,
##                col = col,
##                cex = cex
##         )
##     }
## }
## dev.off()

## pdf(pdf.file, width = 16, height = 9)
## Y <- (X[X$ver=="comb" & X$pref=="ggsim/ggsim" & X$m==200,])
## items <- c("runtimes", "err.refcenter", "err.trace")
## idxs <- c(10:13)
## idxs <- rbind(idxs, c(15:18))
## idxs <- rbind(idxs, c(20:23))
## ps <- 10^(c(0:2) + 3)
## par(mfrow = c(length(ps), nrow(idxs)))
## for(i in 1:length(ps)){
##     for(j in 1:nrow(idxs)){
##         idx <- idxs[j,]
##         p = ps[i]
##         main = paste0("p=", toString(p))
##         pch = 1:4
##         col = 1:4
##         cex = 1.5
##         ylab = items[j]
##         matplot(Y$n[Y$p==p], Y[Y$p==p, idx], type="b", pch=pch, col=col, xlab="n", ylab=ylab, cex=cex)
##         legend("topleft",
##                title = main,
##                ## legend = names(Y[,idx]),
##                legend = methods,
##                pch = pch,
##                col = col,
##                cex = cex
##         )
##     }
## }

## Y <- (X[X$ver=="notr" & X$pref=="ggsNotr/ggsNotr" & X$m==200 & X$n >= 1000 & X$n <= 5000,])
## ## items <- c("runtimes", "err.refcenter")
## items <- c("runtimes", "err.refcenter", "err.onl")
## idxs <- c(11:13) # Runtimes
## idxs <- rbind(idxs, c(16:18)) # err.refcenter
## idxs <- rbind(idxs, c(21:23)) # err.trace
## migs <- c(100,200,400,600,800)
## par(mfcol = c( nrow(idxs), length(migs)))
## for(i in 1:length(migs)){
##     for(j in 1:nrow(idxs)){
##         idx <- idxs[j,]
##         mig = migs[i]
##         main = paste0("mig=", toString(mig))
##         pch = 1:4
##         col = 1:4
##         cex = 1
##         ylab = items[j]
##         matplot(Y$n[Y$mig==mig], Y[Y$mig==mig, idx], type="b", pch=pch, col=col, xlab="n", ylab=ylab, cex=cex)
##         legend("topleft",
##                title = main,
##                ## legend = names(Y[,idx]),
##                legend = methods[2:4],
##                pch = pch,
##                col = col,
##                cex = cex
##         )
##     }
## }

## dev.off()
