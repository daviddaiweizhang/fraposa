# Run the hdpca package's function to predict the PC scores

require(hdpca)
require(MASS)

print("Running hdpca...")
args <- commandArgs(trailingOnly = TRUE)
if(identical(args, character(0))){
    print("Using testing args...")
	args <- c("rand", "hdpcaRand", "tmp", "100000", "1000", "200", "2", "1", "10")
}

ver <- args[1]
subver <- args[2]
pref <- args[3]
pref <- paste(pref, pref, sep = "/")
p <- args[4]
n <- args[5]
m <- args[6]
k <- args[7]
s <- args[8]
mig <- args[9]
name <- paste(pref, p, n, m, k, s, mig, sep = "_")
name <- paste0("../data/", name, ".", ver)


d.file <- paste(name, subver, "d", sep = ".")
d <- c(as.matrix(read.table(d.file)))
if(length(d) < as.integer(n)-1){
  nn <- length(d)
  hn <- nn %/% 2
  reg.x <- c(hn : (nn-1))
  reg.y <- d[reg.x]
  fill.x <- c(nn:as.integer(n))
  fill.y <- exp(predict(lm(log(reg.y) ~ reg.x), data.frame(reg.x = fill.x)))
  d <- c(d[1:(nn-1)], fill.y)
}


vproj.file <- paste(name, subver, "vproj", sep = ".")
vproj.df <- read.table(vproj.file, header = TRUE)
vproj <- as.matrix(vproj.df[,-(1:5)])
vproj.k <- ncol(vproj)
vproj <- vproj %*% diag(1/d[1:vproj.k])


timing <- (system.time({
	print("Running HDPCA (pc_adjust)...")
	vpred <- pc_adjust(d^2, as.numeric(p), as.numeric(n), vproj, method = "d.gsp", n.spikes.max = 20)
  vpred <- vpred %*% diag(d[1:vproj.k])
}))
timing.idx <- 3
hdpca.timing <- unname(timing[timing.idx])

runtimes.file <- paste0(name, ".runtimes")
runtimes <- read.table(runtimes.file, header=TRUE)
method.name <- paste("test", subver, sep = "_")
runtimes$elapse[runtimes$method==method.name] <- runtimes$elapse[runtimes$method==method.name] + hdpca.timing
write.table(runtimes, runtimes.file, sep = "\t", quote=FALSE, row.names=FALSE)

popID <- vproj.df$popID
indivID <- vproj.df$indivID
vpred.df <- data.frame(popID, indivID, L=NaN, K=NaN, t=NaN, vpred)
colnames(vpred.df) <- colnames(vproj.df)
pred.file <- paste(name, subver, "vpred", sep = ".")
write.matrix(vpred, pred.file, sep = "\t")
write.table(vpred.df, pred.file, sep="\t", quote=FALSE, row.names=FALSE)
print("Done!")

