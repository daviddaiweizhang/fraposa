# Run the hdpca package's function to predict the PC scores

require(hdpca)
require(MASS)

print("Running HDPCA...")
args <- commandArgs(trailingOnly = TRUE)
if(identical(args, character(0))){
    print("Using testing args...")
	args <- c("rand", "hdpca", "tmp", "tmp_1000_200_100_2_1_100")
}

out.pref <- args[1]
refver <- args[2]

info.file <- paste(out.pref, "info", sep = ".")
info.df <- read.table(info.file, sep = "\t",
                      colClasses = "character")
n <- as.integer(info.df[info.df[,1]=="n", 2])
p <- as.integer(info.df[info.df[,1]=="p", 2])

d.file <- paste(out.pref, refver, "d", sep = ".")
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

vproj.file <- paste(out.pref, refver, "vproj", sep = ".")
vproj.df <- read.table(vproj.file, header = TRUE)
vproj <- as.matrix(vproj.df[,-(1:5)])
vproj.k <- ncol(vproj)
vproj <- vproj %*% diag(1/d[1:vproj.k])

timing <- (system.time({
	print("Running pc_adjust...")
	vpred <- pc_adjust(d^2, p, n, vproj, method = "d.gsp", n.spikes.max = 20)
  vpred <- vpred %*% diag(d[1:vproj.k])
}))
timing.idx <- 3
hdpca.timing <- unname(timing[timing.idx])

runtimes.file <- paste0(out.pref, ".runtimes")
runtimes <- read.table(runtimes.file, header=TRUE)
method.name <- refver
runtimes$study[runtimes$method==method.name] <- runtimes$study[runtimes$method==method.name] + hdpca.timing
write.table(runtimes, runtimes.file, sep = "\t", quote=FALSE, row.names=FALSE)

popID <- vproj.df$popID
indivID <- vproj.df$indivID
vpred.df <- data.frame(popID, indivID, L=NaN, K=NaN, t=NaN, vpred)
colnames(vpred.df) <- colnames(vproj.df)
pred.file <- paste(out.pref, refver, "vpred", sep = ".")
write.matrix(vpred, pred.file, sep = "\t")
write.table(vpred.df, pred.file, sep="\t", quote=FALSE, row.names=FALSE)
print("Done!")
