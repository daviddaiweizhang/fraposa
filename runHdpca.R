# Run the hdpca package's function to predict the PC scores

require(hdpca)
require(MASS)

print("Running hdpca...")
args <- commandArgs(trailingOnly = TRUE)
if(identical(args, character(0))){
    print("Using testing args...")
	args <- c("comb", "ggsim", "1000", "100", "100", "2", "1", "10")
}

ver <- args[1]
pref <- args[2]
pref <- paste(pref, pref, sep = "/")
p <- args[3]
n <- args[4]
m <- args[5]
k <- args[6]
s <- args[7]
mig <- args[8]
name <- paste(pref, p, n, m, k, s, mig, sep = "_")
name <- paste0("../data/", name, ".", ver)

vproj.file <- paste0(name, ".hdpca.vproj")
vproj.df <- read.table(vproj.file, header = TRUE)
vproj <- as.matrix(vproj.df[,-(1:5)])

d.file <- paste0(name, ".hdpca.d")
d <- c(as.matrix(read.table(d.file)))

timing <- (system.time({
	print("Running HDPCA (pc_adjust)...")
	vpred <- pc_adjust(d^2, as.numeric(p), as.numeric(n), vproj, method = "d.gsp", n.spikes.max = 20)
}))
timing.idx <- 3
hdpca.timing <- unname(timing[timing.idx])

runtimes.file <- paste0(name, ".runtimes")
runtimes <- read.table(runtimes.file, header=TRUE)
runtimes$elapse[runtimes$method=="test_hdpca"] <- runtimes$elapse[runtimes$method=="test_hdpca"] + hdpca.timing
write.table(runtimes, runtimes.file, sep = "\t", quote=FALSE, row.names=FALSE)

popID <- vproj.df$popID
indivID <- vproj.df$indivID
vpred.df <- data.frame(popID, indivID, L=NaN, K=NaN, t=NaN, vpred)
pred.file <- paste0(name, ".hdpca.vpred")
write.matrix(vpred, pred.file, sep = "\t")
write.table(vpred.df, pred.file, sep="\t", quote=FALSE, row.names=FALSE)
print("Done!")

