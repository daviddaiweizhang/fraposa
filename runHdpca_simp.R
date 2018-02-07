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
direc <- args[2] 
name <- args[3]
p <- args[4]
n <- args[5]
name <- paste0("../data/", direc, "/", name)

vproj.file <- paste0(name, ".", ver, ".hdpca.vproj")
vproj.df <- read.table(vproj.file, header = TRUE)
vproj <- as.matrix(vproj.df[,-(1:5)])

d.file <- paste0(name, ".", ver, ".hdpca.d")
d <- c(as.matrix(read.table(d.file)))
timing <- (system.time({
	print("Running HDPCA (pc_adjust)...")
	vpred <- pc_adjust(d^2, as.numeric(p), as.numeric(n), vproj, method = "d.gsp", n.spikes.max = 20)
}))
timing.idx <- 3
hdpca.timing <- unname(timing[timing.idx])

runtimes.file <- paste0(name, ".", ver, ".runtimes")
runtimes <- read.table(runtimes.file, header=TRUE)
runtimes$elapse[runtimes$method=="test_hdpca"] <- runtimes$elapse[runtimes$method=="test_hdpca"] + hdpca.timing
write.table(runtimes, runtimes.file, sep = "\t", quote=FALSE, row.names=FALSE)

popID <- vproj.df$popID
indivID <- vproj.df$indivID
vpred.df <- data.frame(popID, indivID, L=NaN, K=NaN, t=NaN, vpred)
pred.file <- paste0(name, ".", ver, ".hdpca.vpred")
write.matrix(vpred, pred.file, sep = "\t")
write.table(vpred.df, pred.file, sep="\t", quote=FALSE, row.names=FALSE)
print("Done!")

