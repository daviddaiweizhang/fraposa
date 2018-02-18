rm(list=ls())

# List of methods
meth.name <- c("trace",
               "onl",
               "onlRand",
               "proj",
               "projRand",
               "hdpca",
               "hdpcaRand"
               )

# List of pc score file suffixes
suff.name <-  c(
  "ori.ProPC.coord",
  "onl.ProPC.coord",
  "onlRand.ProPC.coord",
  "hdpca.vproj",
  "hdpcaRand.vproj",
  "hdpca.vpred",
  "hdpcaRand.vpred"
)

print("Calculating accuracy...")
args <- commandArgs(trailingOnly = TRUE)
if(identical(args, character(0))){
    print("Using testing args.")
    args <- "../data/tmp/tmp_100000_1000_100_2_1_10.rand"
}

# Info about data file format
meth.n <- length(meth.name)
info.idx <- c(1:2)
pc.n <- 2
ref.pc.idx <- 2 + (1:pc.n)
test.pc.idx <- 5 + (1:pc.n)

# Input and output names for basic info files
out.pref <- args[1]
info.file <- paste(out.pref, "info", sep = ".")
accu.file <- paste(out.pref, "accuracy", sep = ".")
pdf.file <- paste(out.pref, "pdf", sep = ".")

# Full file names
file.name <- c()
for(i in 1:meth.n){
  file.name[i] <- paste(out.pref, suff.name[i], sep = ".")
}

# Read trace ref and test into data frames
trace.ref.file <- paste(out.pref, "RefPC.coord", sep = ".")
trace.test.file <- paste(out.pref, "ori.ProPC.coord", sep = ".")
trace.ref <- read.table(trace.ref.file, header = TRUE)
trace.test <- read.table(trace.test.file, header = TRUE)
ref.mat <- as.matrix(trace.ref[,ref.pc.idx])

# Create reference matrix (ctr)
trace.ref.ctr <- aggregate(trace.ref[,ref.pc.idx], by = list(trace.ref$popID), FUN = mean)
colnames(trace.ref.ctr) <- c("popID", "PC1.ctr", "PC2.ctr")
trace.test.ctr <- merge(trace.test[,info.idx], trace.ref.ctr, by = "popID")
ref.ctr <- as.matrix(trace.test.ctr[,-info.idx])

# Create reference matrix (gold)
ref.gold <- as.matrix(trace.test[,test.pc.idx])

test.df <- list()
for(i in 1:meth.n){
  test.df[[i]] <- read.table(file.name[i], header = TRUE)
}

test.mat <- list()
for(i in 1:meth.n){
  test.mat[[i]] <- as.matrix(test.df[[i]][,test.pc.idx])
}

mat.diff <- function(x,y){
  stopifnot(all(dim(x) == dim(y)))
  z = y - x
  m <- nrow(z)
  n <- ncol(z)
  z.norm <- sqrt(sum(z^2) / (m * n))
  z.norm <- round(z.norm, 2)
  return(z.norm)
}

err.ctr <- c()
for(i in 1:meth.n){
  err.ctr[i] <- mat.diff(ref.ctr, test.mat[[i]])
}

err.gold <- c()
for(i in 1:meth.n){
  err.gold[i] <- mat.diff(ref.gold, test.mat[[i]])
}

accu.df <- data.frame(
  method = meth.name,
  err.center = err.ctr,
  err.golden = err.gold
)
write.table(accu.df, accu.file, quote = FALSE, sep = "\t", row.names = FALSE)
print("Done!")

## Plot pc scores
ref.popu <- as.integer(trace.ref$popID)
popu.n <- length(unique(ref.popu))
meth.col <- rainbow(meth.n)
pdf(
plot(ref.mat[,1], ref.mat[,2],
     pch = ref.popu+1, col = "gray",
     xlab = "PC1", ylab = "PC2",
     main = out.pref)
for(i in 1:meth.n){
  points(test.mat[[i]][,1], test.mat[[i]][,2], pch = 1, col = meth.col[i])
}
legend("topleft",
       title = "method",
       legend = meth.name,
       pch = 1,
       col = meth.col
)
legend("bottomleft",
       title = "err.center",
       legend = err.ctr,
       pch = 1,
       col = meth.col
)
legend("bottomright",
       title = "err.golden",
       legend = err.gold,
       pch = 1,
       col = meth.col
)


