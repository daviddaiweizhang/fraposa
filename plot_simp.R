rm(list=ls())

require(graphics)

## Get the run date for TRACE
## Output in number of seconds after 1970 UTC (POSIXct)
get.tracedate <- function(trace.log){
    trace.log.lineskip <- 5
    trace.log.charskip <- 17
    tzone <- "UTC"
    tformat <- "%b %d %H:%M:%S %Y"
    con <- file(trace.log, open="r")
    readLines(con, n=trace.log.lineskip)
    ourLine <- readLines(con, n=1)
    print(ourLine)
    close(con)
    rundate <- substring(ourLine, first=trace.log.charskip)
    rundate <- as.POSIXct(rundate, format=tformat) 
    attr(rundate, "tzone") <- tzone
    rundate <- as.numeric(rundate)
    return(rundate)
}

# read testing pcs from trace's output file
load.trace <- function(fin) {
    if(!file.exists(fin)){
         return(NaN)
    }
    result <- read.table(fin, header = TRUE)
    return(result)
}
# one way to define the norm of a matrix
mat.norm <- function(x){
    m <- nrow(x)
    n <- ncol(x)
    # depends on which verison you want
    norm <- sqrt(sum(x^2) / (m * n))
    # norm <- sqrt(sum(x^2))
    return(norm)
}
    
# the difference between two matrices
mat.err <- function(x, y){
    mat.norm(x - y)
}

# Create a named list of the properties of the methods
# Each element is a list of the same length
pcprops <- list()

# Input the display names
display.names <- c(
    "ref_trace",
    "test_trace",
    "test_onl",
    "test_proj",
    "test_hdpca"
)
pcprops$display.names <- display.names
meths.n <- length(display.names)

# Read arguements from user
args <- commandArgs(trailingOnly = TRUE)
if(identical(args, character(0))){
    print("Using testing args...")
    ## args <- c("rand", "ggsim", "ggsim_1000_200_2_1_100_0_ggsim_1000_100_2_1_100_1", "1000 ", "200", "100")
    args <- c("comb", "ukb", "kgn_allChr_ukb_orphans_ukb_1k", "145282", "2492", "1000")
    print(args)
}
ver <- args[1]
direc <- args[2]
name <- args[3]
p <- args[4]
n <- args[5]
m <- args[6]
pref <- direc
gridsize <- 2
unbalance <- 1
migration <- 100
# pref <- args[2]
# pref <- paste(pref, pref, sep = "/")
# p <- args[3]
# n <- args[4]
# m <- args[5]
# gridsize <- args[6]
# unbalance <- args[7]
# migration <- args[8]
# name <- paste(pref, p, n, m, gridsize, unbalance, migration , sep = "_")
# name <- paste0("../data/", name)
name <- paste("../data", direc, name, sep = "/")
print(args)

# Input the runtimes for the methods
runtimes.file <- paste0(name, ".", ver, ".runtimes")
runtimes <- read.table(runtimes.file, header=TRUE, stringsAsFactors = FALSE)
pcprops$runtimes <- runtimes$elapse[match(pcprops$display.names, runtimes$method)]

# Input file names
file.names <- c(
    paste(name, ver, "RefPC.coord", sep = "."),
    paste(name, ver, "ori.ProPC.coord", sep = "."),
    paste(name, ver, "onl.ProPC.coord", sep = "."),
    ## paste(name, ver, "rand.ProPC.coord", sep = "."),
    paste(name, ver, "hdpca.vproj", sep = "."),
    paste(name, ver, "hdpca.vpred", sep = ".")
)
pcprops$file.names <- file.names


# Input pc scores
## pcprops$pcs <- vector(mode = "list", length = meths.n)
for(i in 1:meths.n){
    pcprops$datafr[[i]] <- load.trace(pcprops$file.names[i])
    }


# Trace specific parameters
# The index for reference group
idx.ref <- 1
# The index for trace study pcs (the golden standard)
idx.trace.ori <- 2
# The first few columns are not pc scores
nonpcs.cols.n.ref <- 2
nonpcs.cols.n <- 5

# Number of PC's inputed
k <- ncol(pcprops$datafr[[idx.ref]]) - nonpcs.cols.n.ref
# Number of PC's used for calculating errors
l <- 2

# Creat placeholders for missing files
placeholder.df <- pcprops$datafr[[idx.trace.ori]]
placeholder.df[,-(1:nonpcs.cols.n)] <- data.frame(matrix(0, strtoi(m), k))
for (i in 1:meths.n){
    if(!is.data.frame(pcprops$datafr[[i]])){ #
        pcprops$datafr[[i]] <- placeholder.df
    }
}

# Extract the PC scores as a matrix from the data frame
pcprops$pcs[[idx.ref]] <- as.matrix(pcprops$datafr[[1]][,-(1:nonpcs.cols.n.ref)])
for(i in (idx.ref+1):meths.n){
    pcprops$pcs[[i]] <- as.matrix(pcprops$datafr[[i]][,-(1:nonpcs.cols.n)])
}

# # Get the list of populations
# popu <- unique(pcprops$datafr[[idx.ref]]$popID)
# popu.n <- length(popu)
# 
# print("pcs")
# 
# # Calculate the geographical centers based on the reference group PC scores
# centers <- c()
# for(i in 1:popu.n){
# print(i)
#     popu.idxs <- pcprops$datafr[[idx.ref]]$popID == popu[i]
# print(str(pcprops$pcs[[idx.ref]]))
#     ctr <- colMeans(pcprops$pcs[[idx.ref]][popu.idxs, 1:l])
#     centers <- rbind(centers, ctr)
# }
# 

# # Add refcenters and centered pc scores to dataframes
# for(i in 1:meths.n){
#     popID <- pcprops$datafr[[i]]$popID
#     for(j in 1:l){
#         col.name.pcs <- paste0("PC", toString(j))
#         col.name.refcenter <- paste0("PC", toString(j), ".refcenter")
#         col.name.centered <- paste0("PC", toString(j), ".centered")
#         for(a in 1:popu.n){
#             pcprops$datafr[[i]][[col.name.refcenter]][popID == popu[a]] <- centers[a, j]
#         }
#         refcenter <- pcprops$datafr[[i]][[col.name.refcenter]]
#         pcprops$datafr[[i]][[col.name.centered]] <- pcprops$datafr[[i]][[col.name.pcs]] - pcprops$datafr[[i]][[col.name.refcenter]]
#     }
# }

# Input errors of testing pcs from reference pcs
# Error from trace
err.trace <- c(-1) # The first entry is a placeholder for the reference group
mat.trace.ori <- pcprops$pcs[[idx.trace.ori]][,1:l]
for(i in (idx.ref+1):meths.n){
    mat.i <- pcprops$pcs[[i]][,1:l]
    err.trace.i <- mat.err(mat.trace.ori, mat.i)
    err.trace <- c(err.trace, err.trace.i)
}
err.trace <- signif(err.trace, 3)
pcprops$err.trace <- err.trace

# Create the legend for plotting
# Error from trace
plot.err.trace <- paste("err.trace=", pcprops$err.trace, sep = "")
pcprops$plot.err.trace <- plot.err.trace


# # Do the same for error from ref centeres
# # Error from reference centeres
# err.refcenter <- c()
# for(i in 1:meths.n){
#         col.names <- c()
#         for (j in 1:l){
#             colnm <- paste0("PC", toString(j), ".centered")
#             col.names <- c(col.names, colnm)
#         }
#         centered.pcs <- as.matrix(pcprops$datafr[[i]][, col.names])
#         err.refcenter <- c(err.refcenter, mat.norm(centered.pcs))
#     err.refcenter <- signif(err.refcenter, 3)
#     pcprops$err.refcenter <- err.refcenter
#     # Error from ref centeres
#     plot.err.refcenter <- paste("err.refcenter=", pcprops$err.refcenter, sep = "")
#     pcprops$plot.err.refcenter <- plot.err.refcenter
# }


# Input the char types for plotting
pcprops$plot.pch <- c(1 : meths.n)

# Input the colors for plotting
# Getting the superpopulation of the reference individuals
print("Getting superpopu...")
superpop.file <- c("../data/kgn/ALL.panel")
superpop.table <- read.table(superpop.file, header = TRUE)
superpop <- superpop.table$super_pop
refIndv.bool <- match(pcprops$datafr[[1]]$indivID, superpop.table$sample)
superpop <- superpop[refIndv.bool]
superpop.uniq <- unique(superpop)
superpop.n <- length(superpop.uniq)
cols.n <- superpop.n + meths.n - 1
refIndvs.col <- rainbow(superpop.n)[as.numeric(superpop)]
# pcprops$plot.col <- rainbow(cols.n)[superpop.n : cols.n]
pcprops$plot.col <- rep(1, meths.n)


# Plot the pc scores
plot.pc.n <- 2 # Number of PC's plotted
plot.title <- paste0("data=", name, "\n", "ver=", ver, ", #SNPs=", p, ", nTrain=", n, ", nTest=", m)
pdf.file <- paste0(name, '.', ver, '.pdf')
pdf(pdf.file, width = 8, height = 8)
par(mfrow = c(1, 1))
kk = plot.pc.n / 2 - 1 
for (i in 0 : kk) {
    j = i*2 + 1
    ii = 1
    plot(
        pcprops$pcs[[ii]][,j], pcprops$pcs[[ii]][,j+1],
        pch = pcprops$plot.pch[ii],
        col = refIndvs.col,
        main = plot.title,
        xlab = paste0("PC", j),
        ylab = paste0("PC", (j+1))
    )
    for(ii in 2:meths.n){
        points(
            pcprops$pcs[[ii]][,j], pcprops$pcs[[ii]][,j+1],
            pch = pcprops$plot.pch[ii],
            col = pcprops$plot.col[ii]
        )
    }
    legend("topleft",
           title = "method",
           legend = pcprops$display.names,
           pch = pcprops$plot.pch,
           col = pcprops$plot.col
    )
#     legend("bottomleft",
#         title = "err.refcenter",
#         legend = pcprops$err.refcenter,
#         pch = pcprops$plot.pch,
#         col = pcprops$plot.col
#     )
    legend("top",
        title = "err.trace",
        legend = pcprops$err.trace,
        pch = pcprops$plot.pch,
        col = pcprops$plot.col
    )
    legend("topright",
        title = "runtimes",
        legend = pcprops$runtimes,
        pch = pcprops$plot.pch,
        col = pcprops$plot.col
    )
	legend("bottomleft",
title = "superpop",
legend = superpop.uniq,
pch = 1,
col = rainbow(superpop.n)[as.numeric(superpop.uniq)]
) 

#     text(centers[,1], centers[,2], popu)
#     points(centers[,1], centers[,2], pch=20, cex = 2)
}
dev.off()


# Get run datetime from TRACE log file
trace.log <- paste(name, ver, "log", sep=".")
rundate <- get.tracedate(trace.log)


# Get node name for TRACE
nodename.file <- paste(name, ver, "nodename", sep=".")
con <- file(nodename.file, open="r")
nodename <- readLines(con, n=1)
close(con)


# Save result to a database
# Save result as a database entry
pcsummary.entry.par <- list(
    ver,
    pref,
    as.numeric(p),
    as.numeric(n),
    as.numeric(m),
    as.numeric(gridsize),
    as.numeric(unbalance),
    as.numeric(migration),
    nodename,
    rundate
)
pcsummary.entry <- c(
    pcsummary.entry.par,
    as.list(pcprops$runtimes),
    as.list(pcprops$err.refcenter),
    as.list(pcprops$err.trace)
)

# The colnames for identifying results
pcsummary.header.par <- c("ver", "pref", "p", "n", "m", "gridsize", "unbalance", "mig", "nodename", "timestamp")
# Create header
pcsummary.header <- pcsummary.header.par
pcsummary.header <- c(pcsummary.header, paste(pcprops$display.names, "runtimes", sep = "."))
# pcsummary.header <- c(pcsummary.header, paste("err.refcenter", pcprops$display.names, sep = "."))
pcsummary.header <- c(pcsummary.header, paste("err.trace", pcprops$display.names, sep = "."))
# print(pcsummary.entry)
pcsummary.entry <- data.frame(pcsummary.entry)
colnames(pcsummary.entry) <- pcsummary.header

# Modify database
pcsummary.file <- "pcSummary_simp"
writeColNames = FALSE
if(!file.exists(pcsummary.file)){
    writeColNames = TRUE
}
write.table(
        pcsummary.entry, pcsummary.file, row.names=FALSE, quote = FALSE, sep = "\t", 
        append=TRUE, col.names = writeColNames
        )
print("Dataframe file is written.")


