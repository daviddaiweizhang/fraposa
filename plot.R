rm(list=ls())

## Get the run date for TRACE
## Output in number of seconds after 1970 UTC (POSIXct)
get.tracedate <- function(trace.log){
    trace.log.lineskip <- 5
    trace.log.charskip <- 17
    # tzone <- "UTC"
    tformat <- "%b %d %H:%M:%S %Y"
    con <- file(trace.log, open="r")
    readLines(con, n=trace.log.lineskip)
    ourLine <- readLines(con, n=1)
    # print(ourLine)
    close(con)
    rundate <- substring(ourLine, first=trace.log.charskip)
    rundate <- as.POSIXct(rundate, format=tformat) 
    # attr(rundate, "tzone") <- tzone
    # rundate <- as.numeric(rundate)
    rundate <- strftime(rundate, format = "%Y-%m-%d_%H:%M:%S")
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
    "ref_rand",
    "test_trace",
  "test_onl",
  "test_onlRand",
    "test_proj",
    "test_projRand",
    "test_hdpca",
    "test_hdpcaRand"
)
pcprops$display.names <- display.names
meths.n <- length(display.names)

# Read arguements from user
args <- commandArgs(trailingOnly = TRUE)
if(identical(args, character(0))){
    print("Using testing args...")
    args <- c("rand", "tmp", "100000", "1000", "200", '2', '1', '10')
    print(args)
}
ver <- args[1]
pref <- args[2]
pref <- paste(pref, pref, sep = "/")
p <- args[3]
n <- args[4]
m <- args[5]
gridsize <- args[6]
unbalance <- args[7]
migration <- args[8]
name <- paste(pref, p, n, m, gridsize, unbalance, migration
            , sep = "_")
name <- paste0("../data/", name)

# Input the runtimes for the methods
runtimes.file <- paste0(name, ".", ver, ".runtimes")
runtimes <- read.table(runtimes.file, header=TRUE, stringsAsFactors = FALSE)
pcprops$runtimes <- runtimes$elapse[match(pcprops$display.names, runtimes$method)]

# Input file names
file.names <- c(
    paste(name, ver, "RefPC.coord", sep = "."),
    paste(name, ver, "RefPC.rand.coord", sep = "."),
    paste(name, ver, "ori.ProPC.coord", sep = "."),
    paste(name, ver, "onl.ProPC.coord", sep = "."),
    paste(name, ver, "onlRand.ProPC.coord", sep = "."),
    ## paste(name, ver, "rand.ProPC.coord", sep = "."),
    paste(name, ver, "hdpca.vproj", sep = "."),
    paste(name, ver, "hdpcaRand.vproj", sep = "."),
    paste(name, ver, "hdpca.vpred", sep = "."),
    paste(name, ver, "hdpcaRand.vpred", sep = ".")
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
idx.trace.ori <- 3
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
meth.ref.n <- 2
for(i in 1:meth.ref.n){
  pcprops$pcs[[idx.ref+i-1]] <- as.matrix(pcprops$datafr[[i]][,-(1:nonpcs.cols.n.ref)])
}
for(i in (idx.ref+meth.ref.n):meths.n){
    pcprops$pcs[[i]] <- as.matrix(pcprops$datafr[[i]][,-(1:nonpcs.cols.n)])
}

# Get the list of populations
popu <- unique(pcprops$datafr[[idx.ref]]$popID)
popu.n <- length(popu)

# Calculate the geographical centers based on the reference group PC scores
centers <- c()
for(i in 1:popu.n){
    popu.idxs <- pcprops$datafr[[idx.ref]]$popID == popu[i]
    ctr <- colMeans(pcprops$pcs[[idx.ref]][popu.idxs, 1:l])
    centers <- rbind(centers, ctr)
}


# Add refcenters and centered pc scores to dataframes
for(i in 1:meths.n){
    popID <- pcprops$datafr[[i]]$popID
    for(j in 1:l){
        col.name.pcs <- paste0("PC", toString(j))
        col.name.refcenter <- paste0("PC", toString(j), ".refcenter")
        col.name.centered <- paste0("PC", toString(j), ".centered")
        for(a in 1:popu.n){
            pcprops$datafr[[i]][[col.name.refcenter]][popID == popu[a]] <- centers[a, j]
        }
        refcenter <- pcprops$datafr[[i]][[col.name.refcenter]]
        pcprops$datafr[[i]][[col.name.centered]] <- pcprops$datafr[[i]][[col.name.pcs]] - pcprops$datafr[[i]][[col.name.refcenter]]
    }
}

# Input errors of testing pcs from reference pcs
# Error from trace
err.trace <- rep(-1, meth.ref.n) # The first entry is a placeholder for the reference group
mat.trace.ori <- pcprops$pcs[[idx.trace.ori]][,1:l]
for(i in (idx.ref+meth.ref.n):meths.n){
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


# Do the same for error from ref centeres
# Error from reference centeres
err.refcenter <- c()
for(i in 1:meths.n){
        col.names <- c()
        for (j in 1:l){
            colnm <- paste0("PC", toString(j), ".centered")
            col.names <- c(col.names, colnm)
        }
        centered.pcs <- as.matrix(pcprops$datafr[[i]][, col.names])
        err.refcenter <- c(err.refcenter, mat.norm(centered.pcs))
    err.refcenter <- signif(err.refcenter, 3)
    pcprops$err.refcenter <- err.refcenter
    # Error from ref centeres
    plot.err.refcenter <- paste("err.refcenter=", pcprops$err.refcenter, sep = "")
    pcprops$plot.err.refcenter <- plot.err.refcenter
}


# Input the char types for plotting
pcprops$plot.pch <- c(1 : meths.n)

# Input the colors for plotting
pcprops$plot.col <- c(1 : meths.n)

# Plot the pc scores
plot.pc.n <- 2 # Number of PC's plotted
plot.title <- paste0("data=", pref, ", #SNPs=", p, ", nTrain=", n, ", nTest=", m, ",\ngridSize=", gridsize, ", unbalance=", unbalance, ", migration=", migration, ", ver=", ver)
pdf.file <- paste(name, ver, 'pdf', sep=".")
pdf(pdf.file, width = 8, height = 8)
par(mfrow = c(1, 1))
kk = plot.pc.n / 2 - 1 
for (i in 0 : kk) {
    j = i*2 + 1
    ii = 1
    plot(
        pcprops$pcs[[ii]][,j], pcprops$pcs[[ii]][,j+1],
        pch = pcprops$plot.pch[ii],
        col = pcprops$plot.col[ii],
        main = plot.title,
        xlab = paste0("PC", j),
        ylab = paste0("PC", (j+1))
    )
    for(ii in 2:(meths.n)){
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
    legend("bottomleft",
        title = "err.refcenter",
        legend = pcprops$err.refcenter,
        pch = pcprops$plot.pch,
        col = pcprops$plot.col
    )
    legend("bottomright",
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
    text(centers[,1], centers[,2], popu)
    points(centers[,1], centers[,2], pch=20, cex = 2)
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
pcsummary.header <- c(pcsummary.header, paste("err.refcenter", pcprops$display.names, sep = "."))
pcsummary.header <- c(pcsummary.header, paste("err.trace", pcprops$display.names, sep = "."))

# Modify database
pcsummary.file <- "pcSummary"
if(!file.exists(pcsummary.file)){
    print("Creating dataframe file...")
    # Create dataframe
    pcsummary.placeholder <- matrix(0, 1, length(pcsummary.header))
    pcsummary <- data.frame(pcsummary.placeholder, stringsAsFactors = FALSE)
    colnames(pcsummary) <- pcsummary.header
    # Add entry to dataframe
    pcsummary[1,] <- pcsummary.entry
    print("Dataframe is created")
}else{
    print("Reading dataframe file...")
    # Read existing dataframe from disk
    pcsummary <- read.table(pcsummary.file, header = TRUE, stringsAsFactors = FALSE, sep = "\t")
    # Sanity check for the existing database
    stopifnot(colnames(pcsummary) == pcsummary.header)

    ## ## Allow duplicate entries so that we can take averages
    ## ## Actually, since we have timestamp and nodename, we never have duplicate entries
    ## pcsummary <- rbind(pcsummary, pcsummary.entry)
    ## print("New entry is created.")

    ## If we want to avoid duplicate entries, then use below
    ## Find if entry exists
    ## ## Use only the parameters and machine info as identifier
    ## pcsummary.entry.match <- rowSums(pcsummary[,pcsummary.header.par] == pcsummary.entry.par) == length(pcsummary.entry.par)
    ## Use all entries as identifier (including the pc score output)
    pcsummary.entry.match <- rowSums(pcsummary == pcsummary.entry) == length(pcsummary.entry)
    pcsummary.entry.matchIdx <- which(unname(pcsummary.entry.match))
    numMatch <- length(pcsummary.entry.matchIdx)
#    if(numMatch == 0){
        pcsummary <- rbind(pcsummary, pcsummary.entry)
        print("New entry is created.")
#    }else if(numMatch == 1){
#        pcsummary[pcsummary.entry.matchIdx, ] = pcsummary.entry
#        print("Existing entry is replaced.")
#    }else{
#        stop("Duplicate entries exist.")
#    }
}
write.table(pcsummary, pcsummary.file, row.names=FALSE, quote = FALSE, sep = "\t")
print("Dataframe file is written.")


