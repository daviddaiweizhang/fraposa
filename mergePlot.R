library(RColorBrewer)
library(scales)


meth.n <- 7
chg.var <- "n"
chg.jobid <- "27483940"
chg.range <- seq(1000, 3000, 500)
chg.taskid <- c(0:4)
chg.length <- length(chg.range)

file.head <- "../flux/data/"
file.neck <- paste0(
  chg.var,
  "Chg_rand_",
  chg.jobid,
  "_",
  chg.taskid
)
file.upper <- paste0(
  file.head,
  file.neck,
  "/",
  file.neck,
  "_100000_")
if(chg.var == "n"){
  file.ab <- paste(chg.range, "200", sep = "_")
  xlab <- "Reference Size"
} else if(chg.var == "m"){
  file.ab <- paste("600", chg.range, sep = "_")
  xlab <- "Study Size"
} else{
  stop("Invalid changing variable. Must be m or n. ")
}
file.leg <- "_2_1_100.rand"
out.pref <- paste0(file.upper,
                  file.ab,
                  file.leg)

runtimes.ref <- matrix(0, chg.length, meth.n + 1)
runtimes.ref[,1] <- chg.range
runtimes.stu <- matrix(0, chg.length, meth.n + 1)
runtimes.stu[,1] <- chg.range
accuracy.ctr <- matrix(0, chg.length, meth.n + 1)
accuracy.ctr[,1] <- chg.range
accuracy.gold <- matrix(0, chg.length, meth.n + 1)
accuracy.gold[,1] <- chg.range
for(i in 1:chg.length){
  runtimes.file <- paste(out.pref[i], "runtimes", sep = ".")
  accuracy.file <- paste(out.pref[i], "accuracy", sep = ".")
  runtimes.df <- read.table(runtimes.file, header = TRUE)
  accuracy.df <- read.table(accuracy.file, header = TRUE)
  meth.name <- runtimes.df$method
  runtimes.ref[i,(1:meth.n)+1] <- runtimes.df$reference
  runtimes.stu[i,(1:meth.n)+1] <- runtimes.df$study
  accuracy.ctr[i,(1:meth.n)+1] <- accuracy.df$err.center
  accuracy.gold[i,(1:meth.n)+1] <- accuracy.df$err.golden
}
if(chg.var == "n"){
  png("doc/nChg.png", width = 4000, height = 4000, res = 300)
  par(oma = c(4, 4, 4, 2), mar = c(5, 4, 4, 2), mfrow = c(2,2))
  lty = c(1, 1, 2, 1, 2, 1, 2)
  pch = c(15, 16, 1, 17, 2, 18, 5)
  col = alpha(brewer.pal(meth.n, "Paired"), 1)
  col = col[c(meth.n, 1:(meth.n-1))]
  cex = 2
  lwd = 2
  col.main = "gray25"
  matplot(runtimes.ref[,1], runtimes.ref[,(1:meth.n)+1],
          main = "Runtimes (Reference)",
          xlab = xlab,
          ylab = "Seconds",
          lty = lty, pch = pch, col = col, cex = cex, type = "b", lwd = lwd, col.main = col.main)
  matplot(runtimes.stu[,1], runtimes.stu[,(1:meth.n)+1],
          main = "Runtimes (Study)",
          xlab = xlab,
          ylab = "Seconds",
          lty = lty, pch = pch, col = col, cex = cex, type = "b", lwd = lwd, col.main = col.main)
  matplot(accuracy.ctr[,1], accuracy.ctr[,(1:meth.n)+1],
          main = "Deviations (Distance from Centers)",
          xlab = xlab,
          ylab = "PC Score Units",
          lty = lty, pch = pch, col = col, cex = cex, type = "b", lwd = lwd, col.main = col.main)
  matplot(accuracy.gold[,1], accuracy.gold[,(1:meth.n)+1],
          main = "Deviations (Distance from TRACE)",
          xlab = xlab,
          ylab = "PC Score Units",
          lty = lty, pch = pch, col = col, cex = cex, type = "b", lwd = lwd, col.main = col.main)
  par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
  plot(0, 0, type = "n", bty = "n", xaxt = "n", yaxt = "n")
  legend("bottom", legend = meth.name, xpd = TRUE, horiz = TRUE, inset = c(0, 0), bty = "n", pch = pch, col = col, cex = cex*0.8, lwd = lwd, lty = lty)
  title(paste("Runtimes and Deviations by Method and", xlab), outer=TRUE, line = -3, cex.main = 2)
  dev.off()
} else if (chg.var == "m"){
  png("doc/mChg.png", width = 4000, height = 4000, res = 300)
  lty = c(1, 1, 2, 1, 2, 1, 2)
  pch = c(15, 16, 1, 17, 2, 18, 5)
  col = alpha(brewer.pal(meth.n, "Paired"), 1)
  col = col[c(meth.n, 1:(meth.n-1))]
  cex = 2
  lwd = 2
  col.main = "gray25"
  matplot(runtimes.stu[,1], runtimes.stu[,(1:meth.n)+1],
          main = "Runtimes for the Study Individuals",
          xlab = xlab,
          ylab = "Seconds",
          lty = lty, pch = pch, col = col, cex = cex, type = "b", lwd = lwd)
  legend("topleft", legend = meth.name, xpd = FALSE, bty = "o", pch = pch, col = col, cex = cex*0.8, lwd = lwd, lty = lty)
  dev.off()
}
