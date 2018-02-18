rm(list=ls())
out.pref <- "../data/tmp/tmp_1000_200_100_2_1_100.rand"
runtime.file <- paste(out.pref, "runtimes", sep = ".")
runtime.df <- read.table(runtime.file, header = TRUE)
n <- as.integer(nrow(runtime.df) / 2)
meth.names <- runtime.df$method[1:n]
meth.names <- substring(meth.names, 5)
ref.time <- runtime.df$elapse[1:n]
test.time <- runtime.df$elapse[(n+1):(2*n)]
runtime.multi.df <- data.frame(
  method = meth.names,
  reference = ref.time,
  study = test.time
)
write.table(runtime.multi.df, runtime.file, quote = FALSE, sep = "\t", row.names = FALSE)


