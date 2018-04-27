library(hdpca)
X = read.table('hgdp_ref.012')
p.ref = nrow(X)
n.ref = ncol(X)
X.mean = rowMeans(X)
X.std = apply(X, 1, sd)
X.std[X.std==0] = 1
X = X - X.mean
X = X / X.std
X.svd = svd(X)
U = X.svd$u
d = X.svd$d
V = X.svd$v
print(Sys.time())
print("Calculating number of spikes...")
print("n.spikes.max=19")
n.spikes = select.nspike(d^2, p.ref, n.ref, n.spikes.max=19)
print(Sys.time())
print("n.spikes.max=20")
n.spikes = select.nspike(d^2, p.ref, n.ref, n.spikes.max=20)
print(Sys.time())
print("Done.")

## W = read.table('hgdp_stu.012')
## W = W - X.mean
## W = W / X.std
## pcs.ref = V %*% diag(d)
## pcs.proj = t(W) %*% U
## png('pcs.png')
## plot(pcs.ref[,1], pcs.ref[,2], col = 'grey')
## points(pcs.proj[,1], pcs.proj[,2], col = 'red')
## dev.off()
