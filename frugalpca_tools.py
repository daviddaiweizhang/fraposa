PLOT_ALPHA_REF=0.4
PLOT_ALPHA_STU=0.2
PLOT_MARKERS = ['.', '+', 'x', 'd', '*', 's']
LOG_LEVEL = 'info'

def plink_keep(bfile, keep, out):
    bashout = subprocess.run(
            ['plink', '--keep-allele-order', '--make-bed',
            '--indiv-sort', 'file', keep,
            '--bfile', bfile,
            '--keep', keep,
            '--out', out],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert bashout.stderr.decode('utf-8')[:5] is not 'Error'
    # keyword = bashout.stdout.decode('utf-8').split('\n')[-1][:5]
    # assert keyword is not 'Error'
    os.remove(out+'.log')

def plink_merge(bfile, bmerge, out):
    bashout = subprocess.run(
            ['plink', '--keep-allele-order', '--indiv-sort', 'none', '--make-bed',
            '--bfile', bfile,
            '--bmerge', bmerge,
            '--out', out],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert bashout.stderr.decode('utf-8')[:5] is not 'Error'
    os.remove(out+'.log')

def plink_remove(bfile, remove, out):
    bashout = subprocess.run(
        ['plink', '--keep-allele-order', '--indiv-sort', 'none', '--make-bed',
        '--bfile', bfile,
        '--remove', remove,
        '--out', out],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert bashout.stderr.decode('utf-8')[:5] is not 'Error'
    os.remove(out+'.log')

def get_homogen(merged_filepref, homogen_dist_threshold=3):
    merged_popu_df = pd.read_table(merged_filepref+'.popu', header=None)
    merged_popu_df.columns = ['fid', 'iid', 'popu']
    merged_coord = run_pca(merged_filepref, None, method='sp', load_saved_ref_decomp=False, plot_results=True)[0]
    dim_ref = merged_coord.shape[1]
    merged_coord_df = pd.DataFrame(merged_coord)
    merged_coord_df = pd.DataFrame(merged_coord)
    merged_df = pd.concat([merged_popu_df, merged_coord_df], axis=1)
    merged_popu = merged_df['popu']
    merged_popu_uniq = np.unique(merged_popu)
    merged_inlier_df = merged_df[0:0]
    for pp in merged_popu_uniq:
        if pp[-8:] == 'borrowed':
            pp_base = pp[:3]
            pp_ref_df = merged_df[merged_popu == pp_base]
            pp_stu_df = merged_df[merged_popu == pp]
            pp_ref_coord = merged_coord[merged_popu == pp_base]
            pp_stu_coord = merged_coord[merged_popu == pp]
            n_ref = pp_ref_coord.shape[0]
            n_stu = pp_stu_coord.shape[0]
            mn, std, U, s, V, pp_ref_pcs = pca_ref(pp_ref_coord.T, dim_ref=dim_ref)
            se = np.std(pp_ref_pcs, axis=0)
            pp_stu_pcs = pca_stu(pp_stu_coord.T, mn, std, 'sp', U)
            pp_stu_dist = np.sqrt(np.sum(pp_stu_pcs**2 / se**2, axis=1) / pp_stu_pcs.shape[1])
            pp_stu_isin = pp_stu_dist < homogen_dist_threshold
            pp_stu_inlier_df = pp_stu_df[pp_stu_isin]
            # if not remember_borrowed:
            #    pp_stu_inlier_df['popu'] = pp_base
            merged_inlier_df = pd.concat((merged_inlier_df, pp_stu_inlier_df), axis=0)
            # print(se[:4])
            # pp_stu_popu = pp_stu_isin
            # plot_pcs(pp_ref_pcs, pp_stu_pcs, popu_stu=pp_stu_isin, method='sp', out_pref=merged_filepref+'_'+pp_base)
        else:
            pp_ref_df = merged_df[merged_popu == pp]
            merged_inlier_df = pd.concat((merged_inlier_df, pp_ref_df), axis=0)
    # print(merged_df.groupby('popu').count().iloc[:,0])
    # print('Number of samples: ' + str(merged_df.shape[0]))
    # print(merged_inlier_df.groupby('popu').count().iloc[:,0])
    # print('Number of inliers: ' + str(merged_inlier_df.shape[0]))
    logging.info('Samples left: ' + str(merged_inlier_df.shape[0]) + '/' + str(merged_df.shape[0]))
    logging.info('.' * 10)
    merged_inlier_df.iloc[:,:3].to_csv(merged_filepref+'.popu', sep='\t', header=False, index=False)
    plink_keep(merged_filepref, merged_filepref+'.popu', merged_filepref)
    # if os.path.isfile(merged_filepref+'_ref.pcs'):
    #     os.remove(merged_filepref+'_ref.pcs')
    if merged_inlier_df.shape[0] != merged_df.shape[0]:
        get_homogen(merged_filepref, homogen_dist_threshold)

def add_pure_stu(ref_filepref, stu_filepref, n_pure_samples=1000, popu_purity_threshold=0.75, homogen_dist_threshold=2, n_iter_max=10):
    ref_basepref = os.path.basename(ref_filepref)
    stu_popu_file = stu_filepref + '_sturef_' + ref_basepref + '_pred_ap.popu'
    pure_filepref = stu_filepref + '_pure'
    merged_filepref = ref_filepref + '_withloan'
    # Create popu for pure study samples
    # Find samples whose purity is greater than popu_purity_threshold
    logging.info('Finding pure study samples...')
    popu_df = pd.read_table(stu_popu_file, header=None)
    proba = popu_df[3]
    popu_is_pure = proba > popu_purity_threshold
    assert np.any(popu_is_pure)
    popu_pure_df = popu_df.loc[popu_is_pure]
    popu_pure = popu_pure_df[2]

    # Select the top nearest n_pure_samples individuals in each population by distance to neighbors
    logging.info('Select samples with closest neighbors...')
    popu_pure_unique = np.unique(popu_pure)
    popu_pure_near_df = popu_pure_df[0:0]
    for pp in popu_pure_unique:
        popu_pure_this_df = popu_pure_df.loc[popu_pure==pp]
        popu_pure_this_near_ind = np.argsort(popu_pure_this_df[4])[:n_pure_samples] # col 4 = dist to k^th nearest neighbor
        popu_pure_this_near_df = popu_pure_this_df.iloc[popu_pure_this_near_ind]
        popu_pure_near_df = pd.concat((popu_pure_near_df, popu_pure_this_near_df), axis=0)
    for i in range(popu_pure_near_df.shape[0]):
        popu_pure_near_df.iloc[i,2] += '-borrowed'
    popu_pure_near_df = popu_pure_near_df.iloc[:,:3]
    popu_pure_near = popu_pure_near_df[2]
    logging.info("Total: " + str(len(popu_pure_near)))
    logging.info(np.unique(popu_pure_near, return_counts=True))
    popu_pure_near_df.to_csv(pure_filepref+'.popu', sep='\t', header=False, index=False)

    # Create bed, bim, fam for pure study samples
    plink_keep(stu_filepref, pure_filepref+'.popu', pure_filepref)

    # Merge pure study samples with reference samples
    plink_merge(ref_filepref, pure_filepref, merged_filepref)
    concat_files([ref_filepref+'.popu', pure_filepref+'.popu'], merged_filepref+'.popu')
    if os.path.isfile(merged_filepref+'_ref.pcs'):
        os.remove(merged_filepref+'_ref.pcs')

    # Select homogeneous samples within the pure samples
    logging.info('Select homogeneous samples within the pure samples...')
    logging.info('='*10)
    get_homogen(merged_filepref, homogen_dist_threshold)
    logging.info('Homogeneous samples save to ' + merged_filepref)
    ref_merged_df = pd.read_table(merged_filepref+'.popu', header=None)
    logging.info("Merged samples: ")
    logging.info(np.unique(ref_merged_df[2], return_counts=True))
    logging.info('='*10)
    return merged_filepref

def concat_files(inlist, out):
    with open(out, 'w') as outfile:
        for fname in inlist:
            with open(fname) as infile:
                outfile.write(infile.read())

def merge_array_results(ref_filepref, stu_filepref, method, n_chunks):
    ref_basepref = os.path.basename(ref_filepref)
    endcode_list = [str(i).zfill(SAMPLE_SPLIT_PREF_LEN) for i in range(n_chunks)]
    stu_filepref_list = [stu_filepref + '_' + endcode_list[i] + '_sturef_' + ref_basepref for i in range(n_chunks)]
    stu_pcs_filename_list = [fpref + '_stu_' + method + '.pcs' for fpref in stu_filepref_list]
    stu_popu_filename_list = [fpref + '_pred_' + method + '.popu' for fpref in stu_filepref_list]
    stu_fam_filename_list = [stu_filepref + '_' + endcode_list[i] + '.fam' for i in range(n_chunks)]
    stu_pcs_filename = stu_filepref + '_sturef_' + ref_basepref + '_stu_' + method + '.pcs'
    stu_popu_filename = stu_filepref + '_sturef_' + ref_basepref + '_pred_' + method + '.popu'
    stu_fam_filename = stu_filepref + '.fam'
    concat_files(stu_pcs_filename_list, stu_pcs_filename)
    concat_files(stu_popu_filename_list, stu_popu_filename)
    concat_files(stu_fam_filename_list, stu_fam_filename)
    # ref_pcs_filename = ref_filepref + '_ref.pcs'
    # ref_popu_filename = ref_filepref + '.popu'
    # ref_pcs = np.loadtxt(ref_pcs_filename)
    # stu_pcs = np.loadtxt(stu_pcs_filename)
    # ref_popu = np.loadtxt(ref_popu_filename, dtype=np.object)[:,2]
    # stu_popu = np.loadtxt(stu_popu_filename, dtype=np.object)[:,2]
    # plot_pcs(ref_pcs, stu_pcs, ref_popu, stu_popu, method, out_pref=stu_filepref)

def plot_pcs(pcs_ref, pcs_stu=None, popu_ref=None, popu_stu=None, method=None, out_pref=None, markers=PLOT_MARKERS, alpha_ref=PLOT_ALPHA_REF, alpha_stu=PLOT_ALPHA_STU, plot_lim=None, plot_dim=float('inf'), plot_size=None, plot_title=None, plot_color_stu=None, plot_legend=True, plot_centers=False):
    n_ref, dim_ref = pcs_ref.shape
    if pcs_stu is None:
        n_stu = 0
    else:
        n_stu = pcs_stu.shape[0]
    if method is None:
        method = 'stu'
    if popu_ref is None:
        popu_ref = np.array(['ref'] * n_ref)
    else:
        popu_ref = np.array(popu_ref, dtype='str')
    if popu_stu is None:
        popu_stu = np.array([popu_ref[0]] * n_stu)
    else:
        popu_stu = np.array(popu_stu, dtype='str')

    # Get the unique populations and assign plotting colors to them
    popu_unique = sorted(list(set(np.concatenate((popu_ref, popu_stu)))))
    popu_unique = sorted(popu_unique, key=len)
    popu_n = len(popu_unique)
    colormap = plt.get_cmap('tab10')
    popu2color = dict(zip(popu_unique, colormap(range(popu_n))))

    # Plotting may need to change the signs of PC scores
    # Make a copy to avoid modifying original arrays
    pcs_ref = np.copy(pcs_ref)
    if pcs_stu is not None:
        pcs_stu = np.copy(pcs_stu)
    if plot_lim is not None:
        plot_lim = np.copy(plot_lim)

    # Make sure the same population has the same PC score "shape" when analyzed by different methods
    geocenter_coord = geocenter_coordinate(pcs_ref, popu_ref).values()
    geocenter_coord = list(geocenter_coord)
    geocenter_dist = np.array([np.linalg.norm(v) for v in geocenter_coord])
    geocenter_farthest_sign = np.sign(geocenter_coord[geocenter_dist.argmax()])
    for i in range(dim_ref):
        sign_this = geocenter_farthest_sign[i]
        if sign_this < 0:
            pcs_ref[:,i] *= sign_this
            if pcs_stu is not None:
                pcs_stu[:,i] *= sign_this
            if plot_lim is not None:
                plot_lim[:,i] *= sign_this
                plot_lim[:,i] = plot_lim[:,i][::-1]

    n_subplot = int(min(dim_ref, plot_dim) / 2)
    fig, ax = plt.subplots(ncols=n_subplot)
    for j in range(n_subplot):
        plt.subplot(1, n_subplot, j+1)
        plt.xlabel('PC' + str(j*2+1))
        plt.ylabel('PC' + str(j*2+2))
        for i,popu in enumerate(popu_unique):
            ref_is_this_popu = popu_ref == popu
            pcs_ref_this_popu = pcs_ref[ref_is_this_popu, (j*2):(j*2+2)]
            plot_color_this = popu2color[popu]
            if plot_centers:
                label = None
            else:
                label = str(popu)
            plt.scatter(pcs_ref_this_popu[:,0], pcs_ref_this_popu[:,1], marker=markers[-1], alpha=alpha_ref, color=plot_color_this, label=label)
            if plot_centers:
                pcs_ref_this_popu_mean = np.mean(pcs_ref_this_popu, axis=0)
                plt.scatter(pcs_ref_this_popu_mean[0], pcs_ref_this_popu_mean[1], marker=markers[-2], color=plot_color_this, edgecolor='xkcd:grey', s=300, label=str(popu))
        if pcs_stu is not None:
            if plot_color_stu is None:
                plot_color_stu_list = np.array([popu2color[popu_this] for popu_this in popu_stu], dtype=np.object)
            else:
                plot_color_stu_list = np.array([plot_color_stu] * pcs_stu.shape[0], dtype=np.object)
            a = 5
            for i in range(a):
                if i == 0:
                    label = 'stu (' + method + ')'
                else:
                    label = None
                indiv_shuffled_this = np.arange(pcs_stu.shape[0]) % a == i
                plt.scatter(pcs_stu[indiv_shuffled_this, j*2],
                            pcs_stu[indiv_shuffled_this, j*2+1],
                            color=plot_color_stu_list[indiv_shuffled_this].tolist(),
                            marker=markers[0], alpha=alpha_stu, label=label)
            if plot_centers:
                for i,popu in enumerate(popu_unique):
                    stu_is_this_popu = popu_stu == popu
                    pcs_stu_this_popu = pcs_stu[stu_is_this_popu, (j*2):(j*2+2)]
                    pcs_stu_this_popu_mean = np.mean(pcs_stu_this_popu, axis=0)
                    plt.scatter(pcs_stu_this_popu_mean[0], pcs_stu_this_popu_mean[1], marker=markers[-2], color='xkcd:grey', s=100)
        if plot_lim is not None:
            plt.xlim(plot_lim[:,j*2])
            plt.ylim(plot_lim[:,j*2+1])
    if plot_legend:
        plt.legend()
    if plot_title is not None:
        plt.title(str(method)+' '+plot_title, fontsize=30)
    plt.tight_layout()
    if plot_size is not None:
        fig.set_size_inches(plot_size)
    if out_pref is not None:
        fig_filename = out_pref + '_' + method + '.png'
        plt.savefig(fig_filename, dpi=300)
        logging.info('PC plots saved to ' + fig_filename)
    else:
        logging.info('No output path specified.')
        plt.show()
    plt.close('all')

def pred_popu_ref(pcs_ref, n_clusters, out_filename=None, rowhead_df=None):
    popu_ref = KMeans(n_clusters=n_clusters).fit_predict(pcs_ref)
    if out_filename is not None:
        popu_ref_df = pd.DataFrame({'popu':popu_ref})
        popu_ref_df = pd.concat([rowhead_df, popu_ref_df], axis=1)
        popu_ref_df.to_csv(out_filename, sep=DELIMITER, header=False, index=False)
        print('Reference clustering saved to ' + out_filename)
    popu_ref = np.array([str(pp) for pp in popu_ref])
    return popu_ref

def pred_popu_stu(pcs_ref, popu_ref, pcs_stu, out_filename=None, rowhead_df=None, weights='uniform'):
    logging.info('Predicting populations for study individuals...')
    n_stu = pcs_stu.shape[0]
    popu_list = np.sort(np.unique(popu_ref))
    popu_dic = {popu_list[i] : i for i in range(len(popu_list))}
    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights=weights)
    knn.fit(pcs_ref, popu_ref)
    popu_stu_pred = knn.predict(pcs_stu)
    popu_stu_proba_list = knn.predict_proba(pcs_stu)
    popu_stu_proba = [popu_stu_proba_list[i, popu_dic[popu_stu_pred[i]]] for i in range(n_stu)]
    popu_stu_dist = knn.kneighbors(pcs_stu)[0][:,-1]
    popu_stu_dist = np.round(popu_stu_dist, 3)
    if out_filename is not None:
        popuproba_df = pd.DataFrame({'popu':popu_stu_pred, 'proba':popu_stu_proba, 'dist':popu_stu_dist})
        popuproba_df = popuproba_df[['popu', 'proba', 'dist']]
        probalist_df = pd.DataFrame(popu_stu_proba_list)
        populist_df = pd.DataFrame(np.tile(popu_list, (n_stu, 1)))
        popu_stu_pred_df = pd.concat([rowhead_df, popuproba_df, probalist_df, populist_df], axis=1)
        popu_stu_pred_df.to_csv(out_filename, sep=DELIMITER, header=False, index=False)
        logging.info('Predicted study populations saved to ' + out_filename)
    return popu_stu_pred, popu_stu_proba, popu_stu_dist

def bed2trace(filepref, missing=3):
    log = create_logger(filepref, 'info')
    bed, bim, fam = read_bed(filepref)
    for idx, x in np.ndenumerate(bed):
        if x == missing:
            bed[idx] = -9
    bed_df = pd.DataFrame(bed.T)
    tracegeno = pd.concat([fam[['fid', 'iid']], bed_df], axis=1, join_axes=[fam.index])
    tracegeno.to_csv(filepref+'.geno', sep='\t', header=False, index=False)

    tracesite = pd.DataFrame({
        'CHROM': bim['chrom'],
        'POS': bim['pos'],
        'ID': bim.index.values,
        'REF': bim['a1'],
        'ALT': bim['a2']
    })
    tracesite = tracesite[['CHROM', 'POS', 'ID', 'REF', 'ALT']]
    tracesite.to_csv(filepref+'.site', sep='\t', header=True, index=False)

def trace2bed(trace_filepref, missing=3):
    bed_filepref = trace_filepref

    with open(trace_filepref+'.geno', 'r') as f:
        ncols = len(f.readline().split())
    bed = np.loadtxt(trace_filepref+'.geno', dtype=np.int8, usecols=range(2, ncols)).T
    bed[bed == -9] = missing
    bed = 2 - bed
    with PyPlink(bed_filepref, 'w') as pyp:
        for row in bed:
            pyp.write_genotypes(row)

    fam = pd.read_table(trace_filepref+'.geno', usecols=range(0, 2), header=None)
    popu = pd.concat([fam, fam.iloc[:,0]], axis=1)
    popu.to_csv(bed_filepref+'.popu', sep='\t', header=False, index=False)
    nrows = fam.shape[0]
    filler = pd.DataFrame(np.zeros((nrows, 4), dtype=np.int8))
    fam = pd.concat([fam, filler], axis=1)
    fam.to_csv(bed_filepref+'.fam', sep='\t', header=False, index=False)

    site = pd.read_table(trace_filepref+'.site')
    bim = site[['CHR', 'ID', 'POS', 'REF', 'ALT']]
    bim.insert(2, 'cm', 0)
    bim.to_csv(bed_filepref+'.bim', sep='\t', header=False, index=False)

def intersect_ref_stu_snps(pref_ref, pref_stu):
    snps_are_identical = filecmp.cmp(pref_ref+'.bim', pref_stu+'.bim')
    if snps_are_identical:
        logging.info('SNPs and alleles in reference and study samples are identical')
    else:
        logging.error('Error: SNPs and alleles in reference and study samples are not identical')
        assert False
        # logging.info('Intersecting SNPs in reference and study samples...')
        # bashout = subprocess.run(
        #     ['bash', 'intersect_bed.sh', pref_ref, pref_stu],
        #     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # assert len(bashout.stderr.decode('utf-8')) == 0
        # pref_ref_commsnpsrefal, pref_stu_commsnpsrefal = bashout.stdout.decode('utf-8').split('\n')[-3:-1]
        # assert len(pref_ref_commsnpsrefal) > 0
        # assert len(pref_stu_commsnpsrefal) > 0
        # return pref_ref_commsnpsrefal, pref_stu_commsnpsrefal
    return pref_ref, pref_stu

def load_trace(filename, isref=False):
    trace_df = pd.read_table(filename)
    if isref:
        n_col_skip = 2
    else:
        n_col_skip = 6
    trace_pcs = trace_df.iloc[:, n_col_skip:].values
    return trace_pcs

def geocenter_similarity(Y, Y_ctr, X, X_ctr, dim=None):
    if dim is not None:
        X = np.copy(X[:,:dim])
        Y = np.copy(Y[:,:dim])
    assert X.shape[1] == Y.shape[1]
    X_ctr_coord_dic = geocenter_coordinate(X, X_ctr)
    Y_ctr_coord_dic = geocenter_coordinate(Y, Y_ctr)
    dist = 0
    for ctr in Y_ctr_coord_dic:
        dist += np.sum((Y_ctr_coord_dic[ctr] - X_ctr_coord_dic[ctr])**2)
    return dist

def procrustes_similarity(Y_mat, X_mat, dim=None):
    if dim is not None:
        X = np.copy(X[:,:dim])
        Y = np.copy(Y[:,:dim])
    X = np.array(X_mat, dtype=np.double, copy=True)
    Y = np.array(Y_mat, dtype=np.double, copy=True)
    assert X.shape == Y.shape
    X_transformed = procrustes(Y, X, return_transformed=True)[-1]
    Z = Y - X_transformed
    trZZ = np.sum(Z**2)
    Y_mean = np.mean(Y, 0)
    Y -= Y_mean
    trYY = np.sum(Y**2)
    similarity = 1 - (trZZ / trYY)
    assert 0 <= similarity <= 1
    return similarity

def geocenter_coordinate(X, X_ctr):
    X_ctr = np.array(X_ctr)
    p = X.shape[1]
    X_ctr_unique = np.unique(X_ctr)
    X_ctr_unique_n = len(X_ctr_unique)
    X_ctr_unique_coord = np.zeros((X_ctr_unique_n, p))
    for i,ctr in enumerate(X_ctr_unique):
        is_this_ctr = X_ctr == ctr
        if np.sum(is_this_ctr) > 0:
            X_ctr_unique_coord[i] = np.mean(X[is_this_ctr], axis=0)
        else:
            X_ctr_unique_coord[i] = np.zeros(p)
    X_ctr_coord_dic = {X_ctr_unique[i] : X_ctr_unique_coord[i] for i in range(X_ctr_unique_n)}
    return X_ctr_coord_dic
