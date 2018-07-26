
def normratio_diags_indicatormatrix(M, I, ignore_diags=0, exclude_nans_from_paircounts=True):
    """
    Computes, for all upper diagonals, the normalized mean intensity ratio 
    (normratio) between two sets, specified by I, of pixels in M 
    
    The two sets of pixels are specified by the indicator matrix I as follows:
    set1:    all pixels (i,j) with I(i,j)=1
    set2:    all pixels (i,j) with I(i,j)=0
    neither: all pixels (i,j) with I(i,j)=nan  (I(i,j)=nan  =>  M(i,j) ignored)
    
    Properties: 
    - normratio(s) = (set1_avintens(s) - set2_avintens(s)) / (set1_avintens(s) + set2_avintens(s))
                   = (r-1)/(r+1) for r=set1_avintens(s)/set2_avintens(s)
    - normratio(s) is between -1 and 1, 
    - normratio(s)=0 for equal average intensity in  set1 and set2 (in diagonal s) 
    - normratio(s)=+/-1 for zero intensity set 2

    Lower halfes of M and I are ignored but a warning is thrown if M is not symmetric
    
    If exclude_nans_from_paircounts==True, pixels M(i,j)=nan entail I(i,j)=nan 
    (such that those pixels don't count towards neither set1 nor set2)
    
    Note: the bare ratio (not normalized) can be computed from normratio as:
    ratio = (normratio+1)/(normratio-1)
    
    example: 
    ---------------------
    
    M = [3 2 1        I = [1 1 0
         . 1 1             . 1 0
         . . 1]            . . nan]
    
    The resulting below quantities are by diagonal with increasing offset s=0..len(M):
    
    set1_intens    = [4, 2, 0]
    set1_pixels    = [2, 1, 0]
    =>
    set1_avintens  = [2, 2, nan] 

    set2_intens    = [0, 1, 1]
    set1_pixels    = [0, 1, 1]
    =>
    set2_avintens  = [nan, 1, 1] 

    => 
    normratio_diags_indicatormatrix(M, I, ignore_diags=0) 
        = (set1_avintens - set2_avintens) / (set1_avintens + set2_avintens)
        = [nan, 1/3, nan]     
          indeed, only 1st off-diagonal has both types, others have to be nans
    
    
    parameters:
    -----------------------
    M: 2D numpy array: 
        the data, eg. a Hi-C contact matrix
    I: 2D numpy array: 
        indicator matrix can contain only 
        - zeros (for type 1), 
        - ones (for type 2), 
        - NANs (neither) 
        otherwise a value error is raised
        must have same shape as M
    ignore_diags: int: 
        the entries [0:ignore_diags] will be NANs in all returns
    exclude_nans_from_paircounts: Boolean
        if True, I(i,j) gets nans for all pixels (i,j) where M is nan 

    
    return:
    -----------------------
    normratio: 1D array:
        mean normalized intensity ratio, by diagonal s
    
    additional_info: list: [    
        set1_intens: 1D array, 
            nansum(diag(I*M)) for each diag, i.e. number of contacts 
            in indicated parts for each genomic separation
        set1_pixels:  1D array, 
            nansum(diag(I)) for each diag, i.e. number of valid pairs 
            in indicated parts for each genomic separation
        set2_intens: 1D array, 
            nansum(diag((1-I)*M)) for each diag, i.e. number of contacts 
            in non-indicated parts for each genomic separation
        set2_pixels:  1D array, 
            nansum(1-diag(I)) for each diag, i.e. number of valid pairs 
            non-in indicated parts for each genomic separation
        set1_avintens:   1D array, 
            set1_intens/set1_pixels: average intensity within indicated parts, 
            for each diagonal 
        set2_avintens:   1D array, 
            set2_intens/set2_pixels: average intentity in non-indicated parts, 
            for each diagonal (but NANs arent here either)
        ]
    """
    
    import warnings
    
    # check if M is symmetric
    if 1:
        tol = 1e-8
        if not np.allclose(M,M.T,atol=tol):
            warnings.warn("M is not symmetric to within {}, I'm using the upper half".format(tol))
            
    # check if I has only 0,1,nan
    Iu = np.unique(I)
    nanIu = [i for i in Iu if not np.isnan(i)]
    illegalvals = [i for i in nanIu if i not in [0., 1.]]
    if np.any(illegalvals):
        raise ValueError("I is not an indicator matrix: it contains value(s) other than {0,1,np.nan}:", illegalvals)

    # check if M and I have same shape
    if not M.shape == I.shape:
        raise ValueError('M and I must have the same shape')

    L = len(M)

    # make a copy if indicator matrix
    Iuse = np.copy(I).astype(float)
        
    # exclude pixels with M=np.nan from counts towards valid pairs
    if exclude_nans_from_paircounts:
        Iuse[np.isnan(M)] = np.nan
        
    # compute valid paris, counts and COMP score
    set1_pixels = np.zeros(L)*np.nan
    set1_intens = np.zeros(L)*np.nan
    set2_pixels = np.zeros(L)*np.nan
    set2_intens = np.zeros(L)*np.nan
    for s in range(ignore_diags,L):
        diagI = np.diag(Iuse,s) # get diagonal s of indicator matrix
        diagM = np.diag(M,s)    # get diagonal s of data matrix
        
        set1_pixels[s] = np.nansum(diagI)   # number of valid pixels in set 1
        set2_pixels[s] = np.nansum(1-diagI) # number of valid pixels in set 2
        
        set1_intens[s] = np.nansum(diagM*diagI)        
        set2_intens[s] = np.nansum(diagM*(1-diagI))

    with np.errstate(invalid='ignore'):
        set1_avintens = set1_intens/set1_pixels # intensity average of valid pixels in set 1 in diagonal s
        set2_avintens = set2_intens/set2_pixels # intensity average of valid pixels in set 2 in diagonal s

        normratio = (set1_avintens-set2_avintens) / (set1_avintens+set2_avintens)
    
    additional_info = [set1_intens, set1_pixels, set2_intens, set2_pixels, set1_avintens, set2_avintens]

    return normratio, additional_info




def normratio_types(M, v, ignore_diags=0, compute_all_types=False, exclude_nans_from_paircounts=True, verbosity=0):
    """
    Computes, for diagonals with offset s=0..len(M), several "contrasts" in M: 
        normratio_all:  within types to across types, 
                        eg. {AA,BB}-against-{AB}
        normratio_type: computed if compute_all_types==True
                        one type to all others, 
                        eg. AA-against-{AB,BB}
        
    "contrast" is the normalized mean intensity ration (see below for details)
    
    types of loci are the unique values of v (#types = #unique values in v). 
        eg. v=[0,10,0,-2] has types {0,10,-2}, with 0 at [0,2], 10 at [1], -2 at [3]
        eg. v=[1,nan,1] has types {1} at [0,2], the nan is ignored
    
    Loci with v[i]=np.nan are ignored.

    typical usage: 
    M is a contact matrix, e.g. from a Hi-C experiment (simulated or real)
    v is a vector of zeros and ones, representing compartmental types of loci
    
    Note:
        if more specialized normratios are needed, e.g. type1 -vs- type2 but not type3, 
        build indicator matrices yourself and use normratio_diags_indicatormatrix
                    
    Method:
    - the indicator matrix for {within types} -vs- {across types} is:
        - I_all(i,j)  = 1   if i,j of same type
        - I_all(i,j)  = 0   if i,j of different types
    - for each type in v an indicator matrixis computed with: 
        - I_type(i,j) = 1   if i,j of this type
        - I_type(i,j) = 0   if i,j of different types
        - I_type(i,j) = nan if i,j of same type but not this type
    - the normratio of M for the above indicator matrices is computed
    - nans in M are ignored
    
    Definition of "contrast":
    the normalized mean intensity ratio between two sets of pixels (in a given diagonal):
    normratio(s) = (set1_avintens(s) - set2_avintens(s)) / (set1_avintens(s) + set2_avintens(s))
                 = (r-1)/(r+1) for r=set1_avintens(s)/set2_avintens(s)
    see function 'normratio_diags_indicatormatrix' for details
    
    
    
    example:
    --------
    
    v = [0,0,1,2,1]   =>   types = [0 ,1 ,2]    
    
    Indicator matrices:
    
    type:0
    [[  1.   1.   0.   0.   0.]
     [  1.   1.   0.   0.   0.]
     [  0.   0.  nan   0.  nan]
     [  0.   0.   0.  nan   0.]
     [  0.   0.  nan   0.  nan]]
    type:1
    [[ nan  nan   0.   0.   0.]
     [ nan  nan   0.   0.   0.]
     [  0.   0.   1.   0.   1.]
     [  0.   0.   0.  nan   0.]
     [  0.   0.   1.   0.   1.]]
    type:2
    [[ nan  nan   0.   0.   0.]
     [ nan  nan   0.   0.   0.]
     [  0.   0.  nan   0.  nan]
     [  0.   0.   0.   1.   0.]
     [  0.   0.  nan   0.  nan]]
    any type:
    [[ 1.  1.  0.  0.  0.]
     [ 1.  1.  0.  0.  0.]
     [ 0.  0.  1.  0.  1.]
     [ 0.  0.  0.  1.  0.]
     [ 0.  0.  1.  0.  1.]]                
                
                
    Internal construction of those matrices (pseudocode):
    
    for type in types:
        v_aux = nans(len(v))
        v_aux[v==type] = 1
        I_type = np.outer(v_aux,v_aux)  
    =>  I_type: 1 for (i,j) both of this type, nan elsewhere
    
    v_aux = nans(len(v))
    v_aux[~np.isnan(v)] = 0
    I_anytype = np.outer(v_aux, v_aux)
    for i in range(len(types)):
        I_anytype[I_types==1] = 1       
    =>  I_anytype: zero for across types, one for within any type, nan for (i,j) with types[i]==nan or types[j]==nan
        
    for i in range(len(types)):
        I_type[I_anytype==0] = 0        
    =>  I_type: 1 for (i,j) both of this type, 0 for across-types, 
               nan elsewhere (within, but not this type)
        
    
    

    parameters:
    -----------
    M: 2D np.array
        typically a contact matrix
    
    v: sequence of types: 
        has to be len(v)=len(M)
        a vector of types, e.g zeros and ones, 
        the set of types will be np.unique(types)
        v[i]==np.nan => i-th row and column in M is ignored
        !!!  if for example a plain EV is passed ther will be MANY types ...
        !!!  ... and, if compute_all_types==True, normratios are computed for ALL
    
    compute_all_types: Boolean: 
        if False, only normratio_all is computed and the returns "..._types" are []
    
    exclude_nans_from_paircounts: Boolean
        if True, the indicator matrices get nans for all pixels where M is nan 
    

    returns:
    --------
    normratio_anytype: vector: 
        the contrast score   within_types-against-across_types
    
    add_info_anytype: list: 
        [    
        wi_counts: vecotr, nansum(diag(I*M)) for each diag, i.e. number of contacts in I_all for each genomic separation
        wi_pairs:  vector, nansum(diag(I))   for each diag, i.e. number of valid pairs in I_all for each genomic separation
        ac_counts: vecotr, nansum(diag(I*M)) for each diag, i.e. number of contacts in 1-I_all for each genomic separation
        ac_pairs:  vector, nansum(diag(I))   for each diag, i.e. number of valid pairs in 1-I_all for each genomic separation
        wi_mean:   vector, wi_counts/wi_pairs: average intensity within I_all, for each diagonal 
        ac_mean:   vector, ac_counts/ac_pairs: average intensity in 1-I_all, for each diagonal (but NANs arent here either)
        ]
    
    I_anytype: matrix:
        the indicator matrix for within-anytype -vs- across-type
        
    normratio_types: list: 
        [] if compute_all_types==0, otherwise list of normratios(type - all_ohters) for all types
    
    add_info_types: list of lists:
        each as above
    
    I_types: list of matrices:
        the indicator matrices for within-thistype -vs- across-types
    """

    if len(M) != len(v):
        raise ValueError('M and v must have same length')

    v = np.asarray(v)
        
    L = len(v)

    # find types
    types = np.unique(v[~np.isnan(v)]) # this casts all int to floats => work with thhose
    if verbosity:
        print(types)

    # construc indicator matrices
    I_types = []
    for thistype in types:
        v_aux = np.zeros(L)*np.nan
        v_aux[v==thistype] = 1
        I_thistype = np.outer(v_aux,v_aux)   #=> I_thistype: 1 for (i,j) both of this type, nan elsewhere
        I_types.append(I_thistype)

    #I_anytype = np.zeros((L,L))
    v_aux = np.zeros(L)*np.nan
    v_aux[~np.isnan(v)] = 0
    I_anytype = np.outer(v_aux, v_aux)
    for i in range(len(types)):
        I_anytype[I_types[i]==1] = 1         #=> I_anytype: zero for across types, one for withing any type

    for i in range(len(types)):
        I_types[i][I_anytype==0] = 0         #=> I_thistype: 1 for (i,j) both of this type, 0 for across-types, nan elsewhere (within, but not this type)
    
    # compute normratio_anytype 
    normratio_anytype,add_info_anytype = normratio_diags_indicatormatrix(M, I_anytype, ignore_diags=ignore_diags, exclude_nans_from_paircounts=exclude_nans_from_paircounts)
    
    # compute COMPsocre_types
    normratio_types = []
    add_info_types = []
    if compute_all_types:
        for i in range(len(types)):
            normratio_thistype,add_info_thistype = normratio_diags_indicatormatrix(M, I_types[i], ignore_diags=ignore_diags, exclude_nans_from_paircounts=exclude_nans_from_paircounts)
            normratio_types.append(normratio_thistype)
            add_info_types.append(add_info_thistype) 

    return normratio_anytype, add_info_anytype, I_anytype, normratio_types, add_info_types, I_types



def COMPscore_by_s(M, comp_identities ,ignore_diags=0, exclude_nans_from_paircounts=True, verbose=True):
    """
    compute COMPscore, a measure for the "contrast" in M: 
        - COMPscore_by_s (contrast in diagonals with offset s=0..len(M))
        - COMPscore: weighted average over all s
    
    parameters:
    -----------
    M: 2D numpy array or string:
        (filepath of) matrix
        
    comp_indentities: sequence or lambda function or None:
        if sequence:
            true compartmental identities of the monomers, e.g. [1, 1, -1, -1, 1]
            has to be same length as M (before internal coarsegraining)
        if lambda function: 
            comp_identities(EV) is used as compartmental identites of monomers
            where EV is the eigenvector from 
            params: an eigenvector (1D numpy array)
            returns: a vector of compartmental identities of same length
            example: comp_indentities=lambda x: x>np.nanmedian(x)
        if None:
            comp_identities(EV) is used as compartmental identites of monomers, where:
                comp_identities=lambda x: x>np.nanmedian(x)
                EV = cooltools.eigdecomp.cis_eig(M, phasing_track=phasing_track)[1][0]
    
    phasing_track: vector, optional: 
        to flip EVs (only effective if comp_indentities is a lambda function or None)        
        use for example truecomps
        has to be same length as M (before internal coarsegraining)
        
    ignore_diags: integer, optional: 
        number of diagonals to be ignored
        
    exclude_nans_from_paircounts: boolean, optional:
        if True, pixels with NaN in M are not counted towards possible contacts
        
    
    returns:
    --------
    COMPscore_by_s: 1D numpy array, len=len(M): 
        COMPscore(genomic separation) across conditions, 
        using truecompsas compatmental types

    COMPscore: float:
        weighted mean of COMPscore_by_s:
        np.nansum(COMPscore*p)/np.nansum(p) with weights:
        p[s]=add_info_anytype[1][s]*add_info_anytype[3][s] # #wi_pairs*#ac_pairs  
    """

    import joblib
    import os
    #from mirnylib.numutils import coarsegrain
    import cooltools
    from cooltools import eigdecomp
    
    # get heatmap
    if isinstance(M, np.ndarray) and (len(M.shape)==2):
        if verbose:
            print('cmap was supplied as 2D array')
    elif isinstance(cmaps[i], str): 
        if verbose:
            print('folder: ', cmaps[i].split('/')[-2])
            print('hmap:', cmaps[i].split('/')[-1])
        try:
            c = joblib.load(cmaps[i])
        except:
            raise ValueError("cmap could not be loaded")
    else:
        raise ValueError("M was neither 2d-array nor joblib loadable")
    if verbose:
        print('got a contact map')    
    
    # get compartmental identities
    if (comp_identities is None) or (callable(comp_identities)):
        
        # get eigenvector
        _, Evecs = cooltools.eigdecomp.cis_eig(M, phasing_track=phasing_track)
        EV = Evecs[0]
        
        # use default for EV => comp_IDs
        if comp_identities is None: 
            comp_identities = lambda x: x>np.nanmedian(x)

        # get comp_IDs (!!! comp_indentities changes type here - intended)
        comp_identities = comp_identities(EV)
                    
    # get COMPscore_by_s
    COMPscore_anytype, add_info_anytype, I_anytype, COMPscore_types, add_info_types, I_types = normratio_types(M, comp_identities, ignore_diags=ignore_diags)
    COMPscore_by_s = COMPscore_anytype
    
    # get s-average
    p = add_info_anytype[1]*add_info_anytype[3] # weights: #wi_pairs*#ac_pairs            
    COMPscore = np.nansum(COMPscore_by_s*p)/np.nansum(p)
    
    return COMPscore_by_s, COMPscore