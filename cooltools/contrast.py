import warnings
import os
import numpy as np
import joblib
from . import eigdecomp
import cooler


# ==============================================================
#  CONTRAST FUNCTIONS
# ==============================================================


def contrast_diags(
    M,
    modality="AnyAny_vs_Mixed",
    modality_params=None,
    I=None,
    ignore_diags=1,
    exclude_nans_from_paircounts=True,
    i0=None,
    i1=None,
    phasing_track=None,
    normalize=False,
    verbose=False,
):
    """
    Compute the contrast in M between different sets of pixels specified by
    modality, modality_params or by I, if given.

    Returns the contrast in diagonals with offset s=0..len(M), and the
    weighted average over all s. If 'normalize' is True, contrast is
    normalized to be in [-1,1].

    Parameters
    ----------
    M: 2d array
        Typically a contact matrix.

    modality: string, (optional, default: "AnyAny_vs_Mixed")
        Specifies the contrast modality, potentially together ``modality_params``.
        See :func:`constrast.indicatormat` for valid modalities.

    modality_params: list, (optional if modality=='AnyAny_vs_mixed' or None)
        Parameters required by 'modality'. See :func:`contrast.indicatormat()`
        for valid modalities and required parameters.

        Typical format: ``[bin_types, Type]``, where
            bin_types: 1d array
                types of the bins, e.g. [0,0,1,1,1,0,...]
                Note: the numerical values identifying the types are
                inconsequential they are only used to classify the loci into
                groups.
            Type: number
                one of the values in bin_types: singles out a specific type

        If None, modality_params = [EV>np.nanmean(EV)],
            with EV computed here: eigdecomp.cis_eig(M, phasing_track=phasing_track)[1][0]

    I: indicator matrix, optional
        Specifies the two sets of pixels from which to compute contrast in M.
        If given, this overrides ``modality`` and ``modality_params``.
        If None, computed here as ``contrast.indicatormat(modality, modality_params)``.

    normalize: boolean, optional
        If True, contr_diags is normalized as follows
            contr_norm[s]
                = (contr[s]-1)/(contr[s]+1)
                = (set1_avintens[s] - set2_avintens[s]) / (set1_avintens[s] + set2_avintens[s]).
        The weighted mean, contr, is then computed from contr_norm[s].

    ignore_diags: integer, optional
        Number of diagonals in M to be ignored.

    exclude_nans_from_paircounts: boolean, optional
        If True, pixels with NaN in M are not counted towards valid pixels.

    i0, i1: integers, optional
        Will use (trimmed by len(M)):
            M[i0:i1,i0:i1]
            v[i0:i1],
            phasing_track[i0:i1]
        Note: EV is computed from restriced M if modality_params is None

    phasing_track: 1D numpy array, optional
        len(phasing_track) must be len(M)
        to flip EVs
        only used if modality_params is None

    Returns
    -------
    contr_diags: 1D numpy array, length M
        Contrast in diagonals with offsets s=0..len(M)

    contr: float
        Weighted mean of contrast_diags.
        np.nansum(contrast*p)/np.nansum(p)
        with weights (#set1_pixels(s)*#set2_pixels(s))**.5

    I: 2D array
        The computed (or supplied) indicator matrix.

    modality_params: list
        The computed (or supplied) modality_params useful if modality_params
        was computed here.

    additional_info: list
        [
            set1_intens: 1D array,
                nansum(diag(I*M,s)), i.e. total intensity in set1 for all s
            set1_pixels:  1D array,
                nansum(diag(I)) i.e. number of valid pixels in set1 for all s
            set2_intens: 1D array,
                nansum(diag((1-I)*M)), i.e. total intensity in set2 for all s
            set2_pixels:  1D array,
                nansum(1-diag(I)) i.e. number of valid pixels in set2 for all s
        ]

    """
    # get matrix
    if isinstance(M, np.ndarray):
        if len(M.shape) != 2:
            raise ValueError("M was an array, but dimensionality was not 2")
    else:
        try:
            if verbose:
                print("... loading M ...")
            M = joblib.load(M)
        except:
            raise ValueError("cmap could not be loaded")

    # trim M,I and phasing track
    M = M[i0:i1, i0:i1]
    if I is not None:
        I = I[i0:i1, i0:i1]
    if phasing_track is not None:
        phasing_track = phasing_track[i0:i1]

    # get indicator matirx (i.e. the two sets of pixels to be compared)
    if I is None:
        if modality_params is None:
            if modality == "AnyAny_vs_Mixed":
                if verbose:
                    print("... getting types from EV...")
                bin_types = np.ones(len(M)) * np.nan
                EV = eigdecomp.cis_eig(M, phasing_track=phasing_track)[1][0]
                val_inds = np.isfinite(EV)
                get_bin_identities = lambda x: x > np.nanmean(x)
                bin_types[val_inds] = get_bin_identities(EV[val_inds])
                modality_params = [bin_types]
            else:
                raise ValueError(
                    "modality_params=None only allowed with modality='AnyAny_vs_Mixed'. "
                    "For all other modalityes modality_params has to be given (a list). "
                    "See contrast.indicatormatrix() for valid modalities and "
                    "required modality_params."
                )

        if verbose:
            print("... constructing indicator matrix ...")
        I = indicatormat(modality, modality_params)

    # get contrast
    if verbose:
        print("... computing contrast ...")
    contr_diags, add_info = contrast_diags_indicatormatrix(
        M,
        I,
        ignore_diags=ignore_diags,
        exclude_nans_from_paircounts=exclude_nans_from_paircounts,
        normalize=normalize,
    )

    # weighted average over diagonals
    p = (
        add_info[1] * add_info[3]
    ) ** 0.5  # weights of diagonals: (#set1_pairs(s)*#set2_pairs(s))**.5
    contr = np.nansum(contr_diags * p) / np.nansum(p)

    return contr_diags, contr, I, modality_params, add_info


def contrast_diags_indicatormatrix(
    M,
    I,
    ignore_diags=0,
    exclude_nans_from_paircounts=True,
    normalize=True,
    verbose=False,
):
    """
    Computes, for all upper diagonals, the 'contrast' in M, namely the ratio
    of average intensities (values) in M in two sets of pixels.

    More formally, for diagonals with offsets s=0..len(M):
        contr[s] = <diag(M,s)>_set1 / <diag(M,s)>_set2)

    If normalize is True, the contrast is normalized as follows:
        contr_norm[s]
            = (conts[s]-1)/(contr[s]+1)
            = (<diag(M,s)>_set1 - <diag(M,s)>_set2) /
              (<diag(M,s)>_set1 + <diag(M,s)>_set2)

    The normalized contrast is between -1 and 1.

    The two sets of pixels are specified by the indicator matrix I as follows:
        set1:    all pixels (i,j) with I[i,j]=1
        set2:    all pixels (i,j) with I[i,j]=0
        neither: all pixels (i,j) with I[i,j]=nan

    If exclude_nans_from_paircounts==True, pixels M(i,j)=nan entail I(i,j)=nan
    (such that those pixels don't count towards neither set1 nor set2).

    Parameters
    ----------
    M: 2D numpy array
        the data, eg. a Hi-C contact matrix

    I: 2D numpy array
        indicator matrix specifying the two sets of pixels
        can contain only
        - zeros (for set1),
        - ones (for set2),
        - NANs (neither)
        otherwise a value error is raised
        must have same shape as M

    ignore_diags: int
        the entries [0:ignore_diags] will be NANs in all returns

    exclude_nans_from_paircounts: boolean
        if True, I(i,j) gets nans for all pixels (i,j) where M is nan

    normalize: boolean, optional
        if True, contrast is normalized as follows:
        contr_norm[s] = (contr[s]-1)/(contr[s]+1)
                      = (set1_avintens[s] - set2_avintens[s]) /
                        (set1_avintens[s] + set2_avintens[s])

    Returns
    -------
    contr_diags: 1D array
        ratio of mean values in M of set1 and set2, by diagonal s

    additional_info: list
        [
            set1_intens: 1D array,
                nansum(diag(I*M,s)), i.e. total intensity in set1 for all s
            set1_pixels:  1D array,
                nansum(diag(I)) i.e. number of valid pixels in set1 for all s
            set2_intens: 1D array,
                nansum(diag((1-I)*M)), i.e. total intensity in set2 for all s
            set2_pixels:  1D array,
                nansum(1-diag(I)) i.e. number of valid pixels in set2 for all s
        ]

    Notes
    -----
    * Lower halves of M and I are ignored but a warning is thrown if M is not
      symmetric.
    * The bare contrast can be computed from the normalized one as
      contr = (contr_norm + 1)/(contr_norm - 1).

    Examples
    --------
    M = [3 2 1        I = [1 1 0
         . 1 1             . 1 0
         . . 1]            . . nan]

    The resulting below quantities are by diagonal with increasing offset
    s=0..len(M):

    set1_intens    = [4, 2, 0]
    set1_pixels    = [2, 1, 0]
    =>
    set1_avintens  = [2, 2, nan]

    set2_intens    = [0, 1, 1]
    set1_pixels    = [0, 1, 1]
    =>
    set2_avintens  = [nan, 1, 1]
    =>
    contrast_diags_indicatormatrix(M, I, ignore_diats=0)
        = set1_avintens / set2_avintens
        = [nan, 2, nan]
          (indeed, only 1st off-diagonal has both types, others have to be nans)

    contrast_diags_indicatormatrix(M, I, ignore_diats=0, normalize=True)
        = (set1_avintens - set2_avintens) / (set1_avintens + set2_avintens)
        = [nan, 1/3, nan]
          indeed, only 1st off-diagonal has both types, others have to be nans

    """

    # check if M is symmetric
    tol = 1e-8
    if not np.allclose(M, M.T, atol=tol, equal_nan=True):
        warnings.warn(
            "M is not symmetric to within {}, " "I'm using the upper half".format(tol)
        )

    # check if I has only 0,1,nan
    Iu = np.unique(I)
    nanIu = Iu[~np.isnan(Iu)]
    illegalvals = [i for i in nanIu if i not in [0.0, 1.0]]
    if np.any(illegalvals):
        raise ValueError(
            "I is not an indicator matrix: "
            "it contains value(s) other than {0,1,np.nan}:",
            illegalvals,
        )

    # check if M and I have same shape
    if not M.shape == I.shape:
        raise ValueError("M and I must have the same shape")

    L = len(M)

    # make a copy of indicator matrix
    Iuse = np.copy(I).astype(float)

    # exclude pixels with M=np.nan from counts towards valid pairs
    if exclude_nans_from_paircounts:
        Iuse[np.isnan(M)] = np.nan

    # compute valid paris, counts and contrast
    set1_pixels = np.zeros(L) * np.nan
    set1_intens = np.zeros(L) * np.nan
    set2_pixels = np.zeros(L) * np.nan
    set2_intens = np.zeros(L) * np.nan

    if verbose:
        print("... ... starting to loop over s")
        report_progres_points = np.linspace(0, L, 11).astype(int)
    for s in range(ignore_diags, L):
        if verbose and s in report_progres_points:
            print("... ... {}% done".format(int(s / L * 100)))
        diagI = np.diag(Iuse, s)  # get diagonal s of indicator matrix
        diagM = np.diag(M, s)  # get diagonal s of data matrix

        set1_pixels[s] = np.nansum(diagI)  # number of valid pixels in set 1
        set2_pixels[s] = np.nansum(1 - diagI)  # number of valid pixels in set 2

        set1_intens[s] = np.nansum(diagM * diagI)
        set2_intens[s] = np.nansum(diagM * (1 - diagI))

    with np.errstate(invalid="ignore"):
        # intensity average of valid pixels in set 1 in diagonal s
        set1_avintens = set1_intens / set1_pixels
        # intensity average of valid pixels in set 2 in diagonal s
        set2_avintens = set2_intens / set2_pixels

        if normalize is True:
            contr = (set1_avintens - set2_avintens) / (set1_avintens + set2_avintens)
        else:
            contr = set1_avintens / set2_avintens

    additional_info = [set1_intens, set1_pixels, set2_intens, set2_pixels]

    return contr, additional_info


# ==============================================================
#  MISCELLANEOUS FUNCTIONS
# ==============================================================


def diagcounts(I, vals=[1, 0]):
    """
    count occurrances of each v in vals in all diagonals of I with offsets s=0..len(I)

    Returns
    -------
    list of 1D arrays of floats, one for each v in vals, each of len=len(I)
        default: count 1s and 0s in each diagonal:
                 return [S1,S0] with:
                 S1: nansum(diag(I,s)==1) for each offset s=0..len(I)
                 S0: nansum(diag(I,s)==0) for each offset s=0..len(I)
    """
    L = len(I)
    diagcounts = []
    for i in range(len(vals)):
        S = np.zeros(L)
        for s in range(L):
            S[s] = np.nansum(np.diag(I, s) == vals[i])
        diagcounts.append(S)
    return diagcounts


def get_types(v, vals=None):
    """
    Find unique and finite values (bin types) in v.
    If vals is not None, restrict to types in vals (floats and ints are
    considered identical).

    Parameters
    ----------
    v: sequence of numbers
        a vector of "types" used to construct indicator matrices

    vals: sequence of numbers, optional
        default: unique and finite values in v

    Returns
    -------
    valid_types: 1d array

    """
    v = np.asarray(v)
    types = np.unique(v[np.isfinite(v)])  # find types (cast to floats)

    if vals is not None:
        vals = np.asarray(vals)
        vals = np.unique(vals[np.isfinite(vals)])  # cast to floats
        valid_types = np.asarray([t for t in types if t in vals and np.isfinite(t)])
    else:
        valid_types = types

    return valid_types


def discretize_track(v, get_bin_identities=lambda x: x > np.nanmean(x)):
    """
    discretizes a (quasi)continuous track (eg. an EV)
    using the supplied lambda function
    by default: split by nanmean

    returns: discretized track, non-finite values are left intact
    """
    v_disc = np.ones(len(v)) * np.nan
    val_inds = np.isfinite(v)  # valid indices
    v_disc[val_inds] = get_bin_identities(v[val_inds])
    return v_disc


def normalize(v):
    # normalization
    v = np.asarray(v)
    return (v - 1) / (v + 1)


def normalize_inv(v):
    # invert normalization
    v = np.asarray(v)
    return (1 + v) / (1 - v)


# ==============================================================
#  GENERAL PURPOSE INDICATOR MATRIX FUNCTION
# ==============================================================


def indicatormat(modality=None, params=None):
    """
    get indicatormatrix I with a given modality by calling
    the function specified by 'modality'

    Parameters
    ----------
    modality: string, optional
        valid modalities are listed below in dict 'valid_modalities'
        if modality is None: print a list of valid modalities and return nothing
    params: list, optional
        have to match the parameters of the called indicator matrix functions

    Returns
    -------
    I: 2d array
        the computed indicator matrix

    """

    # NOTE: when adding/removing a specific indicator function below, also add/remove it here
    valid_modalities = {
        "AnyAny_vs_Mixed": indicatormat_AnyAny_vs_Mixed,  # params: [v]
        "TypeType_vs_Mixed": indicatormat_TypeType_vs_Mixed,  # params: [v, Type]
        "TypeType_vs_TypeOther": indicatormat_TypeType_vs_TypeOther,  # params: [v, Type]
        "TypeType_vs_NontypeNontype": indicatormat_TypeType_vs_NontypeNontype,  # params: [v, Type]
        "TypeType_vs_Rest": indicatormat_TypeType_vs_Rest,  # params: [v, Type]
        "Segments_vs_Rest": indicatormat_Segments_vs_Rest,  # params: [segments, L=None, bad_bins=[]]
    }

    valid_modalities_names = ""
    for k in valid_modalities.keys():
        valid_modalities_names = valid_modalities_names + k + "\n"

    if modality is None:
        print("valid modalities are:\n=====================\n" + valid_modalities_names)
        return

    if modality not in valid_modalities.keys():
        raise ValueError(
            modality
            + " is not a valid modality. Valid modalities are: \n"
            + valid_modalities_names
        )

    I = valid_modalities[modality](*params)

    return I


# ==============================================================
#  INDICATOR MATRIX FUNCTIONS
#         FOR SPECIFIC COMPARTMENTALIZATION MODALITIES
# ==============================================================

# NOTE: when adding/removing a specific indicator function,
#       also add/remove it to 'valid_modalities' in 'indicatormat()'


def indicatormat_AnyAny_vs_Mixed(v):
    """
    Parameters
    ----------
    v: sequence of numbers
        a vector of "types" used to construct indicator matrices
        types are the unique and finite values in v

    Returns
    -------
    I: 2d array
        indicator matrix with the following properties
        I[i,j]=1    if v[i]==v[j],               i.e. thistype-to-thistype for any type
        I[i,j]=nan  if v[i]==nan or v[j]==nan,   i.e. excluding invalid bins
        I[i,j]=0    otherwise,                   i.e. thistype-to-othertype for valid types

    """
    v = np.asarray(v)
    L = v.size

    types = get_types(v)

    # put nans in rows and cols with type==nan, zero elsewhere
    v_aux = np.zeros(L) * np.nan
    v_aux[~np.isnan(v)] = 0
    I = np.outer(v_aux, v_aux)  # => I[i,j]=nan if v[i]==nan or v[j]==nan

    # put 1 in I[i,j] if type[v[i]]==type[v[j]]
    for thistype in types:
        v_aux = np.zeros(L) * np.nan
        v_aux[v == thistype] = 1
        I_thistype = np.outer(
            v_aux, v_aux
        )  # => I_thistype: 1 for (i,j) both of this type, nan elsewhere
        I[
            I_thistype == 1
        ] = 1  # => I_anytype: zero for across types, one for withing any type, nan elsewhere

    return I


def indicatormat_TypeType_vs_Mixed(v, Type):
    """
    Parameters
    ----------
    v: sequence of numbers
        a vector of "types" used to construct indicator matrices

    Type: number
        type for which to compute indicator matrix with properties

    Returns
    -------
    I: 2d array:
        indicator matrix with the following properties
        I[i,j]=1    if v[i]==v[j]==val,    i.e. thistype-to-thistype
        I[i,j]=0    if v[i]~=v[j]==val,    i.e. sometype-to-othertype
        I[i,j]=nan  otherwise              i.e. othertype-to-othertype and invalid bins

    """
    v = np.asarray(v)
    L = v.size

    if not np.any(v == Type):
        warnings.warn("Type not found in v")

    I_AnyAny_vs_Mixed = indicatormat_AnyAny_vs_Mixed(
        v
    )  # get auxiliury matrix AnyAny_vs_AnyOther

    v_aux = np.zeros(L) * np.nan
    v_aux[v == Type] = 1
    I = np.outer(v_aux, v_aux)  # 1 for (i,j) both of this type, nan elsewhere
    I[I_AnyAny_vs_Mixed == 0] = 0  # 0 for (i,j) of different types

    return I


def indicatormat_TypeType_vs_TypeOther(v, Type):
    """
    Parameters
    -----------
    v: sequence of numbers
        a vector of "types" used to construct indicator matrices

    Type: number
        type for which to compute indicator matrix with properties

    Returns
    -------
    I: 2d array:
        indicator matrix with the following properties
        I[i,j]=1    if v[i]==v[j]==val,         i.e. thistype-to-thistype
        I[i,j]=0    if v[i]==val xor v[j]==val  i.e. thistype-to-othertype
        I[i,j]=nan  otherwise                   i.e. othertype-to-othertype and invalid bins

    """
    v = np.asarray(v)
    L = v.size

    if not np.any(v == Type):
        warnings.warn("Type not found in v")

    if 0:
        # get auxiliury matrix AnyAny_vs_AnyOther
        I_AnyAny_vs_Mixed = indicatormat_AnyAny_vs_Mixed(v)[0]

        v_aux = np.zeros(L) * np.nan
        v_aux[v == Type] = 1
        I = np.outer(v_aux, v_aux)  # 1 for (i,j) both of this type, nan elsewhere
        for i, j in np.ndindex((L, L)):
            if (
                ((v[i] == Type) != (v[j] == Type))
                and np.isfinite(v[i])
                and np.isfinite(v[j])
            ):
                I[i, j] = 0

    if 1:
        # put nans in rows and cols with type==nan, zero elsewhere
        v_aux = np.zeros(L) * np.nan
        v_aux[~np.isnan(v)] = 0
        I_invalid = np.outer(v_aux, v_aux)  # => I[i,j]=nan if v[i]==nan or v[j]==nan

        # put zeros in stripes that contain Type
        v_aux = np.ones(L)
        v_aux[v == Type] = 0
        I_striped = np.outer(v_aux, v_aux)  # 0 if v[i]==Type or v[j]==Type, 1 elsewhere

        # put ones in rectangles that are both Type
        v_aux = np.zeros(L) * np.nan
        v_aux[v == Type] = 1
        I_TypeType = np.outer(v_aux, v_aux)  # 1 if v[i]==v[j]==Type, nan elsewhere

        I = np.zeros((L, L)) * np.nan
        I[I_striped == 0] = 0
        I[I_TypeType == 1] = 1
        I[np.isnan(I_invalid)] = np.nan

    return I


def indicatormat_TypeType_vs_NontypeNontype(v, Type):
    """
    Parameters
    -----------
    v: sequence of numbers
        a vector of "types" used to construct indicator matrices

    Type: number
        type for which to compute indicator matrix with properties

    Returns
    -------
    I: 2d array:
        indicator matrix with the following properties
        I[i,j]=1    if v[i]==v[j]==val,         i.e. thistype-to-thistype
        I[i,j]=0    if v[i]!=val and v[j]!=val  i.e. anything not involving type
        I[i,j]=nan  otherwise                   i.e. type-to-othertype and invalid bins

    """
    v = np.asarray(v)
    L = v.size

    if not np.any(v == Type):
        warnings.warn("Type not found in v")

    # get auxiliury matrix AnyAny_vs_AnyOther
    I_AnyAny_vs_Mixed = indicatormat_AnyAny_vs_Mixed(v)[0]

    v_aux = np.zeros(L) * np.nan
    v_aux[v == Type] = 1
    I = np.outer(v_aux, v_aux)  # 1 for (i,j) both of this type, nan elsewhere

    v_aux = np.zeros(L) * np.nan
    v_aux[(v != Type) * np.isfinite(v)] = 1
    I_aux = np.outer(v_aux, v_aux)
    I[I_aux == 1] = 0

    return I


def indicatormat_TypeType_vs_Rest(v, Type):
    """
    Parameters
    -----------
    v: sequence of numbers
        a vector of "types" used to construct indicator matrices

    Type: number
        type for which to compute indicator matrix with properties

    Returns
    -------
    I: 2d array
        indicator matrix with the following properties
        I[i,j]=1    if v[i]==v[j]==val,          i.e. thistype-to-thistype
        I[i,j]=nan  if v[i]==nan or v[j]==nan,   i.e. excluding invalid bins
        I[i,j]=0    otherwise,                   i.e. thistype-to-othertype and anytype-to-anyother

    """

    v = np.asarray(v)
    L = v.size

    if not np.any(v == Type):
        warnings.warn("Type not found in v")

    # get auxiliury matrix AnyAny_vs_AnyOther
    I_AnyAny_vs_Mixed = indicatormat_AnyAny_vs_Mixed(v)[0]

    # construc indicator matrices
    v_aux = np.zeros(L) * np.nan
    v_aux[np.isfinite(v)] = 0
    v_aux[v == Type] = 1
    I = np.outer(v_aux, v_aux)  # 1 for (i,j) both of this type, nan elsewhere

    return I


def indicatormat_Segments_vs_Rest(segments, L=None, bad_bins=[]):
    """
    Parameters
    ----------
    segments: list of tuples
        [(s0,e0),(s1,e1), ...]
        where (s,e) are the start and end points of segments
        (endpoints are exclusive)
        Note: endpoints are trimmed to L

    L: integer, optional
        default: last endpoint

    bad_bins: list of integers, optional
        bad bins, rows and comuns get np.nan throughout

    Returns
    -------
    I: 2d array
        indicator matrices with the following properties
        I[i,j]=1    if i and j within same segment
        I[i,j]=nan  if i in bad_bins or j in bad_bins
        I[i,j]=0    otherwise

    """
    if L is None:
        L = segments[-1][1]

    v_aux = np.zeros(L)
    if bad_bins != []:
        v_aux[np.asarray(bad_bins)] = np.nan
    I = np.outer(v_aux, v_aux)

    segments_trimmed = []
    for s, e in segments:
        e = min(e, L)
        segments_trimmed.append((s, e))
        I[s:e, s:e] = I[s:e, s:e] + 1
    I[I > 1] = 1

    return I
