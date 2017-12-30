import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import copy





def get_kernel(w,p,ktype='donut',show=True):
    """
    w is the outer boundary
    p id the inner one
    """
    width = 2*w+1
    kernel = np.ones((width,width),dtype=np.int)
    # mesh grid:
    y,x = np.ogrid[-w:w+1, -w:w+1]

    if ktype == 'donut':
        # mask inner pXp square:
        mask = (((-p<=x)&(x<=p))&
                ((-p<=y)&(y<=p)) )
        # mask vertical and horizontal
        # lines of width 1 pixel:
        mask += (x==0)|(y==0)
        # they are all 0:
        kernel[mask] = 0
    else:
        print("Only 'donut' kernel has been"
            "determined so far.")
        raise
    # 
    #
    if show:
        # and plot one as well:
        extent = (0-0.5,
                  x.size-0.5,
                  0-0.5,
                  y.size-0.5)
        ########################
        plt.clf()
        # axes setup
        ax = plt.gca()
        # discrete colormap with 2 colors ...
        cmap = colors.ListedColormap([plt.cm.viridis(0),
                                      plt.cm.viridis(1.0)])
        # kernel:
        imk = ax.imshow(kernel,
                        alpha=0.7,
                        cmap=cmap,
                        extent=extent,
                        interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels('',visible=False)
        ax.set_yticklabels('',visible=False)
        ax.set_title("{} kernel".format(ktype),fontsize=16)

        # add checkerboard to depict pixels:
        checkerboard = np.add.outer(range(x.size),
                                    range(y.size)) % 2
        ax.imshow(checkerboard,
                cmap=plt.cm.gray,
                interpolation='nearest',
                extent=extent,
                alpha=0.3)
        #####################
        # colorbar and that's it:
        cb = plt.colorbar(imk)
        cb.ax.get_yaxis().set_major_locator(ticker.MaxNLocator(1))
        cb.ax.set_yticklabels(['False','True'])
    # 
    # 
    return kernel





def show_heatmap_wfeat(M,
                       h_range,
                       v_range,
                       feature_df=None,
                       b=1,
                       show_bad=True,
                       figsize=(22,22),
                       ax=None,
                       bad_color='white',
                       vmin=None,
                       vmax=None,
                       heat_type='log10'):
    """
    Function the takes a heatmap matrix 
    and plots it within the h_range and v_range.

    feature_df stores a list of pixels
    to be highlighted on top of the 
    heatmap. It is a DataFrame with row
    and col columns.

    b is a bin-size.
    Use b=1 to display corrdinates as X,Y ticks.
    Use b=xxxx bases to display real 
    genomic coordinates as X,Y ticks.

    figsize = (w,h) = (22,22)
    """

    def show_features(feats,ax):
        # stick with out format of features
        # at first.
        # 
        # transition to BEDPE-like stuff soon
        #
        # horizontally within the range ...
        cdh = np.logical_and((feature_df['col'] > h_start),
                             (feature_df['col'] < h_stop))
        # vertically within the range ...
        rdv = np.logical_and((feature_df['row'] > v_start),
                             (feature_df['row'] < v_stop))
        # features within 
        features_in_view = copy(feature_df[cdh & rdv])
        if 'color' not in features_in_view.columns:
            features_in_view['color'] = 'blue'
        if 'marker' not in features_in_view.columns:
            features_in_view['marker'] = 's'
        if 'marker_size' not in features_in_view.columns:
            features_in_view['marker_size'] = 26

        # Put features on top of the heatmap ...
        # that can be rewritten later ...
        for c,r,color,marker,s in features_in_view[['row',
                                                    'col',
                                                    'color',
                                                    'marker',
                                                    'marker_size']].itertuples(index=False):
            # no need to subtract start
            # after you specify extent ...
            ax.scatter(c*b,
                       r*b,
                       color=color,
                       marker=marker,
                       s=s,
                       alpha=0.99)

        return


    def millify(n,pos):
        n = float(n)
        return '{:.1f}{}'.format((n/1e+6),'M')

    # plot a piece of Hi-C heatmap
    # with some features (dots ...)
    
    # unpack ranges:
    h_start, h_stop = h_range
    v_start, v_stop = v_range
    
    # extent
    xmin = h_start*b
    xmax = h_stop*b
    ymin = v_start*b
    ymax = v_stop*b
    # the extent:
    extent=[xmin, xmax, ymax, ymin]
    
    if ax is None:
        # plot heatmap at the base layer of everyhting ...
        plt.clf()
        # figure setup
        f = plt.gcf()
        f.set_size_inches(figsize)
        # axes setup
        ax = plt.gca()
    # this is not working properly for some
    # reason ...
    if show_bad:
        hic_cmap = copy(matplotlib.cm.YlOrRd)
        hic_cmap.set_bad(color=bad_color)
    else:
        hic_cmap = 'YlOrRd'
    #
    #
    im = ax.imshow(M[slice(*h_range),
                     slice(*v_range)],
                  interpolation='nearest',
                  extent = extent,
                  cmap=hic_cmap,
                  vmin = vmin,
                  vmax = vmax)
    # just print the actual range used
    # by imshow to plot heatmap ...
    vmin, vmax = im.get_clim()
    print("vmin,vmax:")
    print(vmin,vmax)

    ############################
    # COLOBAR
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(
        {'linear': 'relative contact frequency',
         'log2'  : 'log 2 ( relative contact frequency )',
         'log10' : 'log 10 ( relative contact frequency )'}[heat_type],
        fontsize=18)
    cb.ax.get_yaxis().set_major_locator(ticker.MaxNLocator(4))
    cb.ax.tick_params(labelsize=16) 

    ################################
    # deal with b and extent etc.
    ################################    
    if b == 1:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(12))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(12))
    else:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(millify))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(millify))
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(20) 
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(20) 
            tick.label.set_rotation('vertical')
            tick.label1.set_verticalalignment('center')

    ###############################################
    # deal with features ...
    ###############################################
    if feature_df is not None:
        show_features(feature_df,ax)
    
    return ax





def show_heatmap_pixel(M,col,row,win,ax=None,vmin=None,vmax=None,ax_visible=False):
    """
    Plot vicinity of the pixel (col,row)
    with the width 2*win+1.
    vmin,vmax - are optional.
    returns ax, just in case.
    """
    assert 0<col<M.shape[0]
    assert 0<row<M.shape[0]
    
    hr = slice(col-win,col+win+1)
    vr = slice(row-win,row+win+1)

    # if no axes has been provided:
    if ax is None:
        plt.clf()
        f = plt.gcf()
        ax = plt.gca()

    hic_cmap = 'YlOrRd'
    im = ax.imshow(M[hr,vr],
              interpolation='nearest',
              cmap=hic_cmap,
              vmin=vmin,
              vmax=vmax)

    ax.xaxis.set_visible(ax_visible)
    ax.yaxis.set_visible(ax_visible)

    
    return ax





def combine_matrix_tris(Mup,Mlo):
    """
    Take upper triangle of Mup,
    lower triangle of Mlo and 
    return the "chimera" matrix.
    """
    # make sure matrices are the 
    # same size:
    assert Mup.shape == Mlo.shape
    #
    triu = np.triu_indices(Mup.shape[0])
    tril = np.tril_indices(Mup.shape[0])
    #
    M = np.empty_like(Mup)
    M[triu] = Mup[triu]
    M[tril] = Mlo[tril]
    return M





##############
# let it be
##############

def main():
    pass

if __name__ == '__main__':
    main()








