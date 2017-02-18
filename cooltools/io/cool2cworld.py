import os
import gzip
import tarfile
import tempfile

import pyximport
pyximport.install()
import fastsavetxt

import cooler

def cool2cworld(
    cooler_path, 
    out_path, 
    iced=False, 
    iced_unity=False):
    '''
    Dump a genome-wide contact matrix from cooler into a CWorld-format 
    text matrix.

    Parameters
    ----------
    cooler_path : str
        The path to the cooler file.

    out_path : str
        The path to the output matrix file. 
        If ends with .gz, the output is gzipped.

    iced : bool, optional
        If True, dump the balanced matrix.

    iced_unity : bool, optional
        If True and `iced` is True, dump the matrix balanced to a unity.
    '''

    c = cooler.Cooler(cooler_path)
    mat = c.matrix(balance=iced)[:]
    if iced and (not iced_unity):
        mat *= (c._load_attrs('/bins/weight')['scale']) ** 2

    res = c.info['bin-size']
    nbins = c.info['nbins']
    gname = c.info['genome-assembly']
        
    col_headers = '\t'.join(
        ['{}x{}'.format(nbins, nbins)] +
        ['{}|{}|{}:{}-{}'.format(binidx, gname, b.chrom, b.start+1, b.end)
         for binidx, b in c.bins()[:].iterrows()
        ]
    ).encode()

    row_headers = [
        '{}|{}|{}:{}-{}'.format(binidx1, gname, b1.chrom, b1.start+1, b1.end).encode()
        for binidx1, b1 in c.bins()[:].iterrows()
    ]
    
    fastsavetxt.array2txt(
        mat,
        out_path,
        format_string = b'%.4lf', #b'%.4lf' if iced else b'%.4lf',
        header=col_headers,
        row_headers = row_headers,
    )



def cool2cworldTar(
    cooler_paths,
    target_folder,
    dataset_name
    ):
    '''
    Makes a CWorld .tar archive with binned contact maps at multiple resolutions
    in .matrix.txt.gz format. 

    Parameters
    ----------
    cooler_paths : a list of str
        The paths to all coolers to dump into a single CWorld tar archive.
        Must correspond to the same dataset and have different resolutions.

    target_folder : str
        The folder to contain the output .tar archive.

    dataset_name : str
        The name of the dataset.

    '''
    
    if not os.path.isdir(target_folder):
        raise Exception('The target folder must exist: {}'.format(target_folder))
    
    
    with tempfile.TemporaryDirectory() as cworld_tmp_path:
        for cooler_path in cooler_paths:
            res = cooler.Cooler(cooler_path).info['bin-size']
            os.mkdir(os.path.join(cworld_tmp_path, 'C-'+str(res)))
            for iced, iced_label in [(True,'iced'), (False, 'raw')]:
                folder_path = os.path.join(cworld_tmp_path, 'C-'+str(res), iced_label)
                os.mkdir(folder_path)

                mat_path = os.path.join(
                    folder_path, 
                    '{}__C-{}-{}.matrix.gz'.format(dataset_name, res, iced_label))

                cool2cworld(
                    cooler_path=cooler_path,
                    out_path=mat_path,
                    iced=iced
                    )
                
        cworld_full_name = dataset_name+'__txt'        
        with tarfile.open(
            os.path.join(target_folder, cworld_full_name) + '.tar', mode='w') as archive:
            archive.add(
                cworld_tmp_path,
                arcname=cworld_full_name, 
                recursive=True)
