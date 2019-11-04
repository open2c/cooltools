import os.path as op
import subprocess
import sys

import numpy as np
import pandas as pd


def test_compartment_cli(request, tmpdir):
    in_cool = op.join(request.fspath.dirname, 'data/sin_eigs_mat.cool')
    out_eig_prefix = op.join(tmpdir, 'test.eigs')
    try:
        result = subprocess.check_output(
            f'python -m cooltools call-compartments -o {out_eig_prefix} {in_cool}',
            shell=True
            ).decode('ascii')
    except subprocess.CalledProcessError as e:
        print(e.output)
        print(sys.exc_info())
        raise e
    test_eigs = pd.read_table(out_eig_prefix+'.cis.vecs.tsv', sep='\t')
    gb = test_eigs.groupby('chrom')
    for chrom in gb.groups:
        chrom_eigs = gb.get_group(chrom)
        r = np.abs(np.corrcoef(chrom_eigs.E1.values,
                               np.sin(chrom_eigs.start * 2 * np.pi / 500))[0,1])
        assert r>0.95


def test_saddle_cli(request, tmpdir):
    in_cool = op.join(request.fspath.dirname, 'data/sin_eigs_mat.cool')
    out_eig_prefix = op.join(tmpdir, 'test.eigs')
    out_expected = op.join(tmpdir, 'test.expected')
    out_saddle_prefix = op.join(tmpdir, 'test.saddle')

    try:
        result = subprocess.check_output(
            f'python -m cooltools call-compartments -o {out_eig_prefix} {in_cool}',
            shell=True
            ).decode('ascii')
    except subprocess.CalledProcessError as e:
        print(e.output)
        print(sys.exc_info())
        raise e

    try:
        result = subprocess.check_output(
            f'python -m cooltools compute-expected {in_cool} > {out_expected}',
            shell=True
            ).decode('ascii')
    except subprocess.CalledProcessError as e:
        print(e.output)
        print(sys.exc_info())
        raise e

    try:
        result = subprocess.check_output(
            f'python -m cooltools compute-saddle -o {out_saddle_prefix} --range -0.5 0.5 '
            +f'--n-bins 30 --scale log {in_cool} {out_eig_prefix}.cis.vecs.tsv {out_expected}',
            shell=True
        ).decode('ascii')
    except subprocess.CalledProcessError as e:
        print(e.output)
        print(sys.exc_info())
        raise e

    log2_sad = np.log2(np.load(out_saddle_prefix + '.saddledump.npz')['saddledata'])
    bins = np.load(out_saddle_prefix + '.saddledump.npz')['binedges']
    binmids = (bins[:-1] + bins[1:]) / 2
    log2_theor_sad = np.log2(1 + binmids[None,:] * binmids[:,None])

    log2_sad_flat = log2_sad[1:-1, 1:-1].flatten()
    log2_theor_sad_flat = log2_theor_sad.flatten()

    mask = np.isfinite(log2_sad_flat) & np.isfinite(log2_theor_sad_flat)

    cc = np.abs(np.corrcoef(log2_sad_flat[mask], log2_theor_sad_flat[mask])[0][1])

    assert cc > 0.9


# def test_digitize_track(request):
#     pass


# def test_make_saddle(request):
#     pass


# def test_saddleplot(request):
#     pass


# def test_saddlestrength(request):
#     pass
