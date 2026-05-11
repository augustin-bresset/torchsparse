import json
import os
import struct
import tempfile

import numpy as np
import torch

import torchsparse
from torchsparse import SparseTensor
from torchsparse import nn as spnn

__all__ = ['test_pts_format', 'test_pts_inference']

_MAGIC = b'\x93TSPTS'
_VERSION_MAJOR = 1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_sparse_tensor(num_points=512, channels=16, stride=1,
                        spatial_range=None, dtype=torch.float32,
                        device='cuda:0'):
    torch.manual_seed(42)
    coords = torch.stack([
        torch.randint(0, 2, (num_points,)),   # batch index
        torch.randint(0, 32, (num_points,)),  # x
        torch.randint(0, 32, (num_points,)),  # y
        torch.randint(0, 32, (num_points,)),  # z
    ], dim=1).int().to(device)
    feats = torch.randn(num_points, channels, dtype=dtype).to(device)
    return SparseTensor(feats=feats, coords=coords,
                        stride=stride, spatial_range=spatial_range)


def _inspect_pts_file(path):
    """Return (header_dict, data_offset) by parsing the raw binary file."""
    with open(path, 'rb') as f:
        magic = f.read(len(_MAGIC))
        version = f.read(2)
        (hlen,) = struct.unpack('<I', f.read(4))
        header = json.loads(f.read(hlen).decode('utf-8'))
        data_offset = len(_MAGIC) + 2 + 4 + hlen
    return magic, version, header, data_offset


# ---------------------------------------------------------------------------
# test functions (called by unittest cases)
# ---------------------------------------------------------------------------

def test_pts_format(device='cuda:0'):
    """
    Vérifie la structure binaire du fichier .pts :
      - magic bytes corrects
      - version cohérente
      - header JSON complet et cohérent avec les tenseurs
      - données alignées sur 64 octets
      - round-trip exact pour coords, feats, stride et spatial_range
    Teste aussi les variantes de dtype et l'API st.save().
    Retourne True si tout est correct.
    """
    results = {}

    with tempfile.TemporaryDirectory() as tmpdir:

        # --- cas principal : float32, stride asymétrique, spatial_range défini ---
        path = os.path.join(tmpdir, 'test.pts')
        st = _make_sparse_tensor(num_points=1000, channels=32, stride=(1, 2, 4),
                                  spatial_range=(2, 32, 32, 32), dtype=torch.float32,
                                  device=device)
        torchsparse.save(st, path)

        # 1. inspection binaire
        magic, version, header, data_offset = _inspect_pts_file(path)
        assert magic == _MAGIC, f'magic incorrect : {magic!r}'
        assert version[0] == _VERSION_MAJOR, f'version majeure incorrecte : {version[0]}'
        assert data_offset % 64 == 0, f'données non alignées sur 64 octets (offset={data_offset})'

        # 2. cohérence du header
        assert header['stride'] == [1, 2, 4], f"stride : {header['stride']}"
        assert header['spatial_range'] == [2, 32, 32, 32], f"spatial_range : {header['spatial_range']}"
        assert header['coord_shape'] == [1000, 4], f"coord_shape : {header['coord_shape']}"
        assert header['feat_shape'] == [1000, 32], f"feat_shape : {header['feat_shape']}"
        assert header['coord_dtype'] == '<i4', f"coord_dtype : {header['coord_dtype']}"
        assert header['feat_dtype'] == '<f4', f"feat_dtype : {header['feat_dtype']}"

        # 3. taille fichier cohérente avec les données annoncées
        coord_bytes = int(np.prod(header['coord_shape'])) * np.dtype(header['coord_dtype']).itemsize
        feat_bytes  = int(np.prod(header['feat_shape']))  * np.dtype(header['feat_dtype']).itemsize
        expected_size = data_offset + coord_bytes + feat_bytes
        actual_size = os.path.getsize(path)
        assert actual_size == expected_size, \
            f'taille fichier {actual_size} != attendue {expected_size}'

        # 4. round-trip exact
        st2 = torchsparse.load(path, device=device)
        assert torch.equal(st.coords.cpu(), st2.coords.cpu()), 'coords ne correspondent pas'
        assert torch.equal(st.feats.cpu(),  st2.feats.cpu()),  'feats ne correspondent pas'
        assert st.stride == st2.stride,                         f'stride : {st.stride} vs {st2.stride}'
        assert st.spatial_range == st2.spatial_range,           f'spatial_range : {st.spatial_range} vs {st2.spatial_range}'
        results['float32_roundtrip'] = True

        # --- variante float16 ---
        path_half = os.path.join(tmpdir, 'test_half.pts')
        st_half = _make_sparse_tensor(channels=8, dtype=torch.float16, device=device)
        st_half.save(path_half)   # test de l'API méthode
        _, _, hdr_half, _ = _inspect_pts_file(path_half)
        assert hdr_half['feat_dtype'] == '<f2', f"feat_dtype half : {hdr_half['feat_dtype']}"
        st_half2 = torchsparse.load(path_half, device=device)
        assert torch.equal(st_half.feats.cpu(), st_half2.feats.cpu()), 'feats float16 ne correspondent pas'
        results['float16_roundtrip'] = True

        # --- spatial_range=None ---
        path_norange = os.path.join(tmpdir, 'test_norange.pts')
        st_norange = _make_sparse_tensor(spatial_range=None, device=device)
        torchsparse.save(st_norange, path_norange)
        _, _, hdr_norange, _ = _inspect_pts_file(path_norange)
        assert hdr_norange['spatial_range'] is None
        st_norange2 = torchsparse.load(path_norange, device=device)
        assert st_norange2.spatial_range is None
        results['no_spatial_range'] = True

        # --- erreur sur fichier invalide ---
        bad_path = os.path.join(tmpdir, 'bad.pts')
        with open(bad_path, 'wb') as f:
            f.write(b'NOTAPTS\x00' * 4)
        try:
            torchsparse.load(bad_path)
            assert False, 'aurait dû lever ValueError'
        except ValueError:
            results['bad_magic_raises'] = True

    # tmpdir supprimé automatiquement à la sortie du context manager
    return results


def test_pts_inference(device='cuda:0'):
    """
    Vérifie qu'un SparseTensor sauvegardé puis rechargé produit
    des sorties identiques lors d'un forward pass sur une couche de convolution.
    Retourne l'écart absolu maximal (doit être 0.0 — tenseurs bit-à-bit identiques).
    """
    torch.manual_seed(0)
    num_points = 256
    in_channels = 8
    out_channels = 16

    st_orig = _make_sparse_tensor(num_points=num_points, channels=in_channels,
                                   dtype=torch.float32, device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'inference.pts')
        torchsparse.save(st_orig, path)
        st_loaded = torchsparse.load(path, device=device)
    # tmpdir et fichier supprimés ici

    conv = spnn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1).to(device)
    conv.eval()

    with torch.no_grad():
        out_orig   = conv(st_orig).feats
        out_loaded = conv(st_loaded).feats

    max_adiff = (out_orig - out_loaded).abs().max().item()
    return max_adiff
