API Reference
=============

.. contents:: Contents
   :depth: 2
   :local:

----

SparseTensor
------------

.. autoclass:: torchsparse.SparseTensor
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: torchsparse.utils.to_dense

----

torchsparse.nn -- Layers
-----------------------

Convolution
~~~~~~~~~~~

.. autoclass:: torchsparse.nn.Conv3d
   :members: __init__, forward
   :show-inheritance:

Normalisation
~~~~~~~~~~~~~

.. autoclass:: torchsparse.nn.BatchNorm
   :members: forward
   :show-inheritance:

.. autoclass:: torchsparse.nn.GroupNorm
   :members: forward
   :show-inheritance:

.. autoclass:: torchsparse.nn.InstanceNorm
   :members: forward
   :show-inheritance:

Pooling
~~~~~~~

.. autoclass:: torchsparse.nn.modules.pooling.GlobalAvgPool
   :members: forward
   :show-inheritance:

Activation
~~~~~~~~~~

.. autoclass:: torchsparse.nn.modules.activation.ReLU
   :members: forward
   :show-inheritance:

.. autoclass:: torchsparse.nn.modules.activation.LeakyReLU
   :members: forward
   :show-inheritance:

----

torchsparse.nn.functional
-------------------------

.. automodule:: torchsparse.nn.functional
   :members:
   :undoc-members:

----

torchsparse.script -- TorchScript export
----------------------------------------

Use these when you need to export a sparse model with
:func:`torch.jit.script` / :func:`torch.jit.save`.

.. autofunction:: torchsparse.script.make_scriptable

.. autoclass:: torchsparse.script.ScriptableConv3d
   :members: __init__, forward
   :show-inheritance:

Example
~~~~~~~

.. code-block:: python

   import torch
   from torchsparse.script import ScriptableConv3d, make_scriptable

   coords = torch.randint(0, 16, (256, 4), dtype=torch.int32).cuda()
   coords[:, 0] = 0
   feats  = torch.randn(256, 16).cuda()
   st = make_scriptable(feats, coords, stride=[1, 1, 1])

   class Net(torch.nn.Module):
       def __init__(self):
           super().__init__()
           self.conv = ScriptableConv3d(16, 32, kernel_size=3, stride=1)

       def forward(self, x: "torch.classes.torchsparse.SparseTensorImpl"):
           return self.conv(x)

   scripted = torch.jit.script(Net().cuda().eval())
   torch.jit.save(scripted, "model.pt")

   loaded = torch.jit.load("model.pt")
   with torch.no_grad():
       out = loaded(st)
   print(out.feats().shape)   # [256, 32]

----

torchsparse.io -- Binary I/O
----------------------------

Save and load :class:`~torchsparse.SparseTensor` objects in the compact
`.pts` binary format (see ``docs/pts_format.md`` for the format specification).

.. autofunction:: torchsparse.io.save

.. autofunction:: torchsparse.io.load

Example
~~~~~~~

.. code-block:: python

   import torch
   from torchsparse import SparseTensor
   from torchsparse.io import save, load

   coords = torch.randint(0, 32, (1000, 4), dtype=torch.int32).cuda()
   coords[:, 0] = 0
   feats  = torch.randn(1000, 64).cuda()
   st = SparseTensor(feats=feats, coords=coords)

   save(st, "pointcloud.pts")
   st2 = load("pointcloud.pts", device="cuda:0")
   print(torch.allclose(st.feats, st2.feats))   # True

----

torchsparse.backends -- Runtime options
----------------------------------------

Global runtime flags that control precision and performance.

.. automodule:: torchsparse.backends
   :members:
   :undoc-members:

Flags
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Flag
     - Default
     - Description
   * - ``benchmark``
     - ``False``
     - Enable autotuning of sparse convolution kernels (similar to
       ``torch.backends.cudnn.benchmark``).
   * - ``allow_tf32``
     - ``True`` on Ampere+
     - Allow TF32 matrix multiplications (GPU capability ≥ 8.0).
   * - ``allow_fp16``
     - ``True`` on Turing+
     - Allow FP16 kernels (GPU capability ≥ 7.5).
   * - ``hash_rsv_ratio``
     - ``2``
     - Multiplier for the GPU hash-table capacity relative to the number of
       input points. Increase if you observe hash-table overflow errors.
