import unittest
from torchsparse.nn import functional as F
from python import (
    test_single_layer_convolution_forward,
    test_to_dense_forward,
    test_pts_format,
    test_pts_inference,
    test_torchscript_basics,
    test_torchscript_save_load,
)


class SparseConvTestCase(unittest.TestCase):
    def test_single_layer(self):
        kernel_sizes = [2, 3, 5]
        strides = [1, 2, 3]
        acc_adiff = 0.0
        acc_rdiff = 0.0
        count = 0

        # hashmap mode by default
        for kernel_size in kernel_sizes:
            for stride in strides:
                mean_adiff, max_rdiff = test_single_layer_convolution_forward(
                    kernel_size=kernel_size, stride=stride
                )
                acc_adiff += mean_adiff
                acc_rdiff += max_rdiff
                count += 1

        # switch to hashmap_on_the_fly
        config = F.conv_config.get_default_conv_config()
        config.kmap_mode = "hashmap_on_the_fly"
        F.conv_config.set_global_conv_config(config)
        for kernel_size in kernel_sizes:
            for stride in strides:
                mean_adiff, max_rdiff = test_single_layer_convolution_forward(
                    kernel_size=kernel_size, stride=stride
                )
                acc_adiff += mean_adiff
                acc_rdiff += max_rdiff
                count += 1

        self.assertLessEqual(acc_adiff / count, 1e-4)
        self.assertLessEqual(acc_rdiff / count, 1e-2)


class ToDenseTestCase(unittest.TestCase):
    def test_to_dense(self):
        max_adiff = test_to_dense_forward()
        self.assertLessEqual(max_adiff, 1e-5)


class TorchScriptTestCase(unittest.TestCase):
    def test_basics(self):
        results = test_torchscript_basics(device="cuda:0")
        for check, ok in results.items():
            self.assertTrue(ok, f"check échoué : {check}")

    def test_save_load(self):
        max_adiff = test_torchscript_save_load(device="cuda:0")
        self.assertEqual(max_adiff, 0.0,
                         f"save/load mismatch : {max_adiff}")


class SparseTensorIOTestCase(unittest.TestCase):
    def test_pts_format(self):
        results = test_pts_format(device='cuda:0')
        for check, ok in results.items():
            self.assertTrue(ok, f'check échoué : {check}')

    def test_pts_inference(self):
        max_adiff = test_pts_inference(device='cuda:0')
        self.assertEqual(max_adiff, 0.0,
                         f'écart après round-trip : {max_adiff} (attendu 0.0)')


if __name__ == "__main__":
    unittest.main()
