import sys
import os
import logging
import mindspore_lite as mslite
import numpy as np


def set_mslite_context():
    # init context, and set target is ascend.
    context = mslite.Context()
    context.target = ["Ascend"]
    context.ascend.device_id = 0
    context.ascend.provider = "ge"
    return context


def build_model(akg_model_path, cfg_file_name):
    context = set_mslite_context()
    model_inc = mslite.Model()
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir)
    model_inc.build_from_file(
        akg_model_path, mslite.ModelType.MINDIR, context, config_path=config_path
    )
    return model_inc


def benchmark(akg_model: mslite.Model, inputs):
    outputs_akg: mslite.Tensor = akg_model.predict(inputs)
    np_output = [tensor.get_data_to_numpy() for tensor in outputs_akg]
    return np_output


def np_net(x, y0, y1):
    a = x @ y0
    b = x @ y1
    c = a * b
    return [c]


def test_net(model_dir_path, cfg_file_name):
    model_path = os.path.join(model_dir_path, "lite_akg", "matmul_nz.mindir")
    models = build_model(model_path, cfg_file_name)
    logging.info("build_end")
    A = np.random.random((16, 512)).astype(np.float16) + 0.01
    B0 = np.random.random((512, 64)).astype(np.float16) + 0.01
    B1 = np.random.random((512, 64)).astype(np.float16) + 0.01
    a = mslite.Tensor(A, A.shape, dtype=mslite.DataType.FLOAT16)
    b0 = mslite.Tensor(B0, B0.shape, dtype=mslite.DataType.FLOAT16)
    b1 = mslite.Tensor(B1, B1.shape, dtype=mslite.DataType.FLOAT16)
    out_akg = benchmark(models, [a, b0, b1])
    out_np = np_net(A, B0, B1)

    for i in range(len(out_akg)):
        is_close = np.allclose(out_np[i], out_akg[i], rtol=2e-3, atol=2e-3)
        logging.info("ref_outputs %d:\n%s", i, out_np[i])
        logging.info("ascend_outputs %d:\n%s", i, out_akg)
        logging.info("ascend output %d is equal to ref output: %s", i, is_close)
        assert is_close

    logging.info("benmchmark_end")


if __name__ == "__main__":
    model_dir = sys.argv[1]
    backend = sys.argv[2]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
        filename="./test.log",
        filemode="w",
    )
    test_net(model_dir, "akg_matmul_nz.cfg")
    test_net(model_dir, "akg_matmul_nd.cfg")
