from te import tik
from topi.cce import util
from mindspore.ops.op_info_register import op_info_register
@op_info_register("""{
    "op_name": "CusFusedAbsMax1",
    "imply_type": "TBE",
    "fusion_type": "OPAQUE",
    "async_flag": false,
    "binfile_name": "fusedabsmax1.so",
    "compute_cost": 10,
    "kernel_name": "CusFusedAbsMax1",
    "partial_flag": true,
    "attr": [
        {
            "name": "origin_shape",
            "param_type": "required",
            "type": "listInt",
            "value": "all"
        }
    ],
    "inputs": [
        {
            "index": 0,
            "dtype": [
                "float32"
            ],
            "format": [
                "DefaultFormat"
            ],
            "name": "x1",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        }
    ],
    "outputs": [
        {
            "index": 0,
            "dtype": [
                "float32"
            ],
            "format": [
                "DefaultFormat"
            ],
            "name": "y",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        }
    ]
}""")
def CusFusedAbsMax1(input_x, output, origin_shape = None, kernel_name="fused_abs_max1"):
    return
