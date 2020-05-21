from te import tik
from topi.cce import util
from mindspore.ops.op_info_register import op_info_register
@op_info_register("""{
    "op_name": "CusTranspose02314",
    "imply_type": "TBE",
    "fusion_type": "OPAQUE",
    "async_flag": false,
    "binfile_name": "transpose02314.so",
    "compute_cost": 10,
    "kernel_name": "CusTranspose02314",
    "partial_flag": true,
    "attr": [
    ],
    "inputs": [
        {
            "index": 0,
            "dtype": [
                "float16"
            ],
            "format": [
                "NC1HWC0"
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
                "float16"
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
 
def CusTranspose02314(input_x, output, kernel_name="transpose021354"):
    return
