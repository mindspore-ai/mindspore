from te import tik
from topi.cce import util
from mindspore.ops.op_info_register import op_info_register
@op_info_register("""{
    "op_name": "CusCholeskyTrsm",
    "imply_type": "TBE",
    "fusion_type": "OPAQUE",
    "async_flag": false,
    "binfile_name": "choleskytrsm.so",
    "compute_cost": 10,
    "kernel_name": "CusCholeskyTrsm",
    "partial_flag": true,
    "attr": [
 
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
 
 
def CusCholeskyTrsm(input_x,output, kernel_name):
    return
