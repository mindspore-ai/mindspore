from te import tik
from topi.cce import util
from mindspore.ops.op_info_register import op_info_register
@op_info_register("""{
    "op_name": "CusImg2ColNC1HWC0",
    "imply_type": "TBE",
    "fusion_type": "OPAQUE",
    "async_flag": false,
    "binfile_name": "img2colnc1hwc0.so",
    "compute_cost": 10,
    "kernel_name": "CusImg2ColNC1HWC0",
    "partial_flag": true,
    "attr": [
        {
            "name": "ksizes",
            "param_type": "required",
            "type": "listInt",
            "value": "all"
        },
        {
            "name": "strides",
            "param_type": "required",
            "type": "listInt",
            "value": "all"
        },
        {
            "name": "dilates",
            "param_type": "required",
            "type": "listInt",
            "value": "all"
        },
        {
            "name": "padding",
            "param_type": "required",
            "type": "str",
            "value": "all"
        }
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
                "FRACTAL_NZ"
            ],
            "name": "y",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        }
    ]
}""")
 
def CusImg2ColNC1HWC0(input_x, output, ksizes, strides, dilates, padding, kernel_name="img2col"):
    return
