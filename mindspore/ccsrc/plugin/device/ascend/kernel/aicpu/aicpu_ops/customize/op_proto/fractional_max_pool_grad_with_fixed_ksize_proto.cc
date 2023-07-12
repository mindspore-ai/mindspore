/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "inc/fractional_max_pool_grad_with_fixed_ksize_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
// -----------------------FractionalMaxPoolGradWithFixedKsize start-------------------------------------
CUST_IMPLEMT_INFERFUNC(FractionalMaxPoolGradWithFixedKsize, FractionalMaxPoolGradWithFixedKsizeInfer) {
  std::string data_format;
  if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
    if (data_format.compare("NCHW") != 0) {
      std::string expected_format_list = ConcatString("NCHW");
      std::string err_msg =
        GetInputFormatNotSupportErrMsg("data_format", expected_format_list, ConcatString(data_format));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGE(TbeGetName(op).c_str(), "GetOpAttr data_format failed!");
    return GRAPH_FAILED;
  }

  TensorDesc orig_input_desc = op.GetInputDescByName("origin_input");
  std::vector<int64_t> orig_input_dims = orig_input_desc.GetShape().GetDims();
  if (orig_input_dims.size() != 4) {
    OP_LOGE(TbeGetName(op).c_str(), "Dim of input[origin_input] must be 4.");
    return GRAPH_FAILED;
  }
  Format orig_input_format = orig_input_desc.GetFormat();
  if (orig_input_format != FORMAT_NCHW) {
    OP_LOGE(TbeGetName(op).c_str(), "The data_format of input[origin_input] must be NCHW.");
    return GRAPH_FAILED;
  }
  Shape y_dims(orig_input_dims);

  TensorDesc out_backprop_desc = op.GetInputDescByName("out_backprop");
  Format out_backprop_format = out_backprop_desc.GetFormat();
  if (out_backprop_format != FORMAT_NCHW) {
    OP_LOGE(TbeGetName(op).c_str(), "The data_format of input[out_backprop] must be NCHW.");
    return GRAPH_FAILED;
  }
  DataType out_backprop_dtype = out_backprop_desc.GetDataType();

  TensorDesc y_desc = op.GetOutputDescByName("y");
  y_desc.SetDataType(out_backprop_dtype);
  y_desc.SetFormat(FORMAT_NCHW);
  y_desc.SetShape(y_dims);
  (void)op.UpdateOutputDesc("y", y_desc);

  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(FractionalMaxPoolGradWithFixedKsize, FractionalMaxPoolGradWithFixedKsizeInfer);
// -----------------------FractionalMaxPoolGradWithFixedKsize end---------------------------------------
}  // namespace ge