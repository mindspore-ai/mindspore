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

#include "inc/fractional_max_pool_with_fixed_ksize_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
// ---------------------------------FractionalMaxPoolWithFixedKsize start-------------------------------------------
CUST_IMPLEMT_INFERFUNC(FractionalMaxPoolWithFixedKsize, FractionalMaxPoolWithFixedKsizeInfer) {
  TensorDesc input_x = op.GetInputDescByName("x");
  DataType input_x_dtype = input_x.GetDataType();
  std::vector<int64_t> input_x_dims = input_x.GetShape().GetDims();
  if (input_x_dims.size() != 4) {
    OP_LOGE(TbeGetName(op).c_str(), "Dim of input x must be 4.");
    return GRAPH_FAILED;
  }
  int64_t output_dim_H;
  int64_t output_dim_W;
  std::vector<int64_t> output_shape_list;
  if (ge::GRAPH_SUCCESS == op.GetAttr("output_shape", output_shape_list)) {
    if (output_shape_list.size() == 1) {
      output_dim_H = output_shape_list[0];
      output_dim_W = output_shape_list[0];
    } else if (output_shape_list.size() == 2) {
      output_dim_H = output_shape_list[0];
      output_dim_W = output_shape_list[1];
    } else {
      OP_LOGE(TbeGetName(op).c_str(), "The length of output_shape must be 1 or 2.");
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGE(TbeGetName(op).c_str(), "GetOpAttr output_shape failed!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> output_y_dims;
  std::string data_format;
  if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
    if (data_format == "NCHW") {
      output_y_dims.push_back(input_x_dims[0]);
      output_y_dims.push_back(input_x_dims[1]);
      output_y_dims.push_back(output_dim_H);
      output_y_dims.push_back(output_dim_W);
    } else {
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
  Shape output_dims(output_y_dims);

  TensorDesc output_y = op.GetOutputDescByName("y");
  output_y.SetDataType(input_x_dtype);
  output_y.SetFormat(ge::FORMAT_NCHW);
  output_y.SetShape(output_dims);
  (void)op.UpdateOutputDesc("y", output_y);

  TensorDesc output_argmax = op.GetOutputDescByName("argmax");
  output_argmax.SetDataType(DT_INT64);
  output_argmax.SetFormat(ge::FORMAT_ND);
  output_argmax.SetShape(output_dims);
  (void)op.UpdateOutputDesc("argmax", output_argmax);

  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(FractionalMaxPoolWithFixedKsize, FractionalMaxPoolWithFixedKsizeInfer);
// ---------------------------------FractionalMaxPoolWithFixedKsize end-------------------------------------------
}  // namespace ge