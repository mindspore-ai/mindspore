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

#include "inc/fractional_max_pool_3d_grad_with_fixed_ksize_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
// -----------------FractionalMaxPool3DGradWithFixedKsize start----------------
IMPLEMT_COMMON_INFERFUNC(FractionalMaxPool3DGradWithFixedKsizeInferShape) {
  const size_t DIM_SIZE4 = 4;
  const size_t DIM_SIZE5 = 5;
  TensorDesc origin_input_desc = op.GetInputDescByName("origin_input");
  TensorDesc out_backprop_desc = op.GetInputDescByName("out_backprop");
  TensorDesc argmax_desc = op.GetInputDescByName("argmax");
  TensorDesc out_desc = op.GetOutputDescByName("y");
  Format input_format = origin_input_desc.GetFormat();
  DataType out_backprop_type = out_backprop_desc.GetDataType();

  std::vector<int64_t> origin_input_shape = origin_input_desc.GetShape().GetDims();
  std::vector<int64_t> out_backprop_shape = out_backprop_desc.GetShape().GetDims();
  std::vector<int64_t> argmax_shape = argmax_desc.GetShape().GetDims();
  auto origin_input_dims = origin_input_shape.size();
  auto out_backprop_dims = out_backprop_shape.size();
  auto argmax_dims = argmax_shape.size();

  if ((origin_input_dims != DIM_SIZE4) && (origin_input_dims != DIM_SIZE5)) {
    OP_LOGE(TbeGetName(op).c_str(), "length of origin_input should be 4 or 5!");
    return GRAPH_FAILED;
  }
  if ((out_backprop_dims != DIM_SIZE4) && (out_backprop_dims != DIM_SIZE5)) {
    OP_LOGE(TbeGetName(op).c_str(), "length of out_backprop should be 4 or 5!");
    return GRAPH_FAILED;
  }
  if ((argmax_dims != DIM_SIZE4) && (argmax_dims != DIM_SIZE5)) {
    OP_LOGE(TbeGetName(op).c_str(), "length of argmax should be 4 or 5!");
    return GRAPH_FAILED;
  }

  std::string data_format;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", data_format)) {
    std::string err_msg = GetInputInvalidErrMsg("data_format");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  if (data_format != "NDHWC" && data_format != "NCDHW") {
    string expected_format_list = ConcatString("NDHWC, NCDHW");
    std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, data_format);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  // set data type
  out_desc.SetDataType(out_backprop_type);
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), std::string("update output[y] desc failed."));
    return GRAPH_FAILED;
  }

  // set  shape
  if ((input_format == FORMAT_NCDHW && data_format != "NCDHW") ||
      (input_format == FORMAT_NDHWC && data_format != "NDHWC")) {
    string expected_format = ConcatString("Format of input must be same with data_format! input_format:", input_format,
                                          ", data_format:", data_format);
    std::string err_msg = OtherErrMsg(expected_format);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<int64_t> output_size;
  int64_t n_dim = 0;
  int64_t c_dim = 0;
  int64_t d_dim = 0;
  int64_t h_dim = 0;
  int64_t w_dim = 0;

  if (origin_input_dims == 4) {
    if (data_format == "NCDHW") {
      c_dim = origin_input_desc.GetShape().GetDim(0);
      d_dim = origin_input_desc.GetShape().GetDim(1);
      h_dim = origin_input_desc.GetShape().GetDim(2);
      w_dim = origin_input_desc.GetShape().GetDim(3);
      output_size.push_back(c_dim);
      output_size.push_back(d_dim);
      output_size.push_back(h_dim);
      output_size.push_back(w_dim);
    } else {
      d_dim = origin_input_desc.GetShape().GetDim(0);
      h_dim = origin_input_desc.GetShape().GetDim(1);
      w_dim = origin_input_desc.GetShape().GetDim(2);
      c_dim = origin_input_desc.GetShape().GetDim(3);
      output_size.push_back(d_dim);
      output_size.push_back(h_dim);
      output_size.push_back(w_dim);
      output_size.push_back(c_dim);
    }
  } else {
    if (data_format == "NCDHW") {
      n_dim = origin_input_desc.GetShape().GetDim(0);
      c_dim = origin_input_desc.GetShape().GetDim(1);
      d_dim = origin_input_desc.GetShape().GetDim(2);
      h_dim = origin_input_desc.GetShape().GetDim(3);
      w_dim = origin_input_desc.GetShape().GetDim(4);
      output_size.push_back(n_dim);
      output_size.push_back(c_dim);
      output_size.push_back(d_dim);
      output_size.push_back(h_dim);
      output_size.push_back(w_dim);
    } else {
      n_dim = origin_input_desc.GetShape().GetDim(0);
      d_dim = origin_input_desc.GetShape().GetDim(1);
      h_dim = origin_input_desc.GetShape().GetDim(2);
      w_dim = origin_input_desc.GetShape().GetDim(3);
      c_dim = origin_input_desc.GetShape().GetDim(4);
      output_size.push_back(n_dim);
      output_size.push_back(d_dim);
      output_size.push_back(h_dim);
      output_size.push_back(w_dim);
      output_size.push_back(c_dim);
    }
  }
  out_desc.SetShape(ge::Shape(output_size));
  out_desc.SetFormat(input_format);

  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Fail to update output y!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_VERIFIER(FractionalMaxPool3DGradWithFixedKsize, FractionalMaxPool3DGradWithFixedKsizeVerify) {
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(FractionalMaxPool3DGradWithFixedKsize, FractionalMaxPool3DGradWithFixedKsizeInferShape);
CUST_VERIFY_FUNC_REG(FractionalMaxPool3DGradWithFixedKsize, FractionalMaxPool3DGradWithFixedKsizeVerify);
// -----------------FractionalMaxPool3DGradWithFixedKsize end----------------
}  // namespace ge