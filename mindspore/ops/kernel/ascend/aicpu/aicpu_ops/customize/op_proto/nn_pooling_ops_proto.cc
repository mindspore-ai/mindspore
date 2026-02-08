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
#include <cmath>
#include <utility>
#include <vector>
#include <unordered_map>
#include "op_proto/inc/nn_pooling_ops.h"
#include "custom_op_proto/cust_nn_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"

namespace ge {
// ---------------AdaptiveAvgPool2D-------------------
CUST_IMPLEMT_INFERFUNC(AdaptiveAvgPool2D, AdaptiveAvgPool2dInferShape) {
  OP_LOGI(TbeGetName(op).c_str(), " AdaptiveAvgPool2d inferShape begin!");
  const size_t DIM_SIZE2 = 2;
  auto input_tensor_desc = op.GetInputDescByName("x");
  auto shape = input_tensor_desc.GetShape();
  // get output_size
  std::vector<int64_t> ouput_size_list;
  if (GRAPH_SUCCESS != op.GetAttr("output_size", ouput_size_list)) {
    OP_LOGE(TbeGetName(op).c_str(), "GetOpAttr ouput_size_list failed!");
    return GRAPH_FAILED;
  }
  // check output size
  if (ouput_size_list.size() != DIM_SIZE2) {
    OP_LOGE(TbeGetName(op).c_str(), "length of output_size must be 2");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> dims_input = shape.GetDims();
  if (dims_input.size() < DIM_SIZE2) {
    OP_LOGE(TbeGetName(op).c_str(), "input rank must be greater than 2.");
    return GRAPH_FAILED;
  }
  // set output shape
  std::vector<int64_t> dim_vector;
  for (size_t i = 0; i < dims_input.size(); i++) {
    int64_t dims = dims_input[i];
    dim_vector.push_back(dims);
  }
  size_t index0 = dims_input.size() - 2;
  size_t index1 = dims_input.size() - 1;
  if (ouput_size_list[0] > 0) {
    dim_vector[index0] = ouput_size_list[0];
  }
  if (ouput_size_list[1] > 0) {
    dim_vector[index1] = ouput_size_list[1];
  }
  TensorDesc td = op.GetOutputDescByName("y");
  DataType input_dtype = input_tensor_desc.GetDataType();
  Shape output_shape(dim_vector);
  td.SetShape(output_shape);
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_VERIFIER(AdaptiveAvgPool2D, AdaptiveAvgPool2dVerify) { return GRAPH_SUCCESS; }

CUST_INFER_FUNC_REG(AdaptiveAvgPool2D, AdaptiveAvgPool2dInferShape);
CUST_VERIFY_FUNC_REG(AdaptiveAvgPool2D, AdaptiveAvgPool2dVerify);
// ---------------AdaptiveAvgPool2D End---------------

// ---------------AdaptiveAvgPool2DGrad-------------------
CUST_IMPLEMT_INFERFUNC(AdaptiveAvgPool2DGrad, AdaptiveAvgPool2dGradInferShape) {
  std::vector<std::string> input_infer_depends = {"orig_input_shape"};
  PREPARE_DYNAMIC_SHAPE(input_infer_depends);
  DataType input_dtype = op.GetInputDescByName("input_grad").GetDataType();
  Shape output_shape;
  Tensor orig_input_shape_tensor;
  if (op.GetInputConstData("orig_input_shape", orig_input_shape_tensor) != GRAPH_SUCCESS) {
    auto output_desc = op.GetOutputDescByName("output_grad");
    output_desc.SetDataType(input_dtype);
    output_desc.SetShape(Shape(ge::UNKNOWN_RANK));
    return op.UpdateOutputDesc("output_grad", output_desc);
  }
  MakeShapeFromShapeTensor(orig_input_shape_tensor, output_shape, op);
  TensorDesc output_grad = op.GetOutputDescByName("output_grad");
  output_grad.SetShape(output_shape);
  output_grad.SetDataType(input_dtype);
  return op.UpdateOutputDesc("output_grad", output_grad);
}

CUST_INFER_FUNC_REG(AdaptiveAvgPool2DGrad, AdaptiveAvgPool2dGradInferShape);
// ---------------AdaptiveAvgPool2DGrad END-------------------

// --------- AdaptiveAvgPool3d ---------------
IMPLEMT_COMMON_INFERFUNC(AdaptiveAvgPool3dInferShape) {
  // verify the dim of output_size
  std::vector<int64_t> output_size;
  if (GRAPH_SUCCESS != op.GetAttr("output_size", output_size)) {
    OP_LOGE(TbeGetName(op).c_str(), "GetOpAttr output_size failed!");
    return GRAPH_PARAM_INVALID;
  }
  ge::AscendString op_name;
  (void)op.GetName(op_name);
  auto input_desc = op.GetInputDescByName("x");
  TensorDesc out_desc = op.GetOutputDescByName("y");

  // update data type
  DataType input_type = input_desc.GetDataType();
  out_desc.SetDataType(input_type);

  std::vector<int64_t> input_size_shape = input_desc.GetShape().GetDims();
  auto input_size_dim_num = input_size_shape.size();
  std::vector<int64_t> output_shape(input_size_shape.begin(), input_size_shape.end());
  auto output_size_num = output_size.size();
  if (input_size_dim_num <= 3) {
    OP_LOGE(TbeGetName(op).c_str(), "Input rank must be greater than 3.");
    return GRAPH_PARAM_INVALID;
  }
  if (output_size_num == 1) {
    for (uint64_t i = input_size_dim_num - 3; i < input_size_dim_num; ++i) {
      if (output_size[0] < 0) {
        continue;
      }
      output_shape[i] = output_size[0];
    }
  } else if (output_size_num == 3) {
    for (uint64_t i = input_size_dim_num - 3; i < input_size_dim_num; ++i) {
      auto data = output_size[i - input_size_dim_num + 3];
      if (data < 0) {
        continue;
      }
      output_shape[i] = data;
    }
  } else {
    OP_LOGE("AdaptiveAvgPool3d", "Shape of output_size is invalid");
    return GRAPH_FAILED;
  }

  out_desc.SetShape(Shape(output_shape));
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    OP_LOGE("AdaptiveAvgPool3d", "failed to update output desc");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(AdaptiveAvgPool3d, AdaptiveAvgPool3dInferShape);
// --------- AdaptiveAvgPool3d end---------------

// --------- AdaptiveAvgPool3dGrad ---------------
CUST_IMPLEMT_VERIFIER(AdaptiveAvgPool3dGrad, AdaptiveAvgPool3dGradVerify) {
  auto input_grad_desc = op.GetInputDescByName("input_grad");
  auto orig_input_shape_desc = op.GetInputDescByName("orig_input_shape");
  ge::AscendString op_name;
  (void)op.GetName(op_name);

  auto orig_input_shape_dim = orig_input_shape_desc.GetShape().GetDimNum();
  if (orig_input_shape_dim != 1) {
    OP_LOGE("AdaptiveAvgPool3dGrad", "Num Dim of orig_input_shape is invalid");
    return GRAPH_PARAM_INVALID;
  }

  auto orig_input_dim_num = orig_input_shape_desc.GetShape().GetShapeSize();
  auto input_grad_dim_num = input_grad_desc.GetShape().GetDimNum();

  if (orig_input_dim_num != static_cast<int64_t>(input_grad_dim_num)) {
    OP_LOGE("AdaptiveAvgPool3dGrad", "Num Dim of orig_input and input_grad should be the same");
    return GRAPH_PARAM_INVALID;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AdaptiveAvgPool3dGradInferShape) {
  auto input_desc = op.GetInputDescByName("input_grad");
  auto orig_input_shape_desc = op.GetInputDescByName("orig_input_shape");
  TensorDesc out_desc = op.GetOutputDescByName("output_grad");
  ge::AscendString op_name;
  (void)op.GetName(op_name);

  // update data type
  DataType input_type = input_desc.GetDataType();
  out_desc.SetDataType(input_type);

  // infer shape
  Tensor orig_input_size_tensor;
  if (op.GetInputConstData("orig_input_shape", orig_input_size_tensor) != GRAPH_SUCCESS) {
    OP_LOGE("AdaptiveAvgPool3dGrad", "failed to get tensor from output_size");
    return GRAPH_FAILED;
  }

  int32_t *orig_input_size_data = reinterpret_cast<int32_t *>(orig_input_size_tensor.GetData());
  if (orig_input_size_data == nullptr) {
    OP_LOGE("AdaptiveAvgPool3dGrad", "output_size data is invalid");
    return GRAPH_PARAM_INVALID;
  }

  auto input_size_dim_num = input_desc.GetShape().GetDimNum();
  std::vector<int64_t> output_shape(input_size_dim_num);

  for (uint64_t i = 0; i < input_size_dim_num; ++i) {
    output_shape[i] = orig_input_size_data[i];
  }

  out_desc.SetShape(Shape(output_shape));
  if (op.UpdateOutputDesc("output_grad", out_desc) != GRAPH_SUCCESS) {
    OP_LOGE("AdaptiveAvgPool3dGrad", "failed to update output desc");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(AdaptiveAvgPool3dGrad, AdaptiveAvgPool3dGradInferShape);
CUST_VERIFY_FUNC_REG(AdaptiveAvgPool3dGrad, AdaptiveAvgPool3dGradVerify);
// --------- AdaptiveAvgPool3dGrad end---------------

// --------- AdaptiveMaxPool3d---------------
CUST_IMPLEMT_INFERFUNC(AdaptiveMaxPool3d, AdaptiveMaxPool3dInferShape) {
  TensorDesc input = op.GetInputDesc(0);
  TensorDesc output_size = op.GetInputDesc(1);
  TensorDesc output = op.GetOutputDesc(0);
  TensorDesc argmax = op.GetOutputDesc(1);

  const size_t input_num_dims = input.GetShape().GetDimNum();
  const std::vector<int64_t> output_size_shape = output_size.GetShape().GetDims();
  if (!(input_num_dims == 4 || input_num_dims == 5)) {
    OP_LOGE(TbeGetName(op), "Input dimensions must be equal to 4 or 5.");
    return GRAPH_FAILED;
  }
  if (output_size_shape.size() != 1) {
    OP_LOGE(TbeGetName(op), "output_size dim should be equal to 1.");
    return GRAPH_FAILED;
  }
  if (output_size_shape[0] != 3) {
    OP_LOGE(TbeGetName(op), "output_size shape[0] should be equal to 3.");
    return GRAPH_FAILED;
  }

  DataType input_dtype = input.GetDataType();
  Shape output_shape(UNKNOWN_SHAPE);
  output.SetDataType(input_dtype);
  output.SetShape(output_shape);
  argmax.SetDataType(DT_INT32);
  argmax.SetShape(output_shape);
  (void)op.UpdateOutputDesc("y", output);
  (void)op.UpdateOutputDesc("argmax", argmax);
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_VERIFIER(AdaptiveMaxPool3d, AdaptiveMaxPool3dVerify) { return GRAPH_SUCCESS; }

CUST_INFER_FUNC_REG(AdaptiveMaxPool3d, AdaptiveMaxPool3dInferShape);
CUST_VERIFY_FUNC_REG(AdaptiveMaxPool3d, AdaptiveMaxPool3dVerify);
// --------- AdaptiveMaxPool3d END---------------

// --------- AdaptiveMaxPool2dGrad---------------
CUST_IMPLEMT_INFERFUNC(AdaptiveMaxPool2dGrad, AdaptiveMaxPool2dGradInferShape) {
  TensorDesc input_grad = op.GetOutputDescByName("x_grad");
  TensorDesc input = op.GetInputDescByName("x");
  DataType input_dtype = input.GetDataType();
  Shape input_shape = input.GetShape();
  input_grad.SetShape(input_shape);
  input_grad.SetDataType(input_dtype);
  if (op.UpdateOutputDesc("x_grad", input_grad) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_VERIFIER(AdaptiveMaxPool2dGrad, AdaptiveMaxPool2dGradVerify) { return GRAPH_SUCCESS; }

CUST_INFER_FUNC_REG(AdaptiveMaxPool2dGrad, AdaptiveMaxPool2dGradInferShape);
CUST_VERIFY_FUNC_REG(AdaptiveMaxPool2dGrad, AdaptiveMaxPool2dGradVerify);
// --------- AdaptiveMaxPool2dGrad END---------------

// --------- AdaptiveMaxPool3dGrad---------------
CUST_IMPLEMT_INFERFUNC(AdaptiveMaxPool3dGrad, AdaptiveMaxPool3dGradInferShape) {
  TensorDesc output_grad = op.GetOutputDescByName("output_grad");
  TensorDesc input = op.GetInputDescByName("x");
  DataType input_dtype = input.GetDataType();
  Shape input_shape = input.GetShape();
  output_grad.SetShape(input_shape);
  output_grad.SetDataType(input_dtype);
  if (op.UpdateOutputDesc("output_grad", output_grad) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_VERIFIER(AdaptiveMaxPool3dGrad, AdaptiveMaxPool3dGradVerify) { return GRAPH_SUCCESS; }

CUST_INFER_FUNC_REG(AdaptiveMaxPool3dGrad, AdaptiveMaxPool3dGradInferShape);
CUST_VERIFY_FUNC_REG(AdaptiveMaxPool3dGrad, AdaptiveMaxPool3dGradVerify);
// --------- AdaptiveMaxPool3dGrad END---------------

// -------------------DataFormatVecPermute---------------------
IMPLEMT_INFERFUNC(DataFormatVecPermute, DataFormatVecPermuteInfer) {
  auto x_desc = op.GetInputDesc(0);

  std::vector<std::pair<int64_t, int64_t>> range;
  if (x_desc.GetShapeRange(range) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  DataType y_type = x_desc.GetDataType();

  auto y_desc = op.GetOutputDesc(0);
  y_desc.SetShape(x_desc.GetShape());
  y_desc.SetShapeRange(range);
  y_desc.SetDataType(y_type);
  op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DataFormatVecPermute, DataFormatVecPermuteInfer);
// -------------------DataFormatVecPermute End---------------------

// -------------------MaxPool3DWithArgmax---------------------
CUST_IMPLEMT_INFERFUNC(MaxPool3DWithArgmax, MaxPool3DWithArgmaxInferShape) {
  TensorDesc input_desc = op.GetInputDescByName("x");
  auto input_shape = input_desc.GetShape().GetDims();
  DataType input_dtype = input_desc.GetDataType();
  TensorDesc output_desc = op.GetOutputDescByName("y");
  TensorDesc argmax_desc = op.GetOutputDescByName("argmax");

  constexpr size_t kRank = 5;
  if (IsUnknownRankShape(input_desc.GetShape()) || IsUnknownShape(input_desc.GetShape())) {
    std::vector<int64_t> output_shape_vec(kRank, ge::UNKNOWN_DIM);
    Shape output_shape(output_shape_vec);
    output_desc.SetShape(output_shape);
    argmax_desc.SetShape(output_shape);
    op.UpdateOutputDesc("y", output_desc);
    op.UpdateOutputDesc("argmax", argmax_desc);
    return GRAPH_SUCCESS;
  }

  if (input_shape.size() != kRank) {
    std::string err_msg = GetAttrSizeErrMsg("input_shape", ConcatString(input_shape.size()), ConcatString(kRank));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> ksize;
  op.GetAttr("ksize", ksize);
  std::vector<int64_t> strides;
  op.GetAttr("strides", strides);
  std::vector<int64_t> dilation;
  op.GetAttr("dilation", dilation);
  std::vector<int64_t> pads;
  op.GetAttr("pads", pads);
  bool ceil_mode;
  op.GetAttr("ceil_mode", ceil_mode);

  const size_t d_idx = 0;
  const size_t h_idx = 1;
  const size_t w_idx = 2;
  auto input_d = input_shape[2];
  auto input_h = input_shape[3];
  auto input_w = input_shape[4];
  int64_t output_d = 0;
  int64_t output_h = 0;
  int64_t output_w = 0;
  int64_t factor = 2;
  if (!ceil_mode) {
    output_d = static_cast<int64_t>(
      std::floor(static_cast<float>(input_d + factor * pads[d_idx] - dilation[d_idx] * (ksize[d_idx] - 1) - 1) /
                   static_cast<float>(strides[d_idx]) +
                 1));
    output_h = static_cast<int64_t>(
      std::floor(static_cast<float>(input_h + factor * pads[h_idx] - dilation[h_idx] * (ksize[h_idx] - 1) - 1) /
                   static_cast<float>(strides[h_idx]) +
                 1));
    output_w = static_cast<int64_t>(
      std::floor(static_cast<float>(input_w + factor * pads[w_idx] - dilation[w_idx] * (ksize[w_idx] - 1) - 1) /
                   static_cast<float>(strides[w_idx]) +
                 1));
  } else {
    output_d = static_cast<int64_t>(
      std::ceil(static_cast<float>(input_d + factor * pads[d_idx] - dilation[d_idx] * (ksize[d_idx] - 1) - 1) /
                  static_cast<float>(strides[d_idx]) +
                1));
    output_h = static_cast<int64_t>(
      std::ceil(static_cast<float>(input_h + factor * pads[h_idx] - dilation[h_idx] * (ksize[h_idx] - 1) - 1) /
                  static_cast<float>(strides[h_idx]) +
                1));
    output_w = static_cast<int64_t>(
      std::ceil(static_cast<float>(input_w + factor * pads[w_idx] - dilation[w_idx] * (ksize[w_idx] - 1) - 1) /
                  static_cast<float>(strides[w_idx]) +
                1));
    // The last pooling starts inside the image.
    if ((output_d - 1) * strides[d_idx] >= input_d + pads[d_idx]) {
      --output_d;
    }
    if ((output_h - 1) * strides[h_idx] >= input_h + pads[h_idx]) {
      --output_h;
    }
    if ((output_w - 1) * strides[w_idx] >= input_w + pads[w_idx]) {
      --output_w;
    }
  }

  std::vector<int64_t> output_shape_vec{input_shape[0], input_shape[1], output_d, output_h, output_w};
  Shape output_shape(output_shape_vec);
  output_desc.SetDataType(input_dtype);
  output_desc.SetShape(output_shape);
  op.UpdateOutputDesc("y", output_desc);

  std::string argmax_type;
  op.GetAttr("argmax_type", argmax_type);
  if (argmax_type == "int32") {
    argmax_desc.SetDataType(DT_INT32);
  } else if (argmax_type == "int64") {
    argmax_desc.SetDataType(DT_INT64);
  } else {
    OP_LOGE(TbeGetName(op), "The 'argmax_type' must be 'int32' or 'int64', but got %s.", argmax_type);
    return GRAPH_FAILED;
  }
  argmax_desc.SetShape(output_shape);
  op.UpdateOutputDesc("argmax", argmax_desc);

  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_VERIFIER(MaxPool3DWithArgmax, MaxPool3DWithArgmaxVerify) {
  constexpr size_t kAttrsSize = 3;
  std::vector<int64_t> ksize;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksize)) {
    std::string err_msg = GetInputInvalidErrMsg("ksize");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  if (ksize.size() != kAttrsSize) {
    std::string err_msg = GetAttrSizeErrMsg("ksize", ConcatString(ksize.size()), ConcatString(kAttrsSize));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> strides;
  if (GRAPH_SUCCESS != op.GetAttr("strides", strides)) {
    std::string err_msg = GetInputInvalidErrMsg("strides");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  if (strides.size() != kAttrsSize) {
    std::string err_msg = GetAttrSizeErrMsg("strides", ConcatString(strides.size()), ConcatString(kAttrsSize));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> pads;
  if (GRAPH_SUCCESS != op.GetAttr("pads", pads)) {
    std::string err_msg = GetInputInvalidErrMsg("pads");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  if (pads.size() != kAttrsSize) {
    std::string err_msg = GetAttrSizeErrMsg("pads", ConcatString(pads.size()), ConcatString(kAttrsSize));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dilation;
  if (GRAPH_SUCCESS != op.GetAttr("dilation", dilation)) {
    std::string err_msg = GetInputInvalidErrMsg("dilation");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  if (dilation.size() != kAttrsSize) {
    std::string err_msg = GetAttrSizeErrMsg("dilation", ConcatString(dilation.size()), ConcatString(kAttrsSize));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  bool ceil_mode = false;
  if (GRAPH_SUCCESS != op.GetAttr("ceil_mode", ceil_mode)) {
    std::string err_msg = GetInputInvalidErrMsg("ceil_mode");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  std::string data_format;
  if (op.GetAttr("data_format", data_format) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "get attr data_format failed.");
    return GRAPH_FAILED;
  }
  if (data_format != "NCDHW") {
    OP_LOGE(TbeGetName(op).c_str(), "Attr data_format(%s) only support NCDHW.", data_format.c_str());
    return GRAPH_FAILED;
  }

  std::string argmax_type;
  if (GRAPH_SUCCESS != op.GetAttr("argmax_type", argmax_type)) {
    std::string err_msg = GetInputInvalidErrMsg("argmax_type");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(MaxPool3DWithArgmax, MaxPool3DWithArgmaxInferShape);
CUST_VERIFY_FUNC_REG(MaxPool3DWithArgmax, MaxPool3DWithArgmaxVerify);
// ------------------MaxPool3DWithArgmax---------------------

// ------------------MaxPool3DGradWithArgMax---------------------
CUST_IMPLEMT_VERIFIER(MaxPool3DGradWithArgmax, MaxPool3DGradWithArgmaxVerify) {
  const size_t DIM_SIZE1 = 1;
  const size_t DIM_SIZE3 = 3;
  const size_t DIM_SIZE5 = 5;

  std::vector<int32_t> ksizeList;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksizeList)) {
    std::string err_msg = GetInputInvalidErrMsg("ksize");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  if ((ksizeList.size() != DIM_SIZE1) && (ksizeList.size() != DIM_SIZE3)) {
    string excepted_size = ConcatString(DIM_SIZE1, " or ", DIM_SIZE3);
    std::string err_msg = GetAttrSizeErrMsg("ksizeList", ConcatString(ksizeList.size()), excepted_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<int32_t> stridesList;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stridesList)) {
    std::string err_msg = GetInputInvalidErrMsg("strides");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  if ((stridesList.size() != DIM_SIZE1) && (stridesList.size() != DIM_SIZE3)) {
    string excepted_size = ConcatString(DIM_SIZE1, " or ", DIM_SIZE3);
    std::string err_msg = GetAttrSizeErrMsg("stridesList", ConcatString(stridesList.size()), excepted_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<int32_t> padsList;
  if (GRAPH_SUCCESS != op.GetAttr("pads", padsList)) {
    std::string err_msg = GetInputInvalidErrMsg("pads");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  if ((padsList.size() != DIM_SIZE1) && (padsList.size() != DIM_SIZE3)) {
    string excepted_size = ConcatString(DIM_SIZE1, " or ", DIM_SIZE3);
    std::string err_msg = GetAttrSizeErrMsg("padsList", ConcatString(padsList.size()), excepted_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<int32_t> dilationList;
  if (GRAPH_SUCCESS != op.GetAttr("dilation", dilationList)) {
    std::string err_msg = GetInputInvalidErrMsg("dilation");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  if ((dilationList.size() != DIM_SIZE1) && (dilationList.size() != DIM_SIZE3) && (dilationList.size() != DIM_SIZE5)) {
    string excepted_value = ConcatString(DIM_SIZE1, " or ", DIM_SIZE3, " or ", DIM_SIZE5);
    std::string err_msg = GetAttrSizeErrMsg("dilationList", ConcatString(dilationList.size()), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  bool ceilMode = false;
  if (GRAPH_SUCCESS != op.GetAttr("ceil_mode", ceilMode)) {
    std::string err_msg = GetInputInvalidErrMsg("ceil_mode");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  std::string data_format;
  if (op.GetAttr("data_format", data_format) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "get attr data_format failed.");
    return GRAPH_FAILED;
  }
  if (data_format != "NCDHW") {
    OP_LOGE(TbeGetName(op).c_str(), "Attr data_format(%s) only support NCDHW.", data_format.c_str());
    return GRAPH_FAILED;
  }

  std::string argmax_type;
  if (GRAPH_SUCCESS != op.GetAttr("argmax_type", argmax_type)) {
    std::string err_msg = GetInputInvalidErrMsg("argmax_type");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  auto grads_desc = op.GetInputDesc("grads");
  vector<int64_t> grads_shape = grads_desc.GetShape().GetDims();
  if (grads_shape.size() != DIM_SIZE5 && !IsUnknownRankShape(grads_shape)) {
    OP_LOGE(TbeGetName(op).c_str(), "grads_shape's dim expect: %lu, but real: %lu.", DIM_SIZE5, grads_shape.size());
    return GRAPH_FAILED;
  }

  TensorDesc inputDesc = op.GetInputDescByName("x");
  vector<int64_t> inputShape = inputDesc.GetShape().GetDims();
  if (inputShape.size() != DIM_SIZE5) {
    OP_LOGE(TbeGetName(op).c_str(), "input x's dim expect: %lu, but real: %lu.", DIM_SIZE5, inputShape.size());
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_INFERFUNC(MaxPool3DGradWithArgmax, MaxPool3DGradWithArgmaxInferShape) {
  auto shape = op.GetInputDescByName("x").GetShape();
  auto shape_dims = shape.GetDims();
  TensorDesc td = op.GetOutputDescByName("y");
  td.SetShape(shape);
  td.SetDataType(op.GetInputDescByName("x").GetDataType());
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}
CUST_INFER_FUNC_REG(MaxPool3DGradWithArgmax, MaxPool3DGradWithArgmaxInferShape);
CUST_VERIFY_FUNC_REG(MaxPool3DGradWithArgmax, MaxPool3DGradWithArgmaxVerify);
// ------------------MaxPool3DGradWithArgMax---------------------

// ------------------NthElement---------------------
IMPLEMT_INFERFUNC(NthElement, NthElementInfer) {
  std::vector<std::string> input_infer_depends = {"n"};
  PREPARE_DYNAMIC_SHAPE(input_infer_depends);
  Shape x_shape;
  auto x_tensor = op.get_input_desc_x();
  if (WithRankAtLeast(x_tensor, 1, x_shape, op) != GRAPH_SUCCESS) {
    std::string err_msg =
      ConcatString("failed to call WithRankAtLeast function, ", "input[x] rank must be at least 1D, but got rank[",
                   op.get_input_desc_x().GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  Tensor n_tensor;
  int64_t n_dim = 0;
  if (op.GetInputConstData("n", n_tensor) != GRAPH_SUCCESS) {
    n_dim = ge::UNKNOWN_DIM;
  } else {
    if (MakeDimForScalarInput(n_tensor, n_dim, op) != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("failed to call MakeDimForScalarInput function, "
                                                                    "get input n shape failed"));
      return GRAPH_FAILED;
    }
  }

  int64_t existing = static_cast<int64_t>(x_shape.GetDimNum());
  int64_t last_input_dim = x_shape.GetDim(existing - 1);
  if ((last_input_dim != ge::UNKNOWN_DIM) && (n_dim != ge::UNKNOWN_DIM) && (last_input_dim <= n_dim)) {
    std::string err_msg =
      ConcatString("input[x] last dim value[", last_input_dim, "] must be greater than [", n_dim, "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  Shape output_shape;
  if (SubShape(x_shape, 0, -1, 1, output_shape, op) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call SubShape function, input[x] shape[",
                                       DebugString(x_shape.GetDims()), "], start[0], end[-1], stride[1]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc y_tensor = op.GetOutputDescByName("y");
  y_tensor.SetDataType(x_tensor.GetDataType());
  y_tensor.SetShape(output_shape);
  return op.UpdateOutputDesc("y", y_tensor);
}

INFER_FUNC_REG(NthElement, NthElementInfer);
// ------------------NthElement END---------------------

// ------------------MaxUnpool2d---------------------
graphStatus MaxUnpool2dVerify(const Operator &op, const std::vector<int64_t> &ksize,
                              const std::vector<int64_t> &strides, const std::vector<int64_t> &pads,
                              const std::string &format) {
  if (format != "NCHW" && format != "NHWC") {
    OP_LOGE(TbeGetName(op).c_str(), "Format '%s' not supported.", format.c_str());
    return GRAPH_FAILED;
  }
  std::unordered_map<std::string, const std::vector<int64_t> &> m{
    {"ksize", ksize},
    {"strides", pads},
    {"pads", pads},
  };
  for (const auto &[name, vec] : m) {
    if (vec.size() != 4) {
      OP_LOGE(TbeGetName(op).c_str(), "'%s' has invalid size [%zu]. Should be 4.", name.c_str(), vec.size());
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_INFERFUNC(MaxUnpool2D, MaxUnpool2DInfer) {
  std::vector<int64_t> ksize;
  std::vector<int64_t> strides;
  std::vector<int64_t> pads;
  std::vector<int64_t> output_shape;
  std::string format;
  RETURN_IF_FAILURE(op.GetAttr("ksize", ksize));
  RETURN_IF_FAILURE(op.GetAttr("strides", strides));
  RETURN_IF_FAILURE(op.GetAttr("pads", pads));
  RETURN_IF_FAILURE(op.GetAttr("data_format", format));
  RETURN_IF_FAILURE(MaxUnpool2dVerify(op, ksize, strides, pads, format));

  auto x_desc = op.GetInputDescByName("x");
  auto x_shape = x_desc.GetShape().GetDims();
  auto y_desc = op.GetOutputDescByName("y");
  y_desc.SetDataType(x_desc.GetDataType());

  constexpr size_t kOutputRank = 4;
  if (op.GetAttr("output_shape", output_shape) == GRAPH_SUCCESS && output_shape.size() == kOutputRank) {
    y_desc.SetShape(Shape(output_shape));
  } else if (!IsUnknown(x_shape)) {
    auto [h_idx, w_idx] = format == "NCHW" ? std::make_pair(2, 3) : std::make_pair(1, 2);
    auto &h = x_shape[h_idx];
    auto &w = x_shape[w_idx];
    h = (h - 1) * strides[h_idx] - 2 * pads[h_idx] + ksize[h_idx];
    w = (w - 1) * strides[w_idx] - 2 * pads[w_idx] + ksize[w_idx];
    y_desc.SetShape(Shape(x_shape));
  } else {
    y_desc.SetShape(Shape({UNKNOWN_DIM, UNKNOWN_DIM, UNKNOWN_DIM, UNKNOWN_DIM}));
  }
  return op.UpdateOutputDesc("y", y_desc);
}

CUST_INFER_FUNC_REG(MaxUnpool2D, MaxUnpool2DInfer);
// ------------------MaxUnpool2D END---------------------

// ------------------MaxUnpool3D---------------------
graphStatus MaxUnpool3dVerify(const Operator &op, const std::vector<int64_t> &ksize,
                              const std::vector<int64_t> &strides, const std::vector<int64_t> &pads,
                              const std::string &format) {
  if (format != "NCDHW" && format != "NDHWC") {
    OP_LOGE(TbeGetName(op).c_str(), "Format '%s' not supported.", format.c_str());
    return GRAPH_FAILED;
  }
  std::unordered_map<std::string, const std::vector<int64_t> &> m{
    {"ksize", ksize},
    {"strides", pads},
    {"pads", pads},
  };
  for (const auto &[name, vec] : m) {
    if (vec.size() != 5) {
      OP_LOGE(TbeGetName(op).c_str(), "'%s' has invalid size [%zu]. Should be 1 or 3.", name.c_str(), vec.size());
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_INFERFUNC(MaxUnpool3D, MaxUnpool3DInfer) {
  std::vector<int64_t> ksize;
  std::vector<int64_t> strides;
  std::vector<int64_t> pads;
  std::vector<int64_t> output_shape;
  std::string format;
  RETURN_IF_FAILURE(op.GetAttr("ksize", ksize));
  RETURN_IF_FAILURE(op.GetAttr("strides", strides));
  RETURN_IF_FAILURE(op.GetAttr("pads", pads));
  RETURN_IF_FAILURE(op.GetAttr("data_format", format));
  RETURN_IF_FAILURE(MaxUnpool3dVerify(op, ksize, strides, pads, format));

  auto x_desc = op.GetInputDescByName("x");
  auto x_shape = x_desc.GetShape().GetDims();
  auto y_desc = op.GetOutputDescByName("y");
  y_desc.SetDataType(x_desc.GetDataType());

  constexpr size_t kOutputRank = 5;
  if (op.GetAttr("output_shape", output_shape) == GRAPH_SUCCESS && output_shape.size() == kOutputRank) {
    y_desc.SetShape(Shape(output_shape));
  } else if (!IsUnknown(x_shape)) {
    auto [d_idx, h_idx, w_idx] = format == "NCDHW" ? std::make_tuple(2, 3, 4) : std::make_tuple(1, 2, 3);
    auto &d = x_shape[d_idx];
    auto &h = x_shape[h_idx];
    auto &w = x_shape[w_idx];
    d = (d - 1) * strides[d_idx] - 2 * pads[d_idx] + ksize[d_idx];
    h = (h - 1) * strides[h_idx] - 2 * pads[h_idx] + ksize[h_idx];
    w = (w - 1) * strides[w_idx] - 2 * pads[w_idx] + ksize[w_idx];
    y_desc.SetShape(Shape(x_shape));
  } else {
    y_desc.SetShape(Shape({UNKNOWN_DIM, UNKNOWN_DIM, UNKNOWN_DIM, UNKNOWN_DIM, UNKNOWN_DIM}));
  }
  return op.UpdateOutputDesc("y", y_desc);
}

CUST_INFER_FUNC_REG(MaxUnpool3D, MaxUnpool3DInfer);
// ------------------MaxUnpool3D END---------------------

// -----------------FractionalMaxPool3DWithFixedKsize start----------------
IMPLEMT_COMMON_INFERFUNC(FractionalMaxPool3DWithFixedKsizeInferShape) {
  const size_t DIM_SIZE1 = 1;
  const size_t DIM_SIZE3 = 3;
  const size_t DIM_SIZE4 = 4;
  const size_t DIM_SIZE5 = 5;
  TensorDesc input_desc = op.GetInputDescByName("x");
  TensorDesc random_samples_desc = op.GetInputDescByName("random_samples");
  TensorDesc out_desc = op.GetOutputDescByName("y");
  TensorDesc argmax_desc = op.GetOutputDescByName("argmax");
  Format input_format = input_desc.GetFormat();
  DataType input_type = input_desc.GetDataType();
  DataType argmax_dtype = argmax_desc.GetDataType();

  std::vector<int64_t> input_shape = input_desc.GetShape().GetDims();
  auto input_dims = input_shape.size();
  if ((input_dims != DIM_SIZE4) && (input_dims != DIM_SIZE5)) {
    OP_LOGE(TbeGetName(op).c_str(), "length of x should be 4 or 5!");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> outputshapeList;
  if (GRAPH_SUCCESS != op.GetAttr("output_shape", outputshapeList)) {
    std::string err_msg = GetInputInvalidErrMsg("output_shape");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  if ((outputshapeList.size() != DIM_SIZE1) && (outputshapeList.size() != DIM_SIZE3)) {
    string excepted_size = ConcatString(DIM_SIZE1, " or ", DIM_SIZE3);
    std::string err_msg = GetAttrSizeErrMsg("outputshapeList", ConcatString(outputshapeList.size()), excepted_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  if (outputshapeList.size() == DIM_SIZE1) {
    for (int64_t i = 0; i < 3; i++) {
      outputshapeList[i] = outputshapeList[0];
    }
  }

  std::vector<int64_t> ksizeList;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksizeList)) {
    std::string err_msg = GetInputInvalidErrMsg("ksize");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  if ((ksizeList.size() != DIM_SIZE1) && (ksizeList.size() != DIM_SIZE3)) {
    string excepted_size = ConcatString(DIM_SIZE1, " or ", DIM_SIZE3);
    std::string err_msg = GetAttrSizeErrMsg("ksizeList", ConcatString(ksizeList.size()), excepted_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  if (ksizeList.size() == DIM_SIZE1) {
    for (int64_t i = 0; i < 3; i++) {
      ksizeList[i] = ksizeList[0];
    }
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
  out_desc.SetDataType(input_type);
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), std::string("update output[y] desc failed."));
    return GRAPH_FAILED;
  }

  if (argmax_dtype == DT_UNDEFINED) {
    argmax_desc.SetDataType(DT_INT64);
  }
  argmax_desc.SetDataType(argmax_dtype);
  if (op.UpdateOutputDesc("argmax", argmax_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), std::string("update output[argmax] desc failed."));
    return GRAPH_FAILED;
  }

  std::vector<int64_t> output_size;
  int64_t n_dim = 0;
  int64_t c_dim = 0;
  int64_t outputT = outputshapeList[0];
  int64_t outputH = outputshapeList[1];
  int64_t outputW = outputshapeList[2];

  if (input_dims == 4) {
    if (data_format == "NCDHW") {
      c_dim = input_desc.GetShape().GetDim(0);
      output_size.push_back(c_dim);
      output_size.push_back(outputT);
      output_size.push_back(outputH);
      output_size.push_back(outputW);
    } else {
      c_dim = input_desc.GetShape().GetDim(3);
      output_size.push_back(outputT);
      output_size.push_back(outputH);
      output_size.push_back(outputW);
      output_size.push_back(c_dim);
    }
  } else {
    if (data_format == "NCDHW") {
      n_dim = input_desc.GetShape().GetDim(0);
      c_dim = input_desc.GetShape().GetDim(1);
      output_size.push_back(n_dim);
      output_size.push_back(c_dim);
      output_size.push_back(outputT);
      output_size.push_back(outputH);
      output_size.push_back(outputW);
    } else {
      n_dim = input_desc.GetShape().GetDim(0);
      c_dim = input_desc.GetShape().GetDim(4);
      output_size.push_back(n_dim);
      output_size.push_back(outputT);
      output_size.push_back(outputH);
      output_size.push_back(outputW);
      output_size.push_back(c_dim);
    }
  }
  out_desc.SetFormat(input_format);
  argmax_desc.SetFormat(ge::FORMAT_ND);

  out_desc.SetShape(ge::Shape(output_size));
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Fail to update output y!");
    return GRAPH_FAILED;
  }
  argmax_desc.SetShape(ge::Shape(output_size));
  if (op.UpdateOutputDesc("argmax", argmax_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Fail to update output argmax!");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_VERIFIER(FractionalMaxPool3DWithFixedKsize, FractionalMaxPool3DWithFixedKsizeVerify) {
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(FractionalMaxPool3DWithFixedKsize, FractionalMaxPool3DWithFixedKsizeInferShape);
CUST_VERIFY_FUNC_REG(FractionalMaxPool3DWithFixedKsize, FractionalMaxPool3DWithFixedKsizeVerify);
// -----------------FractionalMaxPool3DWithFixedKsize end----------------

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

static bool CheckListEmpty(const std::string &opName, const std::vector<int64_t> &list, const std::string &attrName) {
  if (list.empty()) {
    OP_LOGE(opName.c_str(), "the %s is empty !", attrName.c_str());
    return false;
  }
  return true;
}

static std::vector<int64_t> GetAttrValue(const ge::Operator &op, const std::string &key_name) {
  std::vector<int64_t> list;
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return list);
  if (ge::GRAPH_SUCCESS != op.GetAttr(key_name.c_str(), list)) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "GetOpAttr ConstValue failed!");
  }

  return list;
}
// ---------------------MaxUnpool2DGrad---------------------
CUST_IMPLEMT_VERIFIER(MaxUnpool2DGrad, MaxUnpool2DGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "x", "grads")) {
    return GRAPH_FAILED;
  }
  std::string data_format;
  if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
    if (data_format != "NCHW" && data_format != "NHWC") {
      string expected_format_list = ConcatString("NCHW, NHWC");
      std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, data_format);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
      return GRAPH_FAILED;
    }
  }
  std::vector<int64_t> ksize;
  ksize = GetAttrValue(op, "ksize");
  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  std::vector<int64_t> pads;
  pads = GetAttrValue(op, "pads");

  if (!CheckListEmpty(TbeGetName(op).c_str(), ksize, "ksize") ||
      !CheckListEmpty(TbeGetName(op).c_str(), strides, "strides") ||
      !CheckListEmpty(TbeGetName(op).c_str(), pads, "pads")) {
    std::string err_msg = OtherErrMsg("The ksize or strides or pads is empty!");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  if (ksize.size() != 4 || strides.size() != 4 || pads.size() != 4) {
    string excepted_size = ConcatString("4");
    std::string err_msg =
      GetAttrSizeErrMsg("ksize.size or strides.size or pads.size", std::to_string(ksize.size()), excepted_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  if (data_format == "NCHW" &&
      (ksize[0] != 1 || ksize[1] != 1 || strides[0] != 1 || strides[1] != 1 || pads[0] != 1 || pads[1] != 1)) {
    string wrong_value =
      ConcatString(ksize[0], " and ", ksize[1], "and", strides[0], "and", strides[1], "and", pads[0], "and", pads[1]);
    std::string err_msg = GetAttrValueErrMsg(
      "ksize[0] and ksize[1] and strides[0] and strides[1] and pads[0] and pads[1]", wrong_value, ConcatString("1"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  if (data_format == "NHWC" &&
      (ksize[0] != 1 || ksize[3] != 1 || strides[0] != 1 || strides[3] != 1 || pads[0] != 1 || pads[3] != 1)) {
    string wrong_value =
      ConcatString(ksize[0], " and ", ksize[3], "and", strides[0], "and", strides[3], "and", pads[0], "and", pads[3]);
    std::string err_msg = GetAttrValueErrMsg(
      "ksize[0] and ksize[3] and strides[0] and strides[3] and pads[0] and pads[3]", wrong_value, ConcatString("1"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_INFERFUNC(MaxUnpool2DGrad, MaxUnpool2DGradInferShape) {
  auto input_tensor_desc = op.GetInputDescByName("x");
  auto input_shape = input_tensor_desc.GetShape();
  std::string data_format;
  if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
    if (data_format != "NCHW" && data_format != "NHWC") {
      string expected_format_list = ConcatString("NCHW, NHWC");
      std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, data_format);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
      return GRAPH_FAILED;
    }
  }
  TensorDesc td = op.GetOutputDescByName("y");
  DataType input_dtype = input_tensor_desc.GetDataType();
  td.SetShape(input_shape);
  td.SetDataType(input_dtype);
  if (op.UpdateOutputDesc("y", td) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(MaxUnpool2DGrad, MaxUnpool2DGradInferShape);
CUST_VERIFY_FUNC_REG(MaxUnpool2DGrad, MaxUnpool2DGradVerify);
// ---------------------MaxUnpool2DGrad---------------------

// ---------------------MaxUnpool3DGrad---------------------
CUST_IMPLEMT_VERIFIER(MaxUnpool3DGrad, MaxUnpool3DGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "x", "grads")) {
    return GRAPH_FAILED;
  }
  std::string data_format;
  if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
    if (data_format != "NCDHW" && data_format != "NDHWC") {
      string expected_format_list = ConcatString("NCDHW, NDHWC");
      std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, data_format);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
      return GRAPH_FAILED;
    }
  }
  std::vector<int64_t> ksize;
  ksize = GetAttrValue(op, "ksize");
  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  std::vector<int64_t> pads;
  pads = GetAttrValue(op, "pads");

  if (!CheckListEmpty(TbeGetName(op).c_str(), ksize, "ksize") ||
      !CheckListEmpty(TbeGetName(op).c_str(), strides, "strides") ||
      !CheckListEmpty(TbeGetName(op).c_str(), pads, "pads")) {
    std::string err_msg = OtherErrMsg("The ksize or strides or pads is empty!");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  if (ksize.size() != 5 || strides.size() != 5 || pads.size() != 5) {
    string excepted_size = ConcatString("5");
    std::string err_msg =
      GetAttrSizeErrMsg("ksize.size or strides.size or pads.size", std::to_string(ksize.size()), excepted_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  if (data_format == "NCDHW" &&
      (ksize[0] != 1 || ksize[1] != 1 || strides[0] != 1 || strides[1] != 1 || pads[0] != 1 || pads[1] != 1)) {
    string wrong_value =
      ConcatString(ksize[0], " and ", ksize[1], "and", strides[0], "and", strides[1], "and", pads[0], "and", pads[1]);
    std::string err_msg = GetAttrValueErrMsg(
      "ksize[0] and ksize[1] and strides[0] and strides[1] and pads[0] and pads[1]", wrong_value, ConcatString("1"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  if (data_format == "NDHWC" &&
      (ksize[0] != 1 || ksize[4] != 1 || strides[0] != 1 || strides[4] != 1 || pads[0] != 1 || pads[4] != 1)) {
    string wrong_value =
      ConcatString(ksize[0], " and ", ksize[4], "and", strides[0], "and", strides[4], "and", pads[0], "and", pads[4]);
    std::string err_msg = GetAttrValueErrMsg(
      "ksize[0] and ksize[4] and strides[0] and strides[4] and pads[0] and pads[4]", wrong_value, ConcatString("1"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_INFERFUNC(MaxUnpool3DGrad, MaxUnpool3DGradInferShape) {
  auto input_tensor_desc = op.GetInputDescByName("x");
  auto input_shape = input_tensor_desc.GetShape();
  std::string data_format;
  if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
    if (data_format != "NCDHW" && data_format != "NDHWC") {
      string expected_format_list = ConcatString("NCDHW, NDHWC");
      std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, data_format);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
      return GRAPH_FAILED;
    }
  }

  TensorDesc td = op.GetOutputDescByName("y");
  DataType input_dtype = input_tensor_desc.GetDataType();
  td.SetShape(input_shape);
  td.SetDataType(input_dtype);
  if (op.UpdateOutputDesc("y", td) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(MaxUnpool3DGrad, MaxUnpool3DGradInferShape);
CUST_VERIFY_FUNC_REG(MaxUnpool3DGrad, MaxUnpool3DGradVerify);
// ---------------------MaxUnpool3DGrad---------------------
}  // namespace ge
