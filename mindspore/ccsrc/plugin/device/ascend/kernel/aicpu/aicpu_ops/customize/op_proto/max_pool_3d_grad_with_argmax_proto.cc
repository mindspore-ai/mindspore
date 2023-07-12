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

#include "inc/max_pool_3d_grad_with_argmax_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
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

  int dtype = 0;
  if (GRAPH_SUCCESS != op.GetAttr("dtype", dtype)) {
    std::string err_msg = GetInputInvalidErrMsg("dtype");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);
  auto grads_desc = op_desc->MutableInputDesc("grads");
  CHECK_PTR_NULL(grads_desc, "grads desc", return GRAPH_FAILED);
  vector<int64_t> grads_shape = grads_desc->MutableShape().GetDims();
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
}  // namespace ge