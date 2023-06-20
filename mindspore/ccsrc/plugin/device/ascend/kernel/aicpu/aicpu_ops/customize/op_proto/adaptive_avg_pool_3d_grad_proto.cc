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

#include "inc/adaptive_avg_pool_3d_grad_op.h"
#include "register/op_impl_registry.h"
#include "external/graph/operator_reg.h"
#include "utils/util.h"

namespace ge {
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
  map<int, std::string> format2str = {
    {ge::FORMAT_NCHW, "NCHW"},   {ge::FORMAT_NHWC, "NHWC"},   {ge::FORMAT_HWCN, "HWCN"},  {ge::FORMAT_DHWNC, "DHWNC"},
    {ge::FORMAT_DHWCN, "DHWCN"}, {ge::FORMAT_NDHWC, "NDHWC"}, {ge::FORMAT_NCDHW, "NCDHW"}};

  auto input_desc = op.GetInputDescByName("input_grad");
  auto orig_input_shape_desc = op.GetInputDescByName("orig_input_shape");
  TensorDesc out_desc = op.GetOutputDescByName("output_grad");
  ge::AscendString op_name;
  (void)op.GetName(op_name);

  // update format
  Format input_format = input_desc.GetFormat();
  std::string format_str = format2str[input_format];
  if (input_format != FORMAT_NCHW) {
    OP_LOGE("AdaptiveAvgPool3dGrad",
            "Input format only support NCHW"
            ", input format is [%s]",
            format_str.c_str());
    return GRAPH_FAILED;
  }
  out_desc.SetFormat(input_format);

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
}  // namespace ge