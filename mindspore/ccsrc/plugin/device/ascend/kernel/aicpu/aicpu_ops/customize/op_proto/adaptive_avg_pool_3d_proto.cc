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

#include "inc/adaptive_avg_pool_3d_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
// --------- AdaptiveAvgPool3d ---------------
IMPLEMT_COMMON_INFERFUNC(AdaptiveAvgPool3dInferShape) {
  map<int, std::string> format2str = {
    {ge::FORMAT_NCHW, "NCHW"},   {ge::FORMAT_NHWC, "NHWC"},   {ge::FORMAT_HWCN, "HWCN"},  {ge::FORMAT_DHWNC, "DHWNC"},
    {ge::FORMAT_DHWCN, "DHWCN"}, {ge::FORMAT_NDHWC, "NDHWC"}, {ge::FORMAT_NCDHW, "NCDHW"}};

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

  // update format
  Format input_format = input_desc.GetFormat();
  std::string format_str = format2str[input_format];
  if (input_format != FORMAT_NCHW) {
    OP_LOGE("AdaptiveAvgPool3d",
            "Input format only support NCHW"
            ", input format is [%s]",
            format_str.c_str());
    return GRAPH_FAILED;
  }
  out_desc.SetFormat(input_format);

  std::vector<int64_t> input_size_shape = input_desc.GetShape().GetDims();
  auto input_size_dim_num = input_size_shape.size();
  std::vector<int64_t> output_shape(input_size_shape.begin(), input_size_shape.end());
  auto output_size_num = output_size.size();
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
}  // namespace ge