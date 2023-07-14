/**
 * Copyright (c) 2022-202 Huawei Technologies Co., Ltd.  All rights reserved.
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

#include "inc/ops/image_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/image_ops_shape_fns.h"
namespace ge {
// ----------------AdjustHue Start-------------------
IMPLEMT_INFERFUNC(AdjustHue, AdjustHueInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto images_desc = op_desc->MutableInputDesc(0);

  GeShape out;
  auto ret = WithRankAtLeast(images_desc, 3, out, op);
  if (ret != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0, DebugString(images_desc->GetShape().GetDims()), "at least 3D");
    err_msg = string("failed to call WithRankAtLeast function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<std::pair<int64_t, int64_t>> range;
  if (images_desc->GetShapeRange(range) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  auto y_desc = op_desc->MutableOutputDesc(0);
  y_desc->SetShape(out);
  y_desc->SetShapeRange(range);
  y_desc->SetDataType(images_desc->GetDataType());

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AdjustHue, AdjustHueInfer);
// ----------------AdjustHue End-------------------

// ----------------AdjustSaturation Start-------------------
static graphStatus AdjustSaturationCommInferShape(const Operator &op) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto images_desc = op_desc->MutableInputDesc(0);

  GeShape out;
  auto ret = WithRankAtLeast(images_desc, 3, out, op);
  if (ret != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0, DebugString(images_desc->GetShape().GetDims()), "at least 3D");
    err_msg = string("failed to call WithRankAtLeast function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<std::pair<int64_t, int64_t>> range;
  if (images_desc->GetShapeRange(range) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  auto y_desc = op_desc->MutableOutputDesc(0);
  y_desc->SetShape(out);
  y_desc->SetShapeRange(range);
  y_desc->SetDataType(images_desc->GetDataType());

  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(AdjustSaturation, AdjustSaturationInfer) { return AdjustSaturationCommInferShape(op); }

INFER_FUNC_REG(AdjustSaturation, AdjustSaturationInfer);
// ----------------AdjustSaturation END-------------------

// ----------------ExtractGlimpse-------------------
IMPLEMT_INFERFUNC(ExtractGlimpse, ExtractGlimpseInfer) {
  Shape x_shape;
  auto ret = WithRank(op.GetInputDesc(0), 4, x_shape, op);
  if (ret != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op), "input x must be 4-D");
    return GRAPH_PARAM_INVALID;
  }
  Shape offsets_shape;
  ret = WithRank(op.GetInputDesc(2), 2, offsets_shape, op);
  if (ret != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op), "input offsets must be 2-D");
    return GRAPH_PARAM_INVALID;
  }
  auto x_dims = op.GetInputDesc(0).GetShape().GetDims();
  auto offsets_dims = op.GetInputDesc(2).GetShape().GetDims();
  CHECK(x_dims.size() < 4 || offsets_dims.size() < 2, OP_LOGE(TbeGetName(op), "invalid x_dims or offsets_dims."),
        return GRAPH_FAILED);
  int64_t batch_dim;
  if (Merge(x_dims[0], offsets_dims[0], batch_dim) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op), "x dim-0 or offsets dim-0 is invalid");
    return GRAPH_PARAM_INVALID;
  }
  if (offsets_dims[1] != 2) {
    OP_LOGE(TbeGetName(op), "offsets dim-1 must be 2");
    return GRAPH_PARAM_INVALID;
  }

  bool uniform_noise = false;
  if (op.GetAttr("uniform_noise", uniform_noise) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op), "get attr uniform_noise failed");
    return GRAPH_FAILED;
  }
  std::string noise;
  if (op.GetAttr("noise", noise) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op), "get attr noise failed");
    return GRAPH_FAILED;
  }
  if (uniform_noise && (!noise.empty() && noise != "uniform")) {
    OP_LOGE(TbeGetName(op), "The uniform_noise and noise should not be specified at the same time");
    return GRAPH_FAILED;
  }

  TensorDesc desc = op.GetOutputDescByName("y");
  desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  auto channel_dim = x_dims[3];
  TensorDesc input_td = op.GetInputDesc(0);
  if (static_cast<ge::Format>(ge::GetPrimaryFormat(input_td.GetFormat())) == FORMAT_NCHW) {
    channel_dim = x_dims[1];
  }
  return SetOutputToSizedImage(op, batch_dim, "size", channel_dim, "y");
}

INFER_FUNC_REG(ExtractGlimpse, ExtractGlimpseInfer);
// ----------------ExtractGlimpse-------------------
}  // namespace ge