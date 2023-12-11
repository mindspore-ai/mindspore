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

#include <graph/utils/type_utils.h>
#include "inc/ops/image_ops.h"
#include "register/op_impl_registry.h"
#include "utils/image_ops_shape_fns.h"
#include "utils/op_const.h"
#include "utils/util.h"
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
// ----------------ExtractGlimpse END-------------------

// ----------------ResizeArea-------------------
IMPLEMT_INFERFUNC(ResizeArea, ResizeAreaInfer) {
  TensorDesc desc = op.GetOutputDescByName("y");
  desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return ResizeShapeFn(op, "images", "size", "y");
}

INFER_FUNC_REG(ResizeArea, ResizeAreaInfer);
// ----------------ResizeArea END-------------------

// ----------------ResizeBicubic-------------------
IMPLEMT_INFERFUNC(ResizeBicubic, ResizeBicubicInfer) {
  TensorDesc x_desc = op.GetInputDescByName("images");
  TensorDesc y_desc = op.GetOutputDescByName("y");

  DataType data_type = x_desc.GetDataType();
  y_desc.SetDataType(data_type);
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return ResizeShapeFn(op, "images", "size", "y");
}

INFER_FUNC_REG(ResizeBicubic, ResizeBicubicInfer);
// ----------------ResizeBicubic END-------------------

bool ResizeConstInferShape(const Operator &op, const std::pair<uint32_t, std::string> image_info,
                           const std::pair<uint32_t, std::string> size_info,
                           const std::pair<uint32_t, std::string> output_info) {
  static const size_t output_len = 4;
  static const size_t size_len = 2;
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_desc == nullptr, OP_LOGE(TbeGetName(op), "op desc is null."), return false);

  auto input_desc_x = op_desc->MutableInputDesc(image_info.first);
  CHECK(input_desc_x == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), OtherErrMsg("input x is null.")), return false);
  auto output_desc_y = op_desc->MutableOutputDesc(output_info.first);
  CHECK(output_desc_y == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), OtherErrMsg("output y is null.")), return false);

  // infer dtype start
  output_desc_y->SetDataType(input_desc_x->GetDataType());
  // infer dtype end

  // infer shape start
  const GeShape &x_shape = input_desc_x->MutableShape();
  auto input_format = input_desc_x->GetFormat();
  OP_LOGD(TbeGetName(op), "get the format is %s", ge::TypeUtils::FormatToSerialString(input_format).c_str());
  CHECK(input_format != FORMAT_NHWC && input_format != FORMAT_NCHW,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), OtherErrMsg("The input format is valid")),
        return false);
  const int64_t image_n_idx = 0;
  // format is NHWC, c_idx = 3, format is NCHW, c_idx = 1,
  const int64_t image_c_idx = input_format == FORMAT_NHWC ? 3 : 1;
  const int64_t image_h_idx = input_format == FORMAT_NHWC ? 1 : 2;
  const int64_t image_w_idx = input_format == FORMAT_NHWC ? 2 : 3;
  // get const value
  bool is_size_const = true;
  vector<int64_t> size_out;
  if (!ops::GetConstIntData(op, size_info.first, size_out)) {
    OP_LOGW(TbeGetName(op).c_str(), "get const value of input size failed, set out hw = -1, -1");
    size_out = {-1, -1};
    is_size_const = false;
  }

  // the size num must be 2, mean output h, output w
  OP_LOGD(TbeGetName(op), "the size num must be 2. get the num is %zu", size_out.size());
  CHECK(size_out.size() != size_len,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), OtherErrMsg("the input size num must be 2.")),
        return false);

  // get y shape
  GeShape &y_shape = output_desc_y->MutableShape();
  y_shape.SetDimNum(output_len);
  if (!x_shape.IsUnknownDimNum()) {
    OP_LOGD(TbeGetName(op), "the input shape size must be 4. get shape size is %zu", x_shape.GetDimNum());
    CHECK(x_shape.GetDimNum() != output_len,
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), OtherErrMsg("The dim of input x is not 4")),
          return false);
    y_shape.SetDim(image_n_idx, x_shape.GetDim(image_n_idx));
    y_shape.SetDim(image_c_idx, x_shape.GetDim(image_c_idx));
  } else {
    OP_LOGW(TbeGetName(op).c_str(), "the input is unknown rank, will set the out nc = -1, -1");
    y_shape.SetDim(image_n_idx, -1);
    y_shape.SetDim(image_c_idx, -1);
  }
  y_shape.SetDim(image_h_idx, size_out[0]);
  y_shape.SetDim(image_w_idx, size_out[1]);
  // infer shape end

  // charge whether is dynamic, when output is static shape, return true
  CHECK(!y_shape.IsUnknownShape(), OP_LOGD(TbeGetName(op), "the output is static shape. infer succ"), return true);

  OP_LOGD(TbeGetName(op), "the output is dynamic shape. will infer range");
  // infer shape_range start
  std::vector<std::pair<int64_t, int64_t>> x_range;
  vector<int64_t> image_shape{-1, -1, -1, -1};
  // check whether is -2 case
  if (!x_shape.IsUnknownDimNum()) {
    image_shape = x_shape.GetDims();
    (void)input_desc_x->GetShapeRange(x_range);
  }
  MakeUpShapeRange(image_shape, x_range);
  OP_LOGD(TbeGetName(op), "the input range size must be 4. get size is %zu", x_range.size());
  CHECK(x_range.size() != output_len,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), OtherErrMsg("the x range size is not equal 4")),
        return false);
  if (!is_size_const) {
    std::vector<std::pair<int64_t, int64_t>> size_value_range;
    auto input_size_x = op_desc->MutableInputDesc(size_info.first);
    CHECK(input_size_x == nullptr,
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), OtherErrMsg("input size is null.")),
          return false);
    // means no const value, will get the value range
    (void)input_size_x->GetValueRange(size_value_range);
    // the size num must be 2, so the value range num must be 2
    if (size_value_range.size() != size_len) {
      x_range[image_h_idx] = std::pair<int64_t, int64_t>(0, -1);
      x_range[image_w_idx] = std::pair<int64_t, int64_t>(0, -1);
    } else {
      x_range[image_h_idx] = size_value_range[0];
      x_range[image_w_idx] = size_value_range[1];
    }
  } else {
    x_range[image_h_idx] = std::pair<int64_t, int64_t>(size_out[0], size_out[0]);
    x_range[image_w_idx] = std::pair<int64_t, int64_t>(size_out[1], size_out[1]);
  }

  output_desc_y->SetShapeRange(x_range);
  // infer shape_range end
  return true;
}

// ---------------ResizeNearestNeighborV2 Op Start-------------------
IMPLEMT_COMMON_INFERFUNC(ResizeNearestNeighborV2InferShape) {
  static const std::pair<uint32_t, std::string> input_x{0, "x"};
  static const std::pair<uint32_t, std::string> input_size{1, "size"};
  static const std::pair<uint32_t, std::string> output_y{0, "y"};
  const vector<string> depends{input_size.second};
  PREPARE_DYNAMIC_SHAPE(depends);
  if (!ResizeConstInferShape(op, input_x, input_size, output_y)) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(ResizeNearestNeighborV2, ResizeNearestNeighborV2InferShape);
INFER_VALUE_RANGE_DEFAULT_REG(ResizeNearestNeighborV2);
// ---------------ResizeNearestNeighborV2 Op End-------------------

// ----------------ResizeNearestNeighborV2Grad-------------------
IMPLEMT_INFERFUNC(ResizeNearestNeighborV2Grad, ResizeNearestNeighborV2GradInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto y_desc = op_desc->MutableOutputDesc(0);
  auto size_desc = op_desc->MutableInputDesc(1);
  auto grads_desc = op_desc->MutableInputDesc(0);
  if (op.GetInputDesc(0).GetShape().GetDims() == UNKNOWN_RANK ||
      op.GetInputDesc(1).GetShape().GetDims() == UNKNOWN_RANK) {
    y_desc->SetShape(GeShape(UNKNOWN_RANK));
    y_desc->SetDataType(grads_desc->GetDataType());
    return GRAPH_SUCCESS;
  }
  // unknown shape support
  std::vector<std::string> input_infer_depends = {"size"};
  op_desc->SetOpInferDepends(input_infer_depends);

  GeShape grads_shape;
  if (WithRank(grads_desc, 4, grads_shape, op) != GRAPH_SUCCESS) {
    OP_LOGE(op_desc->GetName().c_str(), "Input grads must be 4-D, real rank is [%lu]",
            grads_desc->GetShape().GetDimNum());
    return GRAPH_PARAM_INVALID;
  }

  GeShape size_shape;
  if (WithRank(size_desc, 1, size_shape, op) != GRAPH_SUCCESS) {
    OP_LOGE(op_desc->GetName().c_str(), "Input size must be 1-D, real rank is [%lu]",
            size_desc->GetShape().GetDimNum());
    return GRAPH_PARAM_INVALID;
  }

  auto size_dims = size_shape.GetDims();
  if (size_dims[0] != 2 && size_dims[0] != UNKNOWN_DIM) {
    OP_LOGE(op_desc->GetName().c_str(), "Input size must be 1-D of 2 elements, real dim size is [%ld]", size_dims[0]);
    return GRAPH_PARAM_INVALID;
  }

  int64_t size_height = UNKNOWN_DIM;
  int64_t size_width = UNKNOWN_DIM;
  Tensor size_tensor;
  if (op.GetInputConstData("size", size_tensor) == GRAPH_SUCCESS) {
    auto size_data = reinterpret_cast<const int32_t *>(size_tensor.GetData());
    if (size_data == nullptr) {
      OP_LOGE(op_desc->GetName().c_str(), "Get size data failed");
      return GRAPH_PARAM_INVALID;
    }
    size_height = static_cast<int64_t>(size_data[0]);
    size_width = static_cast<int64_t>(size_data[1]);
  }

  std::vector<int64_t> output_dims;
  auto grads_dims = grads_shape.GetDims();
  auto input_format = static_cast<ge::Format>(ge::GetPrimaryFormat(grads_desc->GetFormat()));
  if (input_format == FORMAT_NCHW) {
    output_dims.push_back(grads_dims[0]);
    output_dims.push_back(grads_dims[1]);
    output_dims.push_back(size_height);
    output_dims.push_back(size_width);
  } else if (input_format == FORMAT_NHWC) {
    output_dims.push_back(grads_dims[0]);
    output_dims.push_back(size_height);
    output_dims.push_back(size_width);
    output_dims.push_back(grads_dims[3]);
  } else {
    OP_LOGE(op_desc->GetName().c_str(), "Not supported this format: [%d]", input_format);
    return GRAPH_PARAM_INVALID;
  }
  GeShape output_shape(output_dims);
  if (ShapeFullyDefined(output_shape) == false) {
    std::vector<std::pair<int64_t, int64_t>> output_range;
    for (const int64_t &output_dim : output_dims) {
      output_range.push_back(output_dim == UNKNOWN_DIM ? std::pair<int64_t, int64_t>{1, -1}
                                                       : std::pair<int64_t, int64_t>{output_dim, output_dim});
    }
    y_desc->SetShapeRange(output_range);
  }
  y_desc->SetDataType(grads_desc->GetDataType());
  y_desc->SetShape(output_shape);

  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(ResizeNearestNeighborV2Grad, ResizeNearestNeighborV2GradInfer);
// ----------------ResizeNearestNeighborV2Grad END-------------------

// ----------------RGBToHSV-------------------
IMPLEMT_INFERFUNC(RGBToHSV, RGBToHSVInfer) {
  TensorDesc desc = op.GetOutputDescByName("y");
  desc.SetDataType(op.GetInputDesc(0).GetDataType());
  if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("update output[y] desc failed"));
    return GRAPH_FAILED;
  }
  return ColorspaceShapeFn(op, "y");
}

INFER_FUNC_REG(RGBToHSV, RGBToHSVInfer);
// ----------------RGBToHSV END-------------------

// ----------------NonMaxSuppressionWithOverlaps-------------------
IMPLEMT_INFERFUNC(NonMaxSuppressionWithOverlaps, NonMaxSuppressionWithOverlapsInfer) {
  Shape overlaps_shape = op.GetInputDescByName("overlaps").GetShape();
  Shape scores_shape = op.GetInputDescByName("scores").GetShape();
  Shape max_output_size_shape = op.GetInputDescByName("max_output_size").GetShape();
  Shape overlap_threshold_shape = op.GetInputDescByName("overlap_threshold").GetShape();
  Shape score_threshold_shape = op.GetInputDescByName("score_threshold").GetShape();
  if (WithRank(op.GetInputDescByName("overlaps"), 2, overlaps_shape, op) != GRAPH_SUCCESS) {
    std::string err_msg =
      ConcatString("failed to call WithRank function, ", "input[overlaps] rank must be 2, but got rank[",
                   op.GetInputDescByName("overlaps").GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDescByName("scores"), 1, scores_shape, op) != GRAPH_SUCCESS) {
    std::string err_msg =
      ConcatString("failed to call WithRank function, ", "input[scores] rank must be 1, but got rank[",
                   op.GetInputDescByName("scores").GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDescByName("max_output_size"), 0, max_output_size_shape, op) != GRAPH_SUCCESS) {
    std::string err_msg =
      ConcatString("failed to call WithRank function, ", "input[max_output_size] rank must be 0, but got rank[",
                   op.GetInputDescByName("max_output_size").GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDescByName("overlap_threshold"), 0, overlap_threshold_shape, op) != GRAPH_SUCCESS) {
    std::string err_msg =
      ConcatString("failed to call WithRank function, ", "input[overlap_threshold] rank must be 0, but got rank[",
                   op.GetInputDescByName("overlap_threshold").GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDescByName("score_threshold"), 0, score_threshold_shape, op) != GRAPH_SUCCESS) {
    std::string err_msg =
      ConcatString("failed to call WithRank function, ", "input[score_threshold] rank must be 0, but got rank[",
                   op.GetInputDescByName("score_threshold").GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  int64_t unused_dim = 0;
  if (Merge(overlaps_shape.GetDim(0), scores_shape.GetDim(0), unused_dim) != GRAPH_SUCCESS) {
    std::string err_msg =
      ConcatString("failed to call Merge function to merge the input[overlaps] 0th dim", "[", overlaps_shape.GetDim(0),
                   "] and the input[scores]'s 0th dim [", scores_shape.GetDim(0), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (Merge(overlaps_shape.GetDim(0), overlaps_shape.GetDim(1), unused_dim) != GRAPH_SUCCESS) {
    std::string err_msg =
      ConcatString("failed to call Merge function to merge the input[overlaps] 0th dim", "[", overlaps_shape.GetDim(0),
                   "] and the input[overlaps]'s 1th dim [", overlaps_shape.GetDim(1), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc selected_indices_desc = op.GetOutputDescByName("selected_indices");
  Shape selecte_indices_shape;
  Vector(ge::UNKNOWN_DIM, selecte_indices_shape);
  selected_indices_desc.SetDataType(DT_INT32);
  selected_indices_desc.SetShape(selecte_indices_shape);
  if (op.UpdateOutputDesc("selected_indices", selected_indices_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("update output[selected_indices] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(NonMaxSuppressionWithOverlaps, NonMaxSuppressionWithOverlapsInfer);
// ----------------NonMaxSuppressionWithOverlaps END-------------------

// ----------------ScaleAndTranslate-------------------
IMPLEMT_INFERFUNC(ScaleAndTranslate, ScaleAndTranslateInfer) {
  TensorDesc desc = op.GetOutputDescByName("y");
  desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("update description for output[y] failed."));
    return GRAPH_FAILED;
  }
  return ResizeShapeFn(op, "images", "size", "y");
}

INFER_FUNC_REG(ScaleAndTranslate, ScaleAndTranslateInfer);
// ----------------ScaleAndTranslate END-------------------

// ----------------ScaleAndTranslateGrad-------------------
IMPLEMT_INFERFUNC(ScaleAndTranslateGrad, ScaleAndTranslateGradInfer) {
  TensorDesc desc = op.GetOutputDescByName("y");
  Format input_format = static_cast<ge::Format>(ge::GetPrimaryFormat(op.GetInputDesc(0).GetFormat()));
  vector<int64_t> grads_shape = op.GetInputDesc(0).GetShape().GetDims();
  vector<int64_t> org_images_shape = op.GetInputDesc(1).GetShape().GetDims();
  vector<int64_t> y_shape;
  if (input_format == FORMAT_NHWC && grads_shape.size() > 3 && org_images_shape.size() > 2) {
    y_shape.push_back(grads_shape[0]);
    y_shape.push_back(org_images_shape[1]);
    y_shape.push_back(org_images_shape[2]);
    y_shape.push_back(grads_shape[3]);
  } else if (input_format == FORMAT_NCHW && grads_shape.size() > 1 && org_images_shape.size() > 3) {
    y_shape.push_back(grads_shape[0]);
    y_shape.push_back(grads_shape[1]);
    y_shape.push_back(org_images_shape[2]);
    y_shape.push_back(org_images_shape[3]);
  } else {
    if (grads_shape.size() < 4) {
      std::string err_msg =
        ConcatString("the 0th input[grads]'s rank should not be less than 4, ", "current rank is ", grads_shape.size());
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
    if (org_images_shape.size() < 2) {
      std::string err_msg = ConcatString("the 1th input[original_images]'s rank should not be less than 2, ",
                                         "current rank is ", org_images_shape.size());
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
    y_shape.push_back(grads_shape[0]);
    y_shape.push_back(org_images_shape[1]);
    y_shape.push_back(org_images_shape[2]);
    y_shape.push_back(grads_shape[3]);
    OP_LOGI(TbeGetName(op).c_str(), "Real format is %d", input_format);
  }

  desc.SetShape(ge::Shape(y_shape));
  desc.SetDataType(DT_FLOAT);
  return op.UpdateOutputDesc("y", desc);
}

INFER_FUNC_REG(ScaleAndTranslateGrad, ScaleAndTranslateGradInfer);
// ----------------ScaleAndTranslateGrad END-------------------

// ----------------ResizeBicubicGrad-------------------
IMPLEMT_INFERFUNC(ResizeBicubicGrad, ResizeBicubicGradInfer) {
  TensorDesc desc = op.GetOutputDescByName("y");
  Format input_format = static_cast<ge::Format>(ge::GetPrimaryFormat(op.GetInputDesc(0).GetFormat()));
  vector<int64_t> grads_shape = op.GetInputDesc(0).GetShape().GetDims();
  vector<int64_t> org_images_shape = op.GetInputDesc(1).GetShape().GetDims();
  vector<int64_t> y_shape;
  if (input_format == FORMAT_NHWC && grads_shape.size() > 3 && org_images_shape.size() > 2) {
    y_shape.push_back(grads_shape[0]);
    y_shape.push_back(org_images_shape[1]);
    y_shape.push_back(org_images_shape[2]);
    y_shape.push_back(grads_shape[3]);
  } else if (input_format == FORMAT_NCHW && grads_shape.size() > 1 && org_images_shape.size() > 3) {
    y_shape.push_back(grads_shape[0]);
    y_shape.push_back(grads_shape[1]);
    y_shape.push_back(org_images_shape[2]);
    y_shape.push_back(org_images_shape[3]);
  } else {
    std::string str_input_format = ge::TypeUtils::FormatToSerialString(input_format);
    std::string err_msg = ConcatString("only supporting NCHW and NHWC, current format is [", str_input_format, "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
  }
  desc.SetShape(ge::Shape(y_shape));
  auto type = op.GetInputDesc(1).GetDataType();
  desc.SetDataType(type);
  return op.UpdateOutputDesc("y", desc);
}

INFER_FUNC_REG(ResizeBicubicGrad, ResizeBicubicGradInfer);
// ----------------ResizeBicubicGrad END-------------------

graphStatus CropAndResizeInputRankCheck(const ge::Operator &op, const GeTensorDescPtr &x_desc, GeShape &x_shape,
                                        const GeTensorDescPtr &boxes_desc, GeShape &boxes_shape,
                                        const GeTensorDescPtr &box_index_desc, GeShape &box_index_shape,
                                        const GeTensorDescPtr &crop_size_desc, GeShape &crop_size_shape) {
  if (WithRank(x_desc, 4, x_shape, op) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0, DebugString(x_desc->GetShape().GetDims()), "4D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(boxes_desc, 2, boxes_shape, op) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1, DebugString(boxes_desc->GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(box_index_desc, 1, box_index_shape, op) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(2, DebugString(box_index_desc->GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(crop_size_desc, 1, crop_size_shape, op) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(3, DebugString(crop_size_desc->GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  auto x_dims = x_shape.GetDims();
  auto boxes_dims = boxes_shape.GetDims();
  auto box_index_dims = box_index_shape.GetDims();
  auto crop_size_dims = crop_size_shape.GetDims();
  CHECK(
    boxes_dims.empty() || box_index_dims.empty(),
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("the 0th input[x]'s shape and 1st input[boxes]'s shape"
                                                              " should not be empty.")),
    return GRAPH_FAILED);
  if (boxes_dims[0] != UNKNOWN_DIM && box_index_dims[0] != UNKNOWN_DIM && boxes_dims[0] != box_index_dims[0]) {
    std::string err_msg =
      ConcatString("the 0th dimension of the 1th input[boxes] and the 2nd input[box_index] must be equal. ",
                   boxes_dims[0], " and ", box_index_dims[0]);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  CHECK(crop_size_dims.empty(), AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), string("empty crop_size dim.")),
        return GRAPH_FAILED);
  if (crop_size_dims[0] != 2 && crop_size_dims[0] != UNKNOWN_DIM) {
    std::string err_msg = ConcatString(
      "the 3rd input[crop_size] must be a 1-D tensor containing 2 elements,"
      " current shape is ",
      DebugString(crop_size_dims));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
// ---------------CropAndResize Op Start-------------------
IMPLEMT_INFERFUNC(CropAndResize, CropAndResizeInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  // unknown shape support
  op_desc->SetOpInferDepends({"crop_size"});
  auto x_desc = op_desc->MutableInputDesc(0);
  GeShape x_shape;
  auto boxes_desc = op_desc->MutableInputDesc(1);
  GeShape boxes_shape;
  auto box_index_desc = op_desc->MutableInputDesc(2);
  GeShape box_index_shape;
  auto crop_size_desc = op_desc->MutableInputDesc(3);
  GeShape crop_size_shape;
  if (CropAndResizeInputRankCheck(op, x_desc, x_shape, boxes_desc, boxes_shape, box_index_desc, box_index_shape,
                                  crop_size_desc, crop_size_shape) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  auto x_dims = x_shape.GetDims();
  auto boxes_dims = boxes_shape.GetDims();
  int64_t crop_height = UNKNOWN_DIM;
  int64_t crop_width = UNKNOWN_DIM;
  Tensor crop_size_tensor;
  if (op.GetInputConstData("crop_size", crop_size_tensor) == GRAPH_SUCCESS) {
    auto size_data = reinterpret_cast<const int32_t *>(crop_size_tensor.GetData());
    crop_height = static_cast<int64_t>(size_data[0]);
    crop_width = static_cast<int64_t>(size_data[1]);
  }
  std::vector<int64_t> y_dims;
  Format input_format = static_cast<ge::Format>(ge::GetPrimaryFormat(op.GetInputDesc(0).GetFormat()));
  if (input_format == FORMAT_NHWC && x_dims.size() > 3) {
    y_dims.push_back(boxes_dims[0]);
    y_dims.push_back(crop_height);
    y_dims.push_back(crop_width);
    y_dims.push_back(x_dims[3]);
  } else if (input_format == FORMAT_NCHW && x_dims.size() > 1) {
    y_dims.push_back(boxes_dims[0]);
    y_dims.push_back(x_dims[1]);
    y_dims.push_back(crop_height);
    y_dims.push_back(crop_width);
  } else {
    std::string str_input_format = ge::TypeUtils::FormatToSerialString(input_format);
    std::string err_msg = ConcatString(
      "only supporting NCHW and NHWC, "
      "current format is [",
      str_input_format, "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  auto y_desc = op_desc->MutableOutputDesc(0);
  GeShape y_shape(y_dims);
  if (!ShapeFullyDefined(y_shape)) {
    std::vector<std::pair<int64_t, int64_t>> y_range;
    MakeUpShapeRange(y_dims, y_range);
    // boxes_dims[0] UNKNOWN_DIM
    if (y_dims[0] == UNKNOWN_DIM) {
      std::vector<std::pair<int64_t, int64_t>> boxes_range;
      boxes_desc->GetShapeRange(boxes_range);
      y_range[0] = boxes_range[0];
    }
    // NCHW x_dims[1] UNKNOWN_DIM
    if (input_format == FORMAT_NCHW && y_dims[AXIS_NCHW_DIM_C] == UNKNOWN_DIM) {
      std::vector<std::pair<int64_t, int64_t>> x_range;
      x_desc->GetShapeRange(x_range);
      y_range[AXIS_NCHW_DIM_C] = x_range[AXIS_NCHW_DIM_C];
    }
    // NHWC x_dims[3] UNKNOWN_DIM
    if (input_format == FORMAT_NHWC && y_dims[AXIS_NHWC_DIM_C] == UNKNOWN_DIM) {
      std::vector<std::pair<int64_t, int64_t>> x_range;
      x_desc->GetShapeRange(x_range);
      y_range[AXIS_NHWC_DIM_C] = x_range[AXIS_NHWC_DIM_C];
    }
    y_desc->SetShapeRange(y_range);
  }
  y_desc->SetShape(y_shape);
  y_desc->SetDataType(boxes_desc->GetDataType());
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(CropAndResize, CropAndResizeInfer);
// ----------------CropAndResize END-------------------
}  // namespace ge
