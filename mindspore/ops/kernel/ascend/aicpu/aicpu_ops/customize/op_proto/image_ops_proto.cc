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

#include "op_proto/inc/image_ops.h"
#include "custom_op_proto/cust_image_ops.h"
#include "register/op_impl_registry.h"
#include "utils/image_ops_shape_fns.h"
#include "utils/op_const.h"
#include "utils/util.h"
namespace ge {
// ----------------AdjustHue Start-------------------
IMPLEMT_INFERFUNC(AdjustHue, AdjustHueInfer) {
  auto images_desc = op.GetInputDesc(0);
  Shape out;
  auto ret = WithRankAtLeast(images_desc, 3, out, op);
  if (ret != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0, DebugString(images_desc.GetShape().GetDims()), "at least 3D");
    err_msg = string("failed to call WithRankAtLeast function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<std::pair<int64_t, int64_t>> range;
  if (images_desc.GetShapeRange(range) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  auto y_desc = op.GetOutputDesc(0);
  y_desc.SetShape(out);
  y_desc.SetShapeRange(range);
  y_desc.SetDataType(images_desc.GetDataType());
  op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AdjustHue, AdjustHueInfer);
// ----------------AdjustHue End-------------------

// ----------------AdjustSaturation Start-------------------
static graphStatus AdjustSaturationCommInferShape(Operator &op) {
  auto images_desc = op.GetInputDesc(0);

  Shape out;
  auto ret = WithRankAtLeast(images_desc, 3, out, op);
  if (ret != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0, DebugString(images_desc.GetShape().GetDims()), "at least 3D");
    err_msg = string("failed to call WithRankAtLeast function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<std::pair<int64_t, int64_t>> range;
  if (images_desc.GetShapeRange(range) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  auto y_desc = op.GetOutputDesc(0);
  y_desc.SetShape(out);
  y_desc.SetShapeRange(range);
  y_desc.SetDataType(images_desc.GetDataType());
  op.UpdateOutputDesc("y", y_desc);
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
  CHECK((x_dims.size() < 4 || offsets_dims.size() < 2), OP_LOGE(TbeGetName(op), "invalid x_dims or offsets_dims."),
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

graphStatus CropAndResizeInputRankCheck(const ge::Operator &op, const TensorDesc &x_desc, Shape &x_shape,
                                        const TensorDesc &boxes_desc, Shape &boxes_shape,
                                        const TensorDesc &box_index_desc, Shape &box_index_shape,
                                        const TensorDesc &crop_size_desc, Shape &crop_size_shape) {
  if (WithRank(x_desc, 4, x_shape, op) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0, DebugString(x_desc.GetShape().GetDims()), "4D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(boxes_desc, 2, boxes_shape, op) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1, DebugString(boxes_desc.GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(box_index_desc, 1, box_index_shape, op) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(2, DebugString(box_index_desc.GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(crop_size_desc, 1, crop_size_shape, op) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(3, DebugString(crop_size_desc.GetShape().GetDims()), "1D");
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
CUST_IMPLEMT_INFERFUNC(CropAndResize, CropAndResizeInfer) {
  // unknown shape support
  std::vector<std::string> depend_names{"crop_size"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  auto x_desc = op.GetInputDesc(0);
  Shape x_shape;
  auto boxes_desc = op.GetInputDesc(1);
  Shape boxes_shape;
  auto box_index_desc = op.GetInputDesc(2);
  Shape box_index_shape;
  auto crop_size_desc = op.GetInputDesc(3);
  Shape crop_size_shape;
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
    if (crop_size_desc.GetDataType() == DT_INT32) {
      auto size_data = reinterpret_cast<const int32_t *>(crop_size_tensor.GetData());
      crop_height = static_cast<int64_t>(size_data[0]);
      crop_width = static_cast<int64_t>(size_data[1]);
    } else {
      auto size_data = reinterpret_cast<const int64_t *>(crop_size_tensor.GetData());
      crop_height = size_data[0];
      crop_width = size_data[1];
    }
  }
  std::vector<int64_t> y_dims;
  Format input_format = static_cast<ge::Format>(ge::GetPrimaryFormat(op.GetInputDesc(0).GetFormat()));
  if (x_dims.size() == 4) {
    y_dims.push_back(boxes_dims[0]);
    y_dims.push_back(crop_height);
    y_dims.push_back(crop_width);
    y_dims.push_back(x_dims[3]);
  } else {
    std::string str_input_format = GeFormatToString(input_format);
    OP_LOGE(TbeGetName(op), "Only support 4D input.");
    return GRAPH_FAILED;
  }
  auto y_desc = op.GetOutputDesc(0);
  Shape y_shape(y_dims);
  if (!ShapeFullyDefined(y_shape)) {
    std::vector<std::pair<int64_t, int64_t>> y_range;
    MakeUpShapeRange(y_dims, y_range);
    // boxes_dims[0] UNKNOWN_DIM
    if (y_dims[0] == UNKNOWN_DIM) {
      std::vector<std::pair<int64_t, int64_t>> boxes_range;
      boxes_desc.GetShapeRange(boxes_range);
      y_range[0] = boxes_range[0];
    }
    if (y_dims[AXIS_NHWC_DIM_C] == UNKNOWN_DIM) {
      std::vector<std::pair<int64_t, int64_t>> x_range;
      x_desc.GetShapeRange(x_range);
      y_range[AXIS_NHWC_DIM_C] = x_range[AXIS_NHWC_DIM_C];
    }
    y_desc.SetShapeRange(y_range);
  }
  y_desc.SetShape(y_shape);
  y_desc.SetDataType(boxes_desc.GetDataType());
  op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(CropAndResize, CropAndResizeInfer);
// ----------------CropAndResize END-------------------

// ----------------CropAndResizeGradImage-------------------
CUST_IMPLEMT_INFERFUNC(CropAndResizeGradImage, CropAndResizeGradImageInfer) {
  std::vector<std::string> input_infer_depends = {"image_size"};
  PREPARE_DYNAMIC_SHAPE(input_infer_depends);

  auto grads_desc = op.GetInputDesc(0);
  Shape grads_shape;
  if (WithRank(grads_desc, 4, grads_shape, op) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0, DebugString(grads_desc.GetShape().GetDims()), "4D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    OP_LOGE(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto boxes_desc = op.GetInputDesc(1);
  Shape boxes_shape;
  if (WithRank(boxes_desc, 2, boxes_shape, op) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1, DebugString(boxes_desc.GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    OP_LOGE(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto box_index_desc = op.GetInputDesc(2);
  Shape box_index_shape;
  if (WithRank(box_index_desc, 1, box_index_shape, op) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(2, DebugString(box_index_desc.GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    OP_LOGE(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto image_size_desc = op.GetInputDesc(3);
  Shape image_size_shape;
  if (WithRank(image_size_desc, 1, image_size_shape, op) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(3, DebugString(image_size_desc.GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    OP_LOGE(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto grads_dims = grads_shape.GetDims();
  auto boxes_dims = boxes_shape.GetDims();
  auto box_index_dims = box_index_shape.GetDims();
  CHECK(grads_dims.empty() || boxes_dims.empty() || box_index_dims.empty(),
        OP_LOGE(TbeGetName(op), string("the 0th input[grads] , the 1st input[boxes] dims and the 2nd input[box_index], "
                                       "must not be empty.")),
        return GRAPH_FAILED);
  if (!DimsAllEqualOrUnknown({grads_dims[0], boxes_dims[0], box_index_dims[0]})) {
    std::string err_msg = ConcatString(
      "the 0th dimension of the 0th input[grads], the 1st input[boxes]"
      " and the 2nd input[box_index] must be equal. ",
      grads_dims[0], ", ", boxes_dims[0], " and ", box_index_dims[0]);
    OP_LOGE(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto image_size_dims = image_size_shape.GetDims();
  CHECK(image_size_dims.empty(),
        AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("the 3rd input[image_size] dims must not be empty.")),
        return GRAPH_FAILED);
  if (image_size_dims[0] != 4 && image_size_dims[0] != UNKNOWN_DIM) {
    std::string err_msg =
      ConcatString("the 3rd input[image_size] must be a 1-D tensor with 4 elements, current image_size is ",
                   DebugString(image_size_dims));
    return GRAPH_FAILED;
  }

  DataType type;
  if (op.GetAttr("T", type) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op), string("get attr[T] failed"));
    return GRAPH_FAILED;
  }

  int64_t batch = UNKNOWN_DIM;
  int64_t image_height = UNKNOWN_DIM;
  int64_t image_width = UNKNOWN_DIM;
  int64_t depth = UNKNOWN_DIM;
  Tensor image_size_tensor;
  if (op.GetInputConstData("image_size", image_size_tensor) == GRAPH_SUCCESS) {
    const int32_t *size_data = reinterpret_cast<const int32_t *>(image_size_tensor.GetData());
    CHECK(image_size_tensor.GetSize() / sizeof(int32_t) < 4,
          OP_LOGE(TbeGetName(op), string("the 3rd input[image_size]'s data nums less then 4, current data num is ",
                                         image_size_tensor.GetSize() / sizeof(int32_t))),
          return GRAPH_FAILED);
    batch = static_cast<int64_t>(size_data[0]);
    image_height = static_cast<int64_t>(size_data[1]);
    image_width = static_cast<int64_t>(size_data[2]);
    depth = static_cast<int64_t>(size_data[3]);
  }

  std::vector<int64_t> y_dims;
  y_dims.push_back(batch);
  y_dims.push_back(image_height);
  y_dims.push_back(image_width);
  y_dims.push_back(depth);

  auto y_desc = op.GetOutputDesc(0);
  Shape y_shape(y_dims);
  if (!ShapeFullyDefined(y_shape)) {
    std::vector<std::pair<int64_t, int64_t>> y_range;
    for (const int64_t &y_dim : y_dims) {
      y_range.push_back(y_dim == UNKNOWN_DIM ? std::pair<int64_t, int64_t>{1, -1}
                                             : std::pair<int64_t, int64_t>{y_dim, y_dim});
    }
    y_desc.SetShapeRange(y_range);
  }
  y_desc.SetShape(y_shape);
  y_desc.SetDataType(type);
  op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(CropAndResizeGradImage, CropAndResizeGradImageInfer);
// ----------------CropAndResizeGradImage-------------------

// ----------------CropAndResizeGradBoxes-------------------
CUST_IMPLEMT_INFERFUNC(CropAndResizeGradBoxes, CropAndResizeGradBoxesInfer) {
  Shape shape;
  auto ret = WithRank(op.GetInputDesc(0), 4, shape, op);
  if (ret != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "4D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    OP_LOGE(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  ret = WithRank(op.GetInputDesc(1), 4, shape, op);
  if (ret != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1, DebugString(op.GetInputDesc(1).GetShape().GetDims()), "4D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    OP_LOGE(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  ret = WithRank(op.GetInputDesc(2), 2, shape, op);
  if (ret != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(2, DebugString(op.GetInputDesc(2).GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    OP_LOGE(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  ret = WithRank(op.GetInputDesc(3), 1, shape, op);
  if (ret != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(3, DebugString(op.GetInputDesc(3).GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    OP_LOGE(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto grads_shape = op.GetInputDesc(0).GetShape().GetDims();
  auto boxes_shape = op.GetInputDesc(2).GetShape().GetDims();
  auto box_index_shape = op.GetInputDesc(3).GetShape().GetDims();

  if (grads_shape[0] != boxes_shape[0] && boxes_shape[0] != box_index_shape[0]) {
    std::string err_msg = ConcatString(
      "the 0th dimension of the 2th input[boxes], 0th input[grads] and the 3rd"
      " input [box_index] must be equal. ",
      grads_shape[0], ", ", boxes_shape[0], " and ", box_index_shape[0]);
    OP_LOGE(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc desc = op.GetOutputDescByName("y");
  desc.SetShape(op.GetInputDesc(2).GetShape());
  desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(CropAndResizeGradBoxes, CropAndResizeGradBoxesInfer);
// ----------------CropAndResizeGradBoxes-------------------
}  // namespace ge
