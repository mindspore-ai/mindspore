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

#include "op_proto/inc/image_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/op_log.h"
#include "utils/common_shape_fns.h"

namespace ge {
IMPLEMT_INFERFUNC(CombinedNonMaxSuppression, CombinedNonMaxSuppressionInfer) {
  DYNAMIC_SHAPE_NOT_SUPPORTED(op);
  Shape boxes;
  Shape scores;
  Shape max_output_size_per_class;
  Shape max_total_size;
  Shape unused_shape;

  std::vector<std::string> input_infer_depends = {"max_total_size", "max_output_size_per_class"};
  PREPARE_DYNAMIC_SHAPE(input_infer_depends);

  if (WithRank(op.GetInputDesc(0), 4, boxes, op) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
                                      GetShapeErrMsg(0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "4D"));
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 3, scores, op) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
                                      GetShapeErrMsg(1, DebugString(op.GetInputDesc(1).GetShape().GetDims()), "3D"));
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(2), 0, max_output_size_per_class, op) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(
      TbeGetName(op), GetShapeErrMsg(2, DebugString(op.GetInputDesc(2).GetShape().GetDims()), "scalar"));
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(3), 0, max_total_size, op) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(
      TbeGetName(op), GetShapeErrMsg(3, DebugString(op.GetInputDesc(3).GetShape().GetDims()), "scalar"));
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(4), 0, unused_shape, op) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(
      TbeGetName(op), GetShapeErrMsg(4, DebugString(op.GetInputDesc(4).GetShape().GetDims()), "scalar"));
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(5), 0, unused_shape, op) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(
      TbeGetName(op), GetShapeErrMsg(5, DebugString(op.GetInputDesc(5).GetShape().GetDims()), "scalar"));
    return GRAPH_FAILED;
  }

  int64_t unused = 0;
  int64_t dim1 = boxes.GetDim(0);
  int64_t dim2 = scores.GetDim(0);
  if (Merge(dim1, dim2, unused) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
                                      ConcatString("call Merge function failed to merge 0th dim of input[boxes]"
                                                   " and input[scores], ",
                                                   dim1, " and ", dim2));
    return GRAPH_FAILED;
  }
  int64_t dim3 = boxes.GetDim(1);
  int64_t dim4 = scores.GetDim(1);
  if (Merge(dim3, dim4, unused) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
                                      ConcatString("call Merge function failed to merge 1th dim of input[boxes]"
                                                   " and input[scores], ",
                                                   dim3, " and ", dim4));
    return GRAPH_FAILED;
  }

  if (boxes.GetDim(3) != 4) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       ConcatString("invalid 3th dim value[", boxes.GetDim(3), "], it should be 4"));
    return GRAPH_FAILED;
  }

  Shape boxes_shape = op.GetInputDesc(0).GetShape();
  Shape scores_shape = op.GetInputDesc(1).GetShape();
  if (ValueKnown(boxes_shape, 2) && ValueKnown(scores_shape, 2)) {
    if (boxes_shape.GetDim(2) != 1 && boxes_shape.GetDim(2) != scores_shape.GetDim(2)) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), ConcatString("2th dim of input[boxes] and input[scores] are not equal, ", boxes_shape.GetDim(2),
                                     " and ", scores_shape.GetDim(2)));
      return GRAPH_FAILED;
    }
  }

  Tensor maxTotalSizeTensor;
  Tensor maxOutputSizePerClassTensor;
  if ((op.GetInputConstData("max_total_size", maxTotalSizeTensor) != GRAPH_SUCCESS) ||
      (op.GetInputConstData("max_output_size_per_class", maxOutputSizePerClassTensor) != GRAPH_SUCCESS)) {
    Shape out_shape0({-1, -1, 4});
    Shape out_shape1({-1, -1});
    Shape out_shape2({-1, -1});
    Shape out_shape3({-1});
    op.GetOutputDesc(0).SetShape(out_shape0);
    op.GetOutputDesc(1).SetShape(out_shape1);
    op.GetOutputDesc(2).SetShape(out_shape2);
    op.GetOutputDesc(3).SetShape(out_shape3);
    return GRAPH_SUCCESS;
  }
  int64_t maxTotalSize;
  if (MakeDimForScalarInput(maxTotalSizeTensor, maxTotalSize, op) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(
      TbeGetName(op), ConcatString("call MakeDimForScalarInput failed to get value from input[max_total_size] tensor"));
    return GRAPH_FAILED;
  }
  if (maxTotalSize <= 0) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
      TbeGetName(op), ConcatString("invalid value[", maxTotalSize, "] of input[max_total_size], should be > 0"));
    return GRAPH_FAILED;
  }

  int64_t maxOutputSizePerClass;
  if (MakeDimForScalarInput(maxOutputSizePerClassTensor, maxOutputSizePerClass, op) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(
      TbeGetName(op),
      ConcatString("call MakeDimForScalarInput failed to get value from input[max_output_size_per_class] tensor"));
    return GRAPH_FAILED;
  }

  int64_t output_size;
  bool pad_per_class;
  if (op.GetAttr("pad_per_class", pad_per_class) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), std::string("get attr[pad_per_class] failed"));
    return GRAPH_FAILED;
  }
  if (!pad_per_class) {
    output_size = maxTotalSize;
  } else {
    if (maxOutputSizePerClass <= 0) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op),
        ConcatString("invalid value[", maxOutputSizePerClass, "] of input[max_output_size_per_class], should be > 0"));
      return GRAPH_FAILED;
    }
    if (maxTotalSize <= maxOutputSizePerClass * scores_shape.GetDim(2)) {
      output_size = maxTotalSize;
    } else {
      output_size = maxOutputSizePerClass * scores_shape.GetDim(2);
    }
  }

  int64_t batch_dim = boxes.GetDim(0);
  Shape shape1({batch_dim, output_size, 4});
  Shape shape2({batch_dim, output_size});
  Shape shape3({batch_dim, output_size});
  Shape shape4({batch_dim});

  TensorDesc desc1 = op.GetOutputDescByName("nmsed_boxes");
  desc1.SetShape(shape1);
  desc1.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("nmsed_boxes", desc1) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), std::string("update output[nmsed_boxes] desc failed"));
    return GRAPH_FAILED;
  }
  TensorDesc desc2 = op.GetOutputDescByName("nmsed_scores");
  desc2.SetShape(shape2);
  desc2.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("nmsed_scores", desc2) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), std::string("update output[nmsed_scores] desc failed"));
    return GRAPH_FAILED;
  }
  TensorDesc desc3 = op.GetOutputDescByName("nmsed_classes");
  desc3.SetShape(shape3);
  desc3.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("nmsed_classes", desc3) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), std::string("update output[nmsed_classes] desc failed"));
    return GRAPH_FAILED;
  }
  TensorDesc desc4 = op.GetOutputDescByName("valid_detections");
  desc4.SetShape(shape4);
  desc4.SetDataType(DT_INT32);
  if (op.UpdateOutputDesc("valid_detections", desc4) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(),
                                       std::string("update output[valid_detections] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(CombinedNonMaxSuppression, CombinedNonMaxSuppressionInfer);
}  // namespace ge