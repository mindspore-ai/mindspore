/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "checker/resize_checker.h"
#include <vector>
#include <string>
#include "common/op_attr.h"
#include "common/check_base.h"
#include "common/op_enum.h"
#include "common/anf_util.h"
#include "mindapi/base/types.h"
#include "include/registry/converter_context.h"

namespace mindspore {
namespace dpico {
namespace {
constexpr int kMaxOutputWOf4Dims = 2048;
bool IsFromCaffe(const api::PrimitivePtr &primitive) {
  if (primitive->GetAttr(ops::kFmkType) != nullptr) {
    auto fmk_type = static_cast<converter::FmkType>(api::GetValue<int64_t>(primitive->GetAttr(ops::kFmkType)));
    if (fmk_type == converter::kFmkTypeCaffe) {
      return true;
    }
  }
  return false;
}
bool CheckInterpOp(const api::CNodePtr &op, const ShapeVector &input_shape, const ShapeVector &output_shape,
                   size_t index_h, size_t index_w) {
  auto prim = api::GetValueNode<api::PrimitivePtr>(op->input(0));
  MS_CHECK_TRUE_MSG(prim != nullptr, false, "prim is nullptr.");
  if (input_shape.at(index_w) > kMaxInputWOf4Dims || output_shape.at(index_w) >= kMaxOutputWOf4Dims) {
    MS_LOG(WARNING) << op->fullname_with_scope() << "'s input_w should be less than " << kMaxInputWOf4Dims
                    << " and output_w should be less than " << kMaxOutputWOf4Dims;
    return false;
  }
  int64_t pad_beg = 0;
  int64_t pad_end = 0;
  if (prim->GetAttr(dpico::kPadBeg) != nullptr) {
    pad_beg = static_cast<int>(api::GetValue<int64_t>(prim->GetAttr(dpico::kPadBeg)));
  }
  if (prim->GetAttr(dpico::kPadEnd) != nullptr) {
    pad_end = api::GetValue<int64_t>(prim->GetAttr(dpico::kPadEnd));
  }
  if (pad_beg > 0 || pad_end > 0) {
    MS_LOG(WARNING) << "pad_beg or pad_end only supports non-negative integer by dpico. " << op->fullname_with_scope();
    return false;
  }
  int64_t valid_input_h = input_shape.at(index_h) + pad_beg + pad_end;
  int64_t valid_input_w = input_shape.at(index_w) + pad_beg + pad_end;
  int64_t output_h = output_shape.at(index_h);
  int64_t output_w = output_shape.at(index_w);
  if (output_h == 0) {
    MS_LOG(WARNING) << "output_h should not be 0";
    return false;
  }
  if (output_w == 0) {
    MS_LOG(WARNING) << "output_w should not be 0";
    return false;
  }
  auto ratio_h = valid_input_h / output_h;
  auto ratio_w = valid_input_w / output_w;
  return ((ratio_h == 1) && (ratio_w == 1)) || ((ratio_h != 1) && (ratio_w != 1));
}
bool IsDoubleResize(const api::CNodePtr &, const ShapeVector &input_shape, const ShapeVector &output_shape,
                    size_t index_h, size_t index_w) {
  const int64_t nums2 = 2;
  return input_shape.at(index_h) * nums2 == output_shape.at(index_h) &&
         input_shape.at(index_w) * nums2 == output_shape.at(index_w);
}
bool CheckResizeOp(const api::CNodePtr &op, const ShapeVector &input_shape, const ShapeVector &output_shape,
                   size_t index_h, size_t index_w) {
  auto prim = api::GetValueNode<api::PrimitivePtr>(op->input(0));
  MS_CHECK_TRUE_MSG(prim != nullptr, false, "prim is nullptr.");
  if (prim->GetAttr(ops::kCoordinateTransformMode) != nullptr) {
    auto coordinate_transform_mode =
      static_cast<CoordinateTransformMode>(api::GetValue<int64_t>(prim->GetAttr(ops::kCoordinateTransformMode)));
    if (coordinate_transform_mode != CoordinateTransformMode::ASYMMETRIC) {
      MS_LOG(WARNING) << "resize only supports CoordinateTransformMode::ASYMMETRIC by dpico. "
                      << op->fullname_with_scope();
      return false;
    }
  }
  if (prim->GetAttr(ops::kMethod) != nullptr) {
    auto interpolation_mode = static_cast<ResizeMethod>(api::GetValue<int64_t>(prim->GetAttr(ops::kMethod)));
    if (interpolation_mode != ResizeMethod::NEAREST) {
      MS_LOG(WARNING) << "resize only supports ResizeMethod::NEAREST by dpico. " << op->fullname_with_scope();
      return false;
    }
  }
  if (prim->GetAttr(ops::kNearestMode) != nullptr) {
    auto nearest_mode = static_cast<mindspore::NearestMode>(api::GetValue<int64_t>(prim->GetAttr(ops::kNearestMode)));
    if (nearest_mode != mindspore::NearestMode::FLOOR) {
      MS_LOG(WARNING) << "resize only supports NearestMode::FLOOR by dpico. " << op->fullname_with_scope();
      return false;
    }
  } else {
    MS_LOG(WARNING) << ops::kNearestMode << " attr is needed. " << op->fullname_with_scope();
    return false;
  }
  if (!IsDoubleResize(op, input_shape, output_shape, index_h, index_w)) {
    MS_LOG(WARNING) << "resize only supports doubling expansion of [h,w] by dpico. " << op->fullname_with_scope();
    return false;
  }
  return true;
}
}  // namespace
bool ResizeChecker::Check(api::CNodePtr op, int32_t output_num, mindspore::Format) {
  auto primitive = api::GetValueNode<api::PrimitivePtr>(op->input(0));
  MS_CHECK_TRUE_MSG(primitive != nullptr, false, "prim is nullptr.");
  ShapeVector input_shape;
  auto abstract = GetCNodeInputAbstract(op, 1);
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "get cnode input abstract failed.";
    return false;
  }
  if (FetchShapeFromAbstract(abstract, &input_shape) != RET_OK) {
    MS_LOG(ERROR) << "fetch input shape failed. " << op->fullname_with_scope();
    return false;
  }
  if (input_shape.size() != kDims4) {
    MS_LOG(ERROR) << "resize cnode input shape should be 4 dims. " << op->fullname_with_scope();
    return false;
  }

  std::vector<ShapeVector> output_shapes;
  if (GetOutputShapesFromCNode(op, &output_shapes) != RET_OK) {
    MS_LOG(ERROR) << "get node shape failed";
    return false;
  }
  if (output_shapes.size() != 1) {
    MS_LOG(ERROR) << "resize should have single output, but in fact it has " << output_shapes.size() << " outputs.";
    return false;
  }
  auto output_shape = output_shapes.at(0);

  Format input_format;
  if (primitive->GetAttr(ops::kFormat) != nullptr) {
    input_format = static_cast<Format>(api::GetValue<int64_t>(primitive->GetAttr(ops::kFormat)));
  } else {
    MS_LOG(ERROR) << ops::kFormat << " attr is needed.";
    return false;
  }
  size_t index_h;
  size_t index_w;
  if (input_format == Format::NHWC) {
    index_h = 1;
    index_w = kAxis2;
  } else if (input_format == Format::NCHW) {
    index_h = kAxis2;
    index_w = kAxis3;
  } else {
    MS_LOG(ERROR) << "resize cnode input format is invalid. " << op->fullname_with_scope();
    return false;
  }

  if (IsFromCaffe(primitive)) {
    return CheckInterpOp(op, input_shape, output_shape, index_h, index_w);
  } else {
    return CheckResizeOp(op, input_shape, output_shape, index_h, index_w);
  }
}

OpCheckerRegistrar g_ResizeChecker("Resize", new ResizeChecker());
}  // namespace dpico
}  // namespace mindspore
