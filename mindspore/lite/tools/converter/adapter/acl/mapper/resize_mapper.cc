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

#include "tools/converter/adapter/acl/mapper/resize_mapper.h"
#include <memory>
#include <vector>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/adapter/acl/common/utils.h"
#include "ops/op_utils.h"
#include "src/common/log_util.h"
#include "mindspore/core/ops/op_name.h"

namespace mindspore {
namespace lite {
namespace {
constexpr auto kNameInputNum = 3;
constexpr auto kShapeFirstIdx = 0;
constexpr auto kShapeSecondIdx = 1;
constexpr auto kShapeThirdIdx = 2;
constexpr auto kShapeForthIdx = 3;
constexpr auto kNameSizeTwo = 2;
constexpr auto kNameSizeFour = 4;
}  // namespace

STATUS ResizeMapper::Mapper(const CNodePtr &cnode) {
  if (cnode->inputs().size() != kNameInputNum) {
    MS_LOG(WARNING) << "Input of resize must be " << kNameInputNum << ", real size: " << cnode->inputs().size()
                    << ", cnode " << cnode->fullname_with_scope();
    return lite::RET_OK;
  }
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get primitive from cnode failed, cnode " << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  if (ProcScaleInput(cnode, src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Proc scale input failed, cnode " << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  auto val_ptr = src_prim->GetAttr(ops::kMethod);
  CHECK_NULL_RETURN(val_ptr);
  int64_t method = GetValue<int64_t>(val_ptr);
  PrimitivePtr dst_prim = nullptr;
  if (method == static_cast<int64_t>(mindspore::ResizeMethod::NEAREST)) {
    dst_prim = std::make_shared<acl::ResizeNearestNeighborV2>();
  } else if (method == static_cast<int64_t>(mindspore::ResizeMethod::LINEAR)) {
    dst_prim = std::make_shared<acl::ResizeBilinearV2>();
  } else {
    MS_LOG(ERROR) << "Not support resize method " << method << ", cnode " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  CHECK_NULL_RETURN(dst_prim);
  auto pytorch_half_pixel_ptr = src_prim->GetAttr("coordinate_transform_mode");
  if (pytorch_half_pixel_ptr != nullptr &&
      GetValue<int64_t>(pytorch_half_pixel_ptr) == mindspore::CoordinateTransformMode::HALF_PIXEL) {
    dst_prim->set_attr("half_pixel_centers", MakeValue(true));
  }
  dst_prim->SetAttrs(src_prim->attrs());
  value_node->set_value(dst_prim);
  return RET_OK;
}

// success to get resize size, first infer shape must be done;
STATUS ResizeMapper::ProcScaleInput(const CNodePtr &cnode, const PrimitivePtr &prim) {
  TypeId type_id;
  auto scale_input = cnode->inputs()[kNameInputNum - 1];
  if (opt::GetDataTypeFromAnfNode(scale_input, &type_id) != RET_OK) {
    MS_LOG(ERROR) << "Get input of resize data type failed.";
    return RET_ERROR;
  }
  int64_t node_format = Format::NCHW;
  if (prim->GetAttr(ops::kFormat) != nullptr) {
    node_format = GetValue<int64_t>(prim->GetAttr(ops::kFormat));
  }
  if (type_id == kNumberTypeFloat32) {
    std::vector<int64_t> shape_vector;
    if (acl::GetShapeVectorFromCNode(cnode, &shape_vector) != RET_OK) {
      MS_LOG(ERROR) << "Get shape of cnode failed.";
      return RET_ERROR;
    }
    int32_t new_height;
    int32_t new_width;
    if (shape_vector.size() == kNameSizeTwo) {
      new_height = static_cast<int32_t>(shape_vector[kShapeFirstIdx]);
      new_width = static_cast<int32_t>(shape_vector[kShapeSecondIdx]);
    } else if (shape_vector.size() == kNameSizeFour) {
      if (node_format == Format::NHWC) {
        new_height = static_cast<int32_t>(shape_vector[kShapeSecondIdx]);
        new_width = static_cast<int32_t>(shape_vector[kShapeThirdIdx]);
      } else {
        new_height = static_cast<int32_t>(shape_vector[kShapeThirdIdx]);
        new_width = static_cast<int32_t>(shape_vector[kShapeForthIdx]);
      }
    } else {
      MS_LOG(ERROR) << "Size of shape must be two or four, real size: " << shape_vector.size();
      return RET_ERROR;
    }
    std::vector<int32_t> new_tensor_size = {new_height, new_width};
    auto func_graph = cnode->func_graph();
    auto param_node =
      opt::BuildIntVecParameterNode(func_graph, new_tensor_size, cnode->fullname_with_scope() + "_resize_shape");
    cnode->set_input(kNameInputNum - 1, param_node);
  }
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameResize, ResizeMapper)
}  // namespace lite
}  // namespace mindspore
