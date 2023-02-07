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
#include "plugin/device/cpu/kernel/nnacl/op_base.h"

namespace mindspore {
namespace lite {
namespace {
constexpr auto kNameInputNum = 3;
constexpr auto kResizeDataInputIndex = 1;
constexpr auto kResizeShapeInputIndex = 2;
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
      MS_LOG(INFO) << "Failed to get shape of resize node " << cnode->fullname_with_scope()
                   << ", shape value size: " << shape_vector.size();
      return CalResizeShape(cnode, prim);
    }
    std::vector<int32_t> new_tensor_size = {new_height, new_width};
    auto func_graph = cnode->func_graph();
    auto param_node = opt::BuildIntVecParameterNode(func_graph, new_tensor_size, scale_input->fullname_with_scope());
    cnode->set_input(kNameInputNum - 1, param_node);
  }
  return lite::RET_OK;
}

STATUS ResizeMapper::CalResizeShape(const CNodePtr &cnode, const PrimitivePtr &prim) {
  auto data_input = cnode->input(kResizeDataInputIndex);
  CHECK_NULL_RETURN(data_input);
  auto scale_input = cnode->input(kResizeShapeInputIndex);
  CHECK_NULL_RETURN(scale_input);
  auto func_graph = cnode->func_graph();
  CHECK_NULL_RETURN(func_graph);
  auto manager = func_graph->manager();
  CHECK_NULL_RETURN(manager);
  auto shape_node = NewCNode(cnode, prim::kPrimShape, {data_input}, {DIMENSION_4D}, kNumberTypeInt32,
                             cnode->fullname_with_scope() + "_shape_shape");
  if (!shape_node) {
    MS_LOG(ERROR) << "Failed to create shape node for node " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  std::vector<int> gather_indices = {kNCHW_H, kNCHW_W};  // fetch H and W dimension
  auto gather_cnode =
    opt::GenGatherNode(func_graph, shape_node, gather_indices, cnode->fullname_with_scope() + "_shape_gather");
  if (gather_cnode == nullptr) {
    MS_LOG(ERROR) << "Failed to create gather node for node " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  auto gather_shape = std::make_shared<abstract::Shape>(ShapeVector{DIMENSION_2D});
  auto gather_abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(kNumberTypeInt32), gather_shape);
  gather_cnode->set_abstract(gather_abstract);

  auto cast_fp32_node = NewCNode(cnode, prim::kPrimCast, {gather_cnode, NewValueNode(TypeIdToType(kNumberTypeFloat32))},
                                 {DIMENSION_2D}, kNumberTypeFloat32, cnode->fullname_with_scope() + "_shape_cast_fp32");
  if (!cast_fp32_node) {
    MS_LOG(ERROR) << "Failed to create cast node for node " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  auto mul_node = NewCNode(cnode, prim::kPrimMul, {cast_fp32_node, scale_input}, {DIMENSION_2D}, kNumberTypeFloat32,
                           cnode->fullname_with_scope() + "_shape_mul");
  if (!mul_node) {
    MS_LOG(ERROR) << "Failed to create mul node for node " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  auto cast_int32_node = NewCNode(cnode, prim::kPrimCast, {mul_node, NewValueNode(TypeIdToType(kNumberTypeInt32))},
                                  {DIMENSION_2D}, kNumberTypeInt32, cnode->fullname_with_scope() + "_shape_cast_int32");
  manager->Replace(scale_input, cast_int32_node);
  return RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameResize, ResizeMapper)
}  // namespace lite
}  // namespace mindspore
