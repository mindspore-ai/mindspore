/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/tbe/tbe_dynaminc_shape_util.h"
#include <memory>
#include <string>
#include <utility>
#include <map>
#include <algorithm>
#include <vector>
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace kernel {
namespace tbe {
bool TbeDynamicShapeUtil::IsDynamicShapeNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto input_num = AnfAlgo ::GetInputTensorNum(cnode);
  for (size_t i = 0; i < input_num; ++i) {
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(cnode, i);
    if (std::any_of(input_shape.begin(), input_shape.end(), [](const size_t &dim) { return dim < 0; })) {
      MS_LOG(INFO) << "Node(" << cnode->fullname_with_scope() << ") is dynamic shape node.";
      return true;
    }
  }
  auto output_num = AnfAlgo ::GetOutputTensorNum(cnode);
  for (size_t i = 0; i < output_num; ++i) {
    auto output_shape = AnfAlgo::GetOutputInferShape(cnode, i);
    if (std::any_of(output_shape.begin(), output_shape.end(), [](const size_t &dim) { return dim < 0; })) {
      MS_LOG(INFO) << "Node(" << cnode->fullname_with_scope() << ") is dynamic shape node.";
      return true;
    }
  }
  return false;
}

bool TbeDynamicShapeUtil::IsDynamicShapeNode(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (anf_node->isa<CNode>()) {
    auto cnode = anf_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    return IsDynamicShapeNode(cnode);
  }
  return false;
}

void TbeDynamicShapeUtil::SetDynamicShapeAttr(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto is_dyanmic_shape = IsDynamicShapeNode(cnode);
  AnfAlgo::SetNodeAttr(kAttrIsDynamicShape, MakeValue(is_dyanmic_shape), cnode);
}

bool TbeDynamicShapeUtil::GetDynamicShapeAttr(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (anf_node->isa<CNode>()) {
    auto cnode = anf_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    return GetDynamicShapeAttr(cnode);
  }
  return false;
}

bool TbeDynamicShapeUtil::GetDynamicShapeAttr(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto is_dynamic_shape = AnfAlgo::HasNodeAttr(kAttrIsDynamicShape, cnode);
  if (!is_dynamic_shape) {
    return false;
  }
  is_dynamic_shape = AnfAlgo::GetNodeAttr<bool>(cnode, kAttrIsDynamicShape);
  return is_dynamic_shape;
}

std::shared_ptr<OpInfo> TbeDynamicShapeUtil::FindOp(const std::string &op_name, const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (anf_node->isa<CNode>()) {
    auto cnode = anf_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    return FindOp(op_name, cnode);
  }
  return nullptr;
}

std::shared_ptr<OpInfo> TbeDynamicShapeUtil::FindOp(const std::string &op_name, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto is_dynamic_shape = GetDynamicShapeAttr(cnode);
  return mindspore::kernel::OpLib::FindOp(op_name, OpImplyType::kTBE, is_dynamic_shape);
}

RangePair TbeDynamicShapeUtil::GetInputDynamicRange(const AnfNodePtr &anf_node, size_t index,
                                                    const std::string &def_format, const TypeId &type) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(anf_node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto format =
    kernel_info->select_kernel_build_info() == nullptr ? def_format : AnfAlgo::GetInputFormat(anf_node, index);
  auto data_type =
    kernel_info->select_kernel_build_info() == nullptr ? type : AnfAlgo::GetInputDeviceDataType(anf_node, index);
  auto input_range_min = AnfAlgo::GetInputMinShape(anf_node, index);
  auto input_range_max = AnfAlgo::GetInputMaxShape(anf_node, index);
  if (input_range_min.size() != input_range_max.size()) {
    MS_EXCEPTION(ArgumentError) << "Input range size is not equal, min size: " << input_range_min.size()
                                << "max size: " << input_range_max.size();
  }
  trans::ShapeRangeTransfer shapeRangeTransfer;
  if (input_range_min.empty() && input_range_max.empty()) {
    RangePair ret = {{1, 1}};
    return shapeRangeTransfer.GetRealRange(ret, format, data_type);
  }
  RangePair ret;
  for (size_t i = 0; i < input_range_min.size(); ++i) {
    ret.emplace_back(input_range_min[i], input_range_max[i]);
  }
  return shapeRangeTransfer.GetRealRange(ret, format, data_type);
}

RangePair TbeDynamicShapeUtil::GetOutputDynamicRange(const AnfNodePtr &anf_node, size_t index,
                                                     const std::string &def_format, const TypeId &type) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(anf_node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto format =
    kernel_info->select_kernel_build_info() == nullptr ? def_format : AnfAlgo::GetOutputFormat(anf_node, index);
  auto data_type =
    kernel_info->select_kernel_build_info() == nullptr ? type : AnfAlgo::GetOutputDeviceDataType(anf_node, index);
  auto output_range_min = AnfAlgo::GetOutputMinShape(anf_node, index);
  auto output_range_max = AnfAlgo::GetOutputMaxShape(anf_node, index);
  if (output_range_min.size() != output_range_max.size()) {
    MS_EXCEPTION(ArgumentError) << "Onput range size is not equal, min size: " << output_range_min.size()
                                << "max size: " << output_range_max.size();
  }
  trans::ShapeRangeTransfer shapeRangeTransfer;
  if (output_range_max.empty() && output_range_min.empty()) {
    RangePair ret = {{1, 1}};
    return shapeRangeTransfer.GetRealRange(ret, format, data_type);
  }
  RangePair ret;
  for (size_t i = 0; i < output_range_min.size(); ++i) {
    ret.emplace_back(output_range_min[i], output_range_max[i]);
  }
  return shapeRangeTransfer.GetRealRange(ret, format, data_type);
}
}  // namespace tbe
}  // namespace kernel
}  // namespace mindspore
