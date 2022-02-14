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

#include "plugin/device/ascend/kernel/tbe/tbe_dynaminc_shape_util.h"
#include <memory>
#include <string>
#include <utility>
#include <map>
#include <algorithm>
#include <vector>
#include "utils/ms_context.h"
#include "backend/common/optimizer/helper.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace kernel {
namespace tbe {
namespace {
void ChangeDynamicAbsToActualAbs(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  // Only support for PyNative
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode) {
    return;
  }
  MS_LOG(INFO) << "CNode is dynamic shape, but have no dynamic shape op, use static op instead";
  common::AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(false), cnode);
  common::AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(false), cnode);

  auto input_size = common::AnfAlgo::GetInputTensorNum(cnode);
  auto &inputs = cnode->inputs();
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "Invalid inputs.";
  }
  AbstractBasePtrList args_spec_list;
  auto primitive = GetValueNode<PrimitivePtr>(inputs[0]);
  // Get actual abs
  for (size_t i = 0; i < input_size; ++i) {
    auto input_node_with_index = common::AnfAlgo::GetPrevNodeOutput(cnode, i);
    auto real_input = input_node_with_index.first;
    if (real_input->has_user_data(kActualAbstract)) {
      const auto &actual_abs = real_input->user_data<abstract::AbstractTensor>(kActualAbstract);
      real_input->set_abstract(actual_abs);
    }
    common::AnfAlgo::AddArgList(&args_spec_list, real_input, input_node_with_index.second);
  }
  // Infer real abstract
  auto eval_result = mindspore::opt::CppInferShape(primitive, args_spec_list);
  cnode->set_abstract(eval_result);
}
}  // namespace

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
  auto is_dynamic_shape = common::AnfAlgo::IsDynamicShape(cnode);
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
  auto op_info = mindspore::kernel::OpLib::FindOp(op_name, OpImplyType::kTBE, is_dynamic_shape);
  // If have no dynamic shape op, get static shape op
  if (op_info != nullptr && !op_info->dynamic_shape() && is_dynamic_shape) {
    ChangeDynamicAbsToActualAbs(cnode);
  }
  return op_info;
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
  auto input_range_min = common::AnfAlgo::GetInputMinShape(anf_node, index);
  auto input_range_max = common::AnfAlgo::GetInputMaxShape(anf_node, index);
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
    if (input_range_min[i] < 0) {
      input_range_min[i] = 1;
    }
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
  auto output_range_min = common::AnfAlgo::GetOutputMinShape(anf_node, index);
  auto output_range_max = common::AnfAlgo::GetOutputMaxShape(anf_node, index);
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
    if (output_range_min[i] < 0) {
      output_range_min[i] = 1;
    }
    ret.emplace_back(output_range_min[i], output_range_max[i]);
  }
  return shapeRangeTransfer.GetRealRange(ret, format, data_type);
}
}  // namespace tbe
}  // namespace kernel
}  // namespace mindspore
