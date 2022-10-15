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

#include "plugin/device/ascend/kernel/tbe/tbe_dynamic_shape_util.h"
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
bool ChangeDynamicAbsToActualAbs(const CNodePtr &cnode, const std::shared_ptr<OpInfo> &op_info) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(op_info);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  // Only support for PyNative
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode) {
    return true;
  }

  auto node_input_num = common::AnfAlgo::GetInputTensorNum(cnode);
  auto node_output_num = common::AnfAlgo::GetOutputTensorNum(cnode);
  if (node_input_num != op_info->inputs_ptr().size() || node_output_num != op_info->outputs_ptr().size()) {
    MS_LOG(DEBUG) << "node_input_num[" << node_input_num << "] is different with op_info->inputs_ptr size["
                  << op_info->inputs_ptr().size() << "] or node_output_num[" << node_output_num
                  << "] is different with op_info->outputs_ptr size[" << op_info->outputs_ptr().size()
                  << "], node:" << cnode->DebugString();
    return false;
  }

  MS_LOG(INFO) << "CNode is dynamic shape, but have no dynamic shape op, use static op instead" << cnode->DebugString();
  common::AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(false), cnode);
  common::AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(false), cnode);

  auto input_size = common::AnfAlgo::GetInputTensorNum(cnode);
  auto &inputs = cnode->inputs();
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "Invalid inputs.";
  }
  AbstractBasePtrList args_spec_list;
  auto primitive = GetValueNode<PrimitivePtr>(inputs[0]);
  MS_EXCEPTION_IF_NULL(primitive);
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
  auto eval_result = mindspore::opt::CppInferShapeAndType(primitive, args_spec_list);
  cnode->set_abstract(eval_result);
  return true;
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
  auto op_info = mindspore::kernel::OpLib::FindOp(op_name, OpImplyType::kImplyTBE, is_dynamic_shape);
  // If have no dynamic shape op, get static shape op
  if (op_info != nullptr && !op_info->dynamic_shape() && is_dynamic_shape) {
    if (!ChangeDynamicAbsToActualAbs(cnode, op_info)) {
      // The number of inputs and outputs is incorrect, and the op is not found.
      return nullptr;
    }
  }
  return op_info;
}

std::shared_ptr<OpInfo> TbeDynamicShapeUtil::FindOp(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_name = common::AnfAlgo::GetCNodeName(cnode);
  return FindOp(op_name, cnode);
}

inline std::string GetPrimitiveName(const AnfNodePtr &node) {
  if (!node->isa<CNode>()) {
    return "";
  }

  return AnfUtils::GetCNodeName(node);
}

inline void GetRangeByShape(const AnfNodePtr &anf_node, const ShapeVector &shape, RangePair *range) {
  constexpr int64_t kConv2DMaxShape = 2048;
  auto name = GetPrimitiveName(anf_node);
  for (auto val : shape) {
    if (val < 0) {
      // for "Conv2Dxxx" operators, the upper bound of range can not exceed 4096
      if (name.find("Conv2D") != std::string::npos) {
        range->emplace_back(std::make_pair(1L, kConv2DMaxShape));
      } else {
        range->emplace_back(std::make_pair(1L, -1L));
      }
    } else {
      range->emplace_back(std::make_pair(val, val));
    }
  }
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

  std::string reshape_type = AnfAlgo::GetInputReshapeType(anf_node, index);
  trans::ShapeRangeTransfer shapeRangeTransfer;
  RangePair ret;

  if (input_range_min.empty() && input_range_max.empty()) {
    auto prev_node = common::AnfAlgo::GetPrevNodeOutput(anf_node, index);
    MS_EXCEPTION_IF_NULL(prev_node.first);
    auto shape = common::AnfAlgo::GetOutputInferShape(prev_node.first, prev_node.second);
    GetRangeByShape(anf_node, shape, &ret);
  } else {
    for (size_t i = 0; i < input_range_min.size(); ++i) {
      (void)ret.emplace_back(input_range_min[i], input_range_max[i]);
    }
  }

  return shapeRangeTransfer.GetRealRange(ret, format, data_type, reshape_type);
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
  std::string reshape_type = AnfAlgo::GetOutputReshapeType(anf_node, index);
  trans::ShapeRangeTransfer shapeRangeTransfer;
  RangePair ret;

  if (output_range_min.empty() && output_range_max.empty()) {
    auto shape = common::AnfAlgo::GetOutputInferShape(anf_node, index);
    GetRangeByShape(anf_node, shape, &ret);
  } else {
    for (size_t i = 0; i < output_range_min.size(); i++) {
      (void)ret.emplace_back(output_range_min[i], output_range_max[i]);
    }
  }

  return shapeRangeTransfer.GetRealRange(ret, format, data_type, reshape_type);
}
}  // namespace tbe
}  // namespace kernel
}  // namespace mindspore
