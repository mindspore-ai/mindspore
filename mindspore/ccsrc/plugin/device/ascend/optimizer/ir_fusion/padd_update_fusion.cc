/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fusion/padd_update_fusion.h"
#include <vector>
#include <memory>
#include <string>
#include <utility>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "frontend/optimizer/opt.h"
#include "backend/common/optimizer/helper.h"
#include "common/util/platform_info.h"

namespace mindspore {
namespace opt {
namespace {
constexpr uint32_t kInputNum = 1;
const std::vector<std::string> kSupportPlatformPattern = {"Ascend310",  "Ascend610", "BS9SX1A",
                                                          "Ascend310P", "Ascend910", "Ascend910B"};
const std::vector<std::vector<int64_t>> kBlackShape = {{1, 3200, 256}, {1, 3204, 256}, {1, 3208, 256}, {1, 3216, 256},
                                                       {1, 3232, 256}, {1, 3264, 256}, {1, 3328, 256}, {1, 3456, 256}};

bool CheckPlatform() {
  fe::PlatformInfo platform_info;
  fe::OptionalInfo optional_info;
  if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) != 0) {
    MS_LOG(WARNING) << "Get platform info failed, quit fusion.";
    return false;
  }
  MS_LOG(INFO) << "Get soc version: " << optional_info.soc_version;
  if (std::find(kSupportPlatformPattern.begin(), kSupportPlatformPattern.end(),
                platform_info.str_info.short_soc_version) == kSupportPlatformPattern.end()) {
    MS_LOG(WARNING) << "Only support 310, 610, BS9SX1A, 310P, 910 series platform";
    return false;
  }
  return true;
}

bool CheckNode(const CNodePtr &node) {
  auto prenode = common::AnfAlgo::GetPrevNodeOutput(node, 0).first;
  MS_EXCEPTION_IF_NULL(prenode);
  if (prenode->isa<ValueNode>() || IsPrimitiveCNode(prenode, prim::kPrimCast)) {
    MS_LOG(INFO) << "Exit PaddUpdateFusion because prenode is Cast or ValueNode";
    return false;
  }
  if (common::AnfAlgo::IsDynamicShape(node)) {
    MS_LOG(INFO) << "Exit PaddUpdateFusion in dynamic shape scenario.";
    return false;
  }
  if (AnfAlgo::GetPrevNodeOutputFormat(node, 0) == kOpFormat_NC1HWC0) {
    MS_LOG(INFO) << "Exit PaddUpdateFusion because the input format is NC1HWC0";
    return false;
  }
  ShapeVector input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
  ShapeVector output_shape = common::AnfAlgo::GetOutputInferShape(node, 0);
  if (input_shape == output_shape) {
    MS_LOG(INFO) << "Input and output of PadD have the same shape.";
    return false;
  }
  if (std::find(kBlackShape.begin(), kBlackShape.end(), input_shape) != kBlackShape.end()) {
    MS_LOG(INFO) << "Pad shape not supported";
    return false;
  }
  return true;
}
}  // namespace

const BaseRef PaddUpdateFusion::DefinePattern() const {
  VarPtr x = std::make_shared<Var>();
  VectorRef pattern({prim::kPrimPadD, x});
  return pattern;
}

const AnfNodePtr PaddUpdateFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                           const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  if (!CheckPlatform()) {
    return nullptr;
  }
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (common::AnfAlgo::GetInputTensorNum(cnode) != kInputNum) {
    return nullptr;
  }
  if (!CheckNode(cnode)) {
    return nullptr;
  }

  if (!common::AnfAlgo::HasNodeAttr(kAttrPaddings, cnode)) {
    MS_LOG(EXCEPTION) << "PadD should have primal attribute paddings";
  }
  std::vector<std::vector<int64_t>> paddings =
    common::AnfAlgo::GetNodeAttr<std::vector<std::vector<int64_t>>>(cnode, kAttrPaddings);
  if (paddings.size() < 1 || paddings[0].size() < 1) {
    MS_LOG(EXCEPTION) << "Failed to get paddings value from PadD node";
  }
  MS_LOG(INFO) << "Begin to convert PadD to Pad.";
  ShapeVector const_shape = {SizeToLong(paddings.size()), SizeToLong(paddings[0].size())};
  tensor::TensorPtr const_tensor = std::make_shared<tensor::Tensor>(kInt64->type_id(), const_shape);
  MS_EXCEPTION_IF_NULL(const_tensor);
  tensor::DeviceInfo device_info{kOpFormat_DEFAULT, kInt64};
  const_tensor->set_device_info(device_info);
  std::vector<int64_t> const_value;
  for (size_t i = 0; i < paddings.size(); i++) {
    const_value.insert(const_value.end(), paddings[i].begin(), paddings[i].end());
  }
  auto data_ptr = const_tensor->data_c();
  MS_EXCEPTION_IF_NULL(data_ptr);
  auto ret_code = memcpy_s(data_ptr, static_cast<size_t>(const_tensor->data().nbytes()),
                           static_cast<void *>(const_value.data()), const_value.size() * sizeof(int64_t));
  if (ret_code != EOK) {
    MS_LOG(EXCEPTION) << "Failed to copy data into tensor, memcpy_s errorno: " << ret_code;
  }
  auto const_node = std::make_shared<ValueNode>(const_tensor);
  MS_EXCEPTION_IF_NULL(const_node);
  const_node->set_abstract(const_tensor->ToAbstract());
  auto const_node_kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(const_node_kernel_info);
  const_node->set_kernel_info(const_node_kernel_info);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder op_builder;
  op_builder.SetOutputsFormat({kOpFormat_DEFAULT});
  op_builder.SetOutputsDeviceType({kNumberTypeInt64});
  op_builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  AnfAlgo::SetSelectKernelBuildInfo(op_builder.Build(), const_node.get());

  auto prim = std::make_shared<Primitive>(kPadOpName);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), cnode->input(1), const_node};
  auto pad_node = kernel_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(pad_node);
  pad_node->set_abstract(cnode->abstract());
  pad_node->set_scope(cnode->scope());
  auto kernel_info = std::make_shared<device::KernelInfo>();
  pad_node->set_kernel_info(kernel_info);
  KernelSelectPtr kernel_select = std::make_shared<KernelSelect>();
  kernel_select->SelectKernel(pad_node);

  kernel_graph->AddValueNodeToGraph(const_node);
  MS_LOG(INFO) << "Succeed to convert PadD to Pad.";
  return pad_node;
}
}  // namespace opt
}  // namespace mindspore
