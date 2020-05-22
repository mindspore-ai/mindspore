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
#include "pre_activate/ascend/ir_fission/topk_split.h"
#include <vector>
#include <memory>
#include <unordered_set>
#include "pre_activate/common/helper.h"
#include "kernel/kernel_build_info.h"
#include "utils/utils.h"
#include "session/kernel_graph.h"
#include "session/anf_runtime_algorithm.h"
#include "device/kernel_info.h"
#include "utils/context/ms_context.h"

namespace mindspore {
namespace opt {
constexpr size_t kFloat16Len = 2;  // size of float16;
constexpr size_t kTopkIndexK = 1;
namespace {
tensor::TensorPtr CreateTensor(const AnfNodePtr &node) {
  // 1 create tensor
  auto shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
  auto last_dim = shape[shape.size() - 1];
  std::vector<int> indices_shape = {SizeToInt(last_dim)};
  TensorTypePtr tensor_type = std::make_shared<TensorType>(kFloat16);
  MS_EXCEPTION_IF_NULL(tensor_type);
  tensor::DeviceInfo device_info{kOpFormat_DEFAULT, tensor_type};
  tensor::TensorPtr indices_tensor = std::make_shared<tensor::Tensor>(kFloat16->type_id(), indices_shape);
  MS_EXCEPTION_IF_NULL(indices_tensor);
  indices_tensor->set_device_info(device_info);

  // 2 set value of tensor
  auto data_ptr = indices_tensor->data_c(true);
  MS_EXCEPTION_IF_NULL(data_ptr);
  std::vector<Eigen::half> half_data;
  for (size_t i = 0; i < last_dim; ++i) {
    half_data.emplace_back(Eigen::half(static_cast<float>(i)));
  }
  auto elem_num = last_dim * kFloat16Len;
  auto ret_code = memcpy_s(data_ptr, static_cast<size_t>(indices_tensor->data().nbytes()), half_data.data(), elem_num);
  if (ret_code != 0) {
    MS_LOG(ERROR) << "Failed to copy data into Tensor.";
    return nullptr;
  }
  return indices_tensor;
}

ValueNodePtr CreateValueNode(const AnfNodePtr &node) {
  tensor::TensorPtr indices_tensor = CreateTensor(node);
  MS_EXCEPTION_IF_NULL(indices_tensor);
  auto indices_const = std::make_shared<ValueNode>(indices_tensor);
  MS_EXCEPTION_IF_NULL(indices_const);
  auto indices_abstract = indices_tensor->ToAbstract();
  indices_const->set_abstract(indices_abstract);
  auto indices_kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(indices_kernel_info);
  indices_const->set_kernel_info(indices_kernel_info);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder1;
  builder1.SetOutputsFormat({kOpFormat_DEFAULT});
  builder1.SetOutputsDeviceType({kNumberTypeFloat16});
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), indices_const.get());
  return indices_const;
}

kernel::KernelBuildInfoPtr CreateKernelBuildInfo() {
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetKernelType(TBE_KERNEL);
  builder.SetFusionType(kernel::OPAQUE);
  builder.SetProcessor(kernel::AICORE);
  builder.SetInputsFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT});
  builder.SetOutputsFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT});
  builder.SetInputsDeviceType({kNumberTypeFloat16, kNumberTypeFloat16});
  builder.SetOutputsDeviceType({kNumberTypeFloat16, kNumberTypeInt32});
  return builder.Build();
}
}  // namespace

const BaseRef TopKSplit::DefinePattern() const {
  VarPtr X1 = std::make_shared<Var>();
  VarPtr X2 = std::make_shared<Var>();
  auto prim = std::make_shared<Primitive>(kTopKOpName);
  return VectorRef({prim, X1, X2});
}

const AnfNodePtr TopKSplit::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  // set value node as topk's input
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // Copy a new node to check supported.
  std::vector<AnfNodePtr> new_inputs{NewValueNode(std::make_shared<Primitive>(kTopKOpName))};
  new_inputs.insert(new_inputs.end(), cnode->inputs().begin() + 1, cnode->inputs().end());
  CNodePtr new_cnode = func_graph->NewCNode(new_inputs);
  MS_EXCEPTION_IF_NULL(new_cnode);
  new_cnode->set_abstract(cnode->abstract());
  new_cnode->set_scope(cnode->scope());
  AnfAlgo::CopyNodeAttrs(cnode, new_cnode);
  CheckCNodeInputSize(new_cnode, kTopkInputNum);
  // Convert the tensor input to scalar and convert it to attr
  auto input_k = new_cnode->input(kTopkIndexK + 1);
  MS_EXCEPTION_IF_NULL(input_k);
  if (!IsValueNode<tensor::Tensor>(input_k)) {
    return nullptr;
  }
  ValuePtr value = GetValueNode(input_k);
  MS_EXCEPTION_IF_NULL(value);
  auto tensor = value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor);
  int32_t *data = reinterpret_cast<int32_t *>(tensor->data_c());
  MS_EXCEPTION_IF_NULL(data);
  auto new_value_node = std::make_shared<ValueNode>(MakeValue(*data));
  new_cnode->set_input(kTopkIndexK + 1, new_value_node);

  std::unordered_set<size_t> attr_index{kTopkIndexK};
  ConstInputToAttr(new_cnode, attr_index);
  auto indices_const = CreateValueNode(new_cnode);
  new_cnode->add_input(indices_const);
  MS_EXCEPTION_IF_NULL(supported_checker_);
  if (!supported_checker_->CheckAiCoreSupported(new_cnode, CreateKernelBuildInfo())) {
    MS_LOG(INFO) << "split topk failed, check to aicpu.";
    return nullptr;
  }

  if (kernel_graph != nullptr) {
    MS_LOG(INFO) << "split topk success. use tbe aicore.";
    kernel_graph->AddValueNodeToGraph(indices_const);
  }

  return new_cnode;
}
}  // namespace opt
}  // namespace mindspore
