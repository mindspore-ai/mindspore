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
#include "backend/optimizer/ascend/ir_fission/lin_space_fission.h"
#include <vector>
#include <memory>
#include <string>
#include "backend/session/anf_runtime_algorithm.h"
#include "frontend/optimizer/opt.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kLinSpaceInputNum = 3;
constexpr size_t kFloat32Len = 4;
tensor::TensorPtr CreateTensor(const AnfNodePtr &node) {
  // 1 get tensor value of input num
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto input_num = cnode->input(kLinSpaceInputNum);
  MS_EXCEPTION_IF_NULL(input_num);
  if (!IsValueNode<tensor::Tensor>(input_num)) {
    return nullptr;
  }
  ValuePtr value = GetValueNode(input_num);
  MS_EXCEPTION_IF_NULL(value);
  auto tensor = value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor);
  int32_t *data = reinterpret_cast<int32_t *>(tensor->data_c());
  MS_EXCEPTION_IF_NULL(data);

  // 2 create tensor
  int64_t assist_num = *data;
  std::vector<int64_t> assist_shape = {assist_num};
  TensorTypePtr tensor_type = std::make_shared<TensorType>(kFloat32);
  MS_EXCEPTION_IF_NULL(tensor_type);
  tensor::DeviceInfo device_info{kOpFormat_DEFAULT, tensor_type};
  tensor::TensorPtr assist_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), assist_shape);
  MS_EXCEPTION_IF_NULL(assist_tensor);
  assist_tensor->set_device_info(device_info);

  // 3 set value of tensor
  auto data_ptr = assist_tensor->data_c();
  MS_EXCEPTION_IF_NULL(data_ptr);
  std::vector<float> float_data;
  size_t data_num = LongToSize(assist_num);
  for (size_t i = 0; i < data_num; ++i) {
    float_data.emplace_back(static_cast<float>(i));
  }

  auto elem_num = data_num * kFloat32Len;
  auto ret_code = memcpy_s(data_ptr, static_cast<size_t>(assist_tensor->data().nbytes()), float_data.data(), elem_num);
  if (ret_code != 0) {
    MS_LOG(ERROR) << "Failed to copy data into Tensor while creating assist input for LinSpace op.";
    return nullptr;
  }
  return assist_tensor;
}

ValueNodePtr CreateValueNode(const AnfNodePtr &node) {
  tensor::TensorPtr assist_tensor = CreateTensor(node);
  MS_EXCEPTION_IF_NULL(assist_tensor);
  auto assist_const = std::make_shared<ValueNode>(assist_tensor);
  MS_EXCEPTION_IF_NULL(assist_const);
  auto assist_abstract = assist_tensor->ToAbstract();
  assist_const->set_abstract(assist_abstract);
  auto assist_kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(assist_kernel_info);
  assist_const->set_kernel_info(assist_kernel_info);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder op_builder;
  op_builder.SetOutputsFormat({kOpFormat_DEFAULT});
  op_builder.SetOutputsDeviceType({kNumberTypeFloat32});
  AnfAlgo::SetSelectKernelBuildInfo(op_builder.Build(), assist_const.get());
  return assist_const;
}
}  // namespace

const BaseRef LinSpaceFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto lin_space_prim = std::make_shared<Primitive>(kLinSpaceOpName);
  return VectorRef({lin_space_prim, Xs});
}

const AnfNodePtr LinSpaceFission::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->size() != kLinSpaceInputNum + 1) {
    MS_LOG(INFO) << "The node " << cnode->DebugString() << " is not equal to " << kLinSpaceInputNum << " inputs";
    return nullptr;
  }
  std::vector<AnfNodePtr> new_inputs{NewValueNode(std::make_shared<Primitive>(kLinSpaceOpName))};
  auto assist_const = CreateValueNode(cnode);
  new_inputs.push_back(assist_const);
  new_inputs.insert(new_inputs.end(), cnode->inputs().begin() + 1, cnode->inputs().end());
  CNodePtr new_cnode = graph->NewCNode(new_inputs);
  MS_EXCEPTION_IF_NULL(new_cnode);
  new_cnode->set_abstract(cnode->abstract());
  new_cnode->set_scope(cnode->scope());
  AnfAlgo::CopyNodeAttrs(cnode, new_cnode);
  if (kernel_graph != nullptr) {
    kernel_graph->AddValueNodeToGraph(assist_const);
    MS_LOG(INFO) << "Split linspace op success.";
  }
  return new_cnode;
}
}  // namespace opt
}  // namespace mindspore
