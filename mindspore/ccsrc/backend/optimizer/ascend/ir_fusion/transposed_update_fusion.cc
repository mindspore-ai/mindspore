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
#include "backend/optimizer/ascend/ir_fusion/transposed_update_fusion.h"
#include <set>
#include "backend/optimizer/ascend/ascend_helper.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "debug/anf_ir_dump.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kInt64Len = 8;

tensor::TensorPtr CreatePermTensor(const CNodePtr &transposed) {
  auto perm = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(transposed, kAttrPerm);
  std::vector<int64_t> perm_shape = {SizeToLong(perm.size())};
  TensorTypePtr tensor_type = std::make_shared<TensorType>(kInt64);
  tensor::DeviceInfo device_info{kOpFormat_DEFAULT, tensor_type};
  auto perm_tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt64, perm_shape);
  perm_tensor->set_device_info(device_info);
  MS_EXCEPTION_IF_NULL(perm_tensor);
  auto data_ptr = perm_tensor->data_c();
  MS_EXCEPTION_IF_NULL(data_ptr);
  auto elem_num = perm.size() * kInt64Len;
  auto ret_code = memcpy_s(data_ptr, static_cast<size_t>(perm_tensor->data().nbytes()),
                           reinterpret_cast<void *>(perm.data()), elem_num);
  if (ret_code != 0) {
    MS_LOG(ERROR) << "Failed to copy data into Tensor.";
    return nullptr;
  }
  return perm_tensor;
}

ValueNodePtr CreatePermValueNode(const CNodePtr &transposed) {
  tensor::TensorPtr perm_tensor = CreatePermTensor(transposed);
  MS_EXCEPTION_IF_NULL(perm_tensor);
  auto perm_const = std::make_shared<ValueNode>(perm_tensor);
  MS_EXCEPTION_IF_NULL(perm_const);
  auto perm_abstract = perm_tensor->ToAbstract();
  perm_const->set_abstract(perm_abstract);
  auto perm_kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(perm_kernel_info);
  perm_const->set_kernel_info(perm_kernel_info);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder op_builder;
  op_builder.SetOutputsFormat({kOpFormat_DEFAULT});
  op_builder.SetOutputsDeviceType({kNumberTypeInt64});
  AnfAlgo::SetSelectKernelBuildInfo(op_builder.Build(), perm_const.get());
  return perm_const;
}
}  // namespace

const BaseRef TransposedUpdateFusion::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  return VectorRef({prim::kPrimTranspose, X});
}

const AnfNodePtr TransposedUpdateFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                 const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto transposed = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(transposed);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto perm_vnode = CreatePermValueNode(transposed);
  std::vector<AnfNodePtr> transpose_inputs = {NewValueNode(std::make_shared<Primitive>(kTransposeNODOpName)),
                                              transposed->input(1), perm_vnode};
  auto transpose = kernel_graph->NewCNode(transpose_inputs);
  transpose->set_scope(transposed->scope());
  transpose->set_abstract(transposed->abstract());

  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> kernel_info_list;
  tbe_kernel_query_->GetTbeKernelMetaInfo(transpose, &kernel_info_list);
  if (kernel_info_list.empty()) {
    return nullptr;
  }

  kernel_select_->SelectKernel(transpose);
  auto ori_build_info = AnfAlgo::GetSelectKernelBuildInfo(transpose);
  MS_EXCEPTION_IF_NULL(ori_build_info);
  auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(ori_build_info);
  auto input_format = AnfAlgo::GetInputFormat(node, 0);
  auto output_format = AnfAlgo::GetOutputFormat(node, 0);
  builder->SetInputsFormat({input_format, kOpFormat_DEFAULT});
  builder->SetOutputsFormat({output_format});
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), transpose.get());
  kernel_graph->AddValueNodeToGraph(perm_vnode);
  return transpose;
}
}  // namespace opt
}  // namespace mindspore
