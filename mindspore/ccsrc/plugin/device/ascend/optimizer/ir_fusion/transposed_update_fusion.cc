/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fusion/transposed_update_fusion.h"
#include <set>
#include <vector>
#include <algorithm>
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/debug/anf_ir_dump.h"
#include "utils/trace_base.h"

namespace mindspore::opt {
namespace {
constexpr size_t kInt32Len = 4;

tensor::TensorPtr CreatePermTensor(const CNodePtr &transposed) {
  auto perm_attr = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(transposed, kAttrPerm);
  std::vector<int32_t> perm;
  (void)std::transform(perm_attr.begin(), perm_attr.end(), std::back_inserter(perm), [&perm_attr](auto v) {
    return v < 0 ? SizeToInt(perm_attr.size()) + LongToInt(v) : LongToInt(v);
  });
  std::vector<int64_t> perm_shape = {SizeToLong(perm.size())};
  TensorTypePtr tensor_type = std::make_shared<TensorType>(kInt32);
  tensor::DeviceInfo device_info{kOpFormat_DEFAULT, tensor_type};
  auto perm_tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt32, perm_shape);
  perm_tensor->set_device_info(device_info);
  auto data_ptr = perm_tensor->data_c();
  MS_EXCEPTION_IF_NULL(data_ptr);
  auto elem_num = perm.size() * kInt32Len;
  auto ret_code = memcpy_s(data_ptr, static_cast<size_t>(perm_tensor->data().nbytes()),
                           reinterpret_cast<void *>(perm.data()), elem_num);
  if (ret_code != EOK) {
    MS_LOG(ERROR) << "Failed to copy data into tensor, memcpy_s errorno: " << ret_code;
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
  op_builder.SetOutputsDeviceType({kNumberTypeInt32});
  AnfAlgo::SetSelectKernelBuildInfo(op_builder.Build(), perm_const.get());
  return perm_const;
}
}  // namespace

const BaseRef TransposedUpdateFusion::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  return VectorRef({prim::kPrimTransposeD, X});
}

const AnfNodePtr TransposedUpdateFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                 const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto transposed = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(transposed);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (common::AnfAlgo::HasNodeAttr(kAttrNopOp, transposed) &&
      common::AnfAlgo::GetNodeAttr<bool>(transposed, kAttrNopOp)) {
    MS_LOG(INFO) << "Node [" << transposed->fullname_with_scope() << "] is a nop op, skip update.";
    return nullptr;
  }

  auto perm_vnode = CreatePermValueNode(transposed);
  std::vector<AnfNodePtr> transpose_inputs = {NewValueNode(std::make_shared<Primitive>(kTransposeOpName)),
                                              transposed->input(1), perm_vnode};
  auto transpose = NewCNode(transpose_inputs, kernel_graph);
  transpose->set_scope(transposed->scope());
  transpose->set_abstract(transposed->abstract());

  if (!CheckAICoreSupportedAny(transpose)) {
    return nullptr;
  }

  MS_EXCEPTION_IF_NULL(kernel_select_);
  kernel_select_->SelectKernel(transpose);
  auto selected_build_info = AnfAlgo::GetSelectKernelBuildInfo(transpose);
  MS_EXCEPTION_IF_NULL(selected_build_info);
  auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(selected_build_info);
  MS_EXCEPTION_IF_NULL(builder);
  auto input_format = AnfAlgo::GetInputFormat(node, 0);
  auto output_format = AnfAlgo::GetOutputFormat(node, 0);
  builder->SetInputsFormat({input_format, kOpFormat_DEFAULT});
  builder->SetOutputsFormat({output_format});
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), transpose.get());
  kernel_graph->AddValueNodeToGraph(perm_vnode);
  if (common::AnfAlgo::HasNodeAttr(kAttrIsKernelDynamicImpl, transposed)) {
    common::AnfAlgo::EraseNodeAttr(kAttrIsKernelDynamicImpl, transposed);
  }
  common::AnfAlgo::CopyNodeAttrs(transposed, transpose);
  kernel_graph->ReplaceRefPair({transposed, 0}, {transpose, 0});
  return transpose;
}
}  // namespace mindspore::opt
