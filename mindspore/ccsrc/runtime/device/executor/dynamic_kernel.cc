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

#include "runtime/device/executor/dynamic_kernel.h"
#include <vector>
#include <algorithm>
#include <stack>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/optimizer/helper.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "pipeline/jit/static_analysis/static_analysis.h"
#include "abstract/dshape.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "abstract/param_validator.h"

namespace mindspore {
namespace device {
void DynamicKernel::Initialize() {
  MS_LOG(INFO) << "Init Start";
  auto cnode = cnode_ptr_.lock();
  MS_EXCEPTION_IF_NULL(cnode);
  is_dynamic_shape_ = common::AnfAlgo::IsDynamicShape(cnode);
  if (!is_dynamic_shape_) {
    MS_LOG(DEBUG) << "cnode is not dynamic shape:" << cnode->fullname_with_scope();
    return;
  }

  is_input_dynamic_shape_ = common::AnfAlgo::GetBooleanAttr(cnode, kAttrInputIsDynamicShape);
  is_output_dynamic_shape_ = common::AnfAlgo::GetBooleanAttr(cnode, kAttrOutputIsDynamicShape);

  auto ret = abstract::GetDependsFormMap(cnode);
  if (ret.empty()) {
    MS_LOG(DEBUG) << "No dynamic_shape_depends found";
    return;
  }
  MS_LOG(INFO) << "Have depends";
  (void)std::transform(ret.begin(), ret.end(), std::inserter(depend_list_, depend_list_.begin()),
                       [](const int64_t &value) { return static_cast<int>(value); });
  MS_LOG(INFO) << "Init End";
}

int DynamicKernel::GetKernelType() const { return static_cast<int>(AnfAlgo::GetKernelType(cnode_ptr_.lock())); }

void DynamicKernel::InferShape() {
  auto cnode = cnode_ptr_.lock();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(INFO) << "InferShape start, node:" << cnode->fullname_with_scope();
  depend_tensor_map_.clear();
  auto inputs = cnode->inputs();
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "Invalid inputs";
  }
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  AbstractBasePtrList args_spec_list;
  auto primitive = GetValueNode<PrimitivePtr>(inputs[0]);
  auto input_size = common::AnfAlgo::GetInputTensorNum(cnode);
  for (size_t i = 0; i < input_size; i++) {
    auto input_node_with_index = common::AnfAlgo::GetPrevNodeOutput(cnode, i);
    auto real_input = input_node_with_index.first;
    MS_EXCEPTION_IF_NULL(real_input);
    auto cnode_input = cnode->input(i + 1);
    MS_EXCEPTION_IF_NULL(cnode_input);
    InferShapeForNopNode(&real_input);
    if (depend_list_.find(i) != depend_list_.end()) {
      auto pre_node_with_index = common::AnfAlgo::GetPrevNodeOutput(cnode, i);
      bool skip_nop_node = !context->get_param<bool>(MS_CTX_ENABLE_MINDRT);
      auto output_addr = AnfAlgo::GetPrevNodeMutableOutputAddr(cnode, i, skip_nop_node);
      std::vector<int64_t> shapes =
        trans::GetRuntimePaddingShape(pre_node_with_index.first, pre_node_with_index.second);
      auto host_type = common::AnfAlgo::GetOutputInferDataType(pre_node_with_index.first, pre_node_with_index.second);
      auto out_tensor = std::make_shared<tensor::Tensor>(host_type, shapes);
      MS_EXCEPTION_IF_NULL(out_tensor);
      // The second parameter must be false, otherwise the device address cannot be released and allocated, and the
      // address size will be wrong in the dynamic shape scenario.
      out_tensor->set_device_address(output_addr, false);
      auto ret = depend_tensor_map_.try_emplace(i, out_tensor);
      if (!ret.second) {
        MS_LOG(EXCEPTION) << "Insert map failed";
      }
      out_tensor->data_sync();
      auto real_abs = real_input->abstract();
      if (real_abs->isa<abstract::AbstractTensor>()) {
        real_input->abstract()->set_value(out_tensor);
      } else if (real_abs->isa<abstract::AbstractTuple>()) {
        auto tuple_get_item_index = common::AnfAlgo::GetTupleGetItemOutIndex(cnode_input->cast<CNodePtr>());
        auto abstract_tuple = real_abs->cast<abstract::AbstractTuplePtr>();
        MS_EXCEPTION_IF_NULL(abstract_tuple);
        auto tuple_elements = abstract_tuple->elements()[tuple_get_item_index];
        tuple_elements->set_value(out_tensor);
      }
    }
    common::AnfAlgo::AddArgList(&args_spec_list, real_input, input_node_with_index.second);
  }
  auto eval_result = opt::CppInferShape(primitive, args_spec_list);
  cnode->set_abstract(eval_result);
}

void DynamicKernel::InferShapeForNopNode(AnfNodePtr *input_node) {
  MS_EXCEPTION_IF_NULL(*input_node);
  if (!common::AnfAlgo::IsNopNode(*input_node) || !common::AnfAlgo::IsDynamicShape(*input_node)) {
    MS_LOG(INFO) << "Input node is not a nop node, no need infer.";
    return;
  }
  MS_LOG(INFO) << "Infer shape for nop node.";
  std::stack<AnfNodePtr> nop_road;
  nop_road.push(*input_node);

  auto temp_node = input_node;
  bool loop = true;
  while (loop) {
    auto input_node_with_idx = common::AnfAlgo::GetPrevNodeOutput(*temp_node, 0);
    auto in_node = input_node_with_idx.first;
    MS_EXCEPTION_IF_NULL(in_node);
    if (common::AnfAlgo::IsNopNode(in_node)) {
      nop_road.push(in_node);
      temp_node = &in_node;
    } else {
      loop = false;
    }
  }

  while (!nop_road.empty()) {
    auto nop_node = nop_road.top();
    MS_EXCEPTION_IF_NULL(nop_node);
    AnfAlgo::InferShape(nop_node->cast<CNodePtr>());
    nop_road.pop();
  }
}
}  // namespace device
}  // namespace mindspore
