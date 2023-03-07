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

#include "plugin/device/ascend/optimizer/enhancer/transpose_optimizer.h"
#include <vector>
#include <algorithm>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/debug/anf_ir_dump.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
namespace {
const int kPermIndex = 2;
template <typename T>
void UpdatePermValue(const tensor::TensorPtr &tensor, std::vector<T> *new_value, bool *need_change_flag) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(need_change_flag);
  MS_EXCEPTION_IF_NULL(new_value);
  auto *data = static_cast<T *>(tensor->data_c());
  for (size_t i = 0; i < tensor->DataSize(); i++) {
    auto v = *(data + i);
    if (v < 0) {
      *need_change_flag = true;
      (void)new_value->emplace_back(static_cast<T>(tensor->DataSize()) + v);
    } else {
      (void)new_value->emplace_back(v);
    }
  }
}
}  // namespace

const BaseRef TransposeOptimizer::DefinePattern() const {
  VarPtr x1 = std::make_shared<Var>();
  VarPtr x2 = std::make_shared<Var>();
  return VectorRef({prim::kPrimTranspose, x1, x2});
}

const AnfNodePtr TransposeOptimizer::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                             const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (common::AnfAlgo::GetInputTensorNum(node) <= 1) {
    return nullptr;
  }
  auto perm_input = cnode->input(kPermIndex);
  MS_EXCEPTION_IF_NULL(perm_input);
  if (!perm_input->isa<ValueNode>()) {
    return nullptr;
  }
  auto value_node = perm_input->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    auto tensor_type = tensor->data_type_c();
    tensor::TensorPtr perm_tensor = nullptr;
    bool need_change_flag = false;
    switch (tensor_type) {
      case TypeId::kNumberTypeInt64: {
        std::vector<int64_t> value64;
        UpdatePermValue(tensor, &value64, &need_change_flag);
        if (!need_change_flag) {
          return nullptr;
        }
        perm_tensor = std::make_shared<tensor::Tensor>(value64, kInt64);
        break;
      }
      case TypeId::kNumberTypeInt32: {
        std::vector<int32_t> value32;
        UpdatePermValue(tensor, &value32, &need_change_flag);
        if (!need_change_flag) {
          return nullptr;
        }
        perm_tensor = std::make_shared<tensor::Tensor>(value32, kInt32);
        break;
      }
      default:
        MS_LOG(EXCEPTION) << "Unexpected tensor data type :" << tensor_type;
    }
    auto ret = memcpy_s(tensor->data_c(), tensor->Size(), perm_tensor->data_c(), perm_tensor->Size());
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "Copy data error, src data: " << perm_tensor->ToString()
                        << ", copy size: " << perm_tensor->Size() << "dst capacity: " << tensor->Size();
    }
  }
  return node;
}
}  // namespace opt
}  // namespace mindspore
