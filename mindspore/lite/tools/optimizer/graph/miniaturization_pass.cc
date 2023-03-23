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

#include "tools/optimizer/graph/miniaturization_pass.h"

#include <functional>
#include <vector>
#include <memory>
#include "nnacl/op_base.h"
#include "ops/op_utils.h"
#include "ops/fill.h"
#include "src/common/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore::opt {
static inline tensor::TensorPtr GetTensorFromNode(const AnfNodePtr &node) {
  MS_ASSERT(node != nullptr);
  auto value_node = node->cast<ValueNodePtr>();
  if (value_node == nullptr) {
    return nullptr;
  }
  auto value = value_node->value();
  if (value == nullptr || !(value->isa<tensor::Tensor>())) {
    return nullptr;
  }
  auto tensor = value->cast<tensor::TensorPtr>();
  if (tensor == nullptr || tensor->data_ptr() == nullptr || tensor->data_c() == nullptr) {
    return nullptr;
  }
  return tensor;
}

bool MiniaturizationPass::NeedCompress(const tensor::TensorPtr &tensor) {
  auto tensor_data_ptr = tensor->data_ptr();
  auto item_size = tensor_data_ptr->itemsize();
  auto item_num = tensor_data_ptr->size();
  auto data_ptr = tensor_data_ptr->data();
  // No need cast to fill ops while tensor data size is small.
  if (item_num < COMPRESS_TRIGGER_SIZE_) {
    return false;
  }
  int ret = 0;
  for (ssize_t idx = 1; idx < item_num; idx++) {
    auto offset = idx * item_size;
    // No memcmp_s provide in secure lib of huawei
    ret = memcmp(static_cast<uint8_t *>(data_ptr) + offset, static_cast<uint8_t *>(data_ptr) + offset - item_size,
                 item_size);
    if (ret != 0) {
      break;
    }
  }
  return ret == 0;
}

static inline ValuePtr GetFirstVal(const tensor::TensorPtr &tensor) {
  auto tensor_data_ptr = tensor->data_ptr();
  auto data_type = tensor->data_type();
  auto data_ptr = tensor_data_ptr->data();
  if (data_type == kNumberTypeFloat32) {
    float val = static_cast<float *>(data_ptr)[0];
    return MakeValue(val);
  }
  if (data_type == kNumberTypeUInt32) {
    int32_t val = static_cast<int32_t *>(data_ptr)[0];
    return MakeValue(val);
  }
  return nullptr;
}

bool MiniaturizationPass::Run(const FuncGraphPtr &func_graph) {
  auto node_list = TopoSort(func_graph->get_return());
  auto manager = Manage(func_graph, true);
  MS_ASSERT(manager != nullptr);
  bool changed = false;
  // this pass replace the tensor that has large data size and same value with fill ops
  for (auto &node : node_list) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    changed = ProcessOneCNode(func_graph, cnode);
  }
  return changed;
}
bool MiniaturizationPass::ProcessOneCNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  bool changed;
  auto inputs = cnode->inputs();
  for (size_t i = 1; i < inputs.size(); i++) {
    auto tensor = GetTensorFromNode(inputs[i]);
    if (tensor == nullptr) {
      continue;
    }
    // No need cast to fill ops while tensor data size is small.
    if (!NeedCompress(tensor)) {
      continue;
    }

    ValuePtr input1 = GetFirstVal(tensor);
    if (input1 == nullptr) {
      MS_LOG(WARNING) << cnode->fullname_with_scope() << " input " << i << " converter to Fill ops failed.";
      continue;
    }

    ops::Fill ops_fill;
    auto input0 = ops_fill.GetPrim();
    auto node_input0 = NewValueNode(input0);
    auto node_input1 = NewValueNode(input1);
    auto input2 = MakeValue(tensor->shape());
    auto node_input2 = NewValueNode(input2);
    AnfNodePtr fill_node =
      std::make_shared<CNode>(std::vector<AnfNodePtr>{node_input0, node_input1, node_input2}, func_graph);
    func_graph->manager()->Replace(cnode->input(i), fill_node);
    changed = true;
  }
  return changed;
}
}  // namespace mindspore::opt
