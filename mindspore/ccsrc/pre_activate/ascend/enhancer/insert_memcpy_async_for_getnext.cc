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
#include "pre_activate/ascend/enhancer/insert_memcpy_async_for_getnext.h"
#include <vector>
#include <memory>
#include "pre_activate/ascend/ascend_helper.h"
#include "pre_activate/common/helper.h"
#include "session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
AnfNodePtr InsertMemcpyAsyncForGetNextOutputs(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  if (func_graph == nullptr || node == nullptr) {
    return nullptr;
  }

  size_t output_num = AnfAlgo::GetOutputTensorNum(node);
  if (output_num == 0) {
    MS_LOG(DEBUG) << "Output number is zero, no need to insert memcpy_async!";
    return node;
  }

  // getnext output is tuple and dynamic
  std::vector<AnfNodePtr> make_tuple_inputs;
  make_tuple_inputs.push_back(NewValueNode(prim::kPrimMakeTuple));

  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    auto tuple_get_item = CreatTupleGetItemNode(func_graph, node, output_index);
    auto new_node = CreateMemcpyAsyncOp(func_graph, tuple_get_item);
    if (new_node == nullptr) {
      MS_LOG(EXCEPTION) << "Create memcpy_async op failed!";
    }
    AnfAlgo::SetNodeAttr(kAttrLabelForInsertStreamActive, MakeValue(true), new_node);
    make_tuple_inputs.push_back(new_node);
  }
  AnfNodePtr make_tuple = func_graph->NewCNode(make_tuple_inputs);
  return make_tuple;
}

const BaseRef InsertMemcpyAsyncForGetNext::DefinePattern() const {
  std::shared_ptr<Var> Xs = std::make_shared<SeqVar>();
  auto prim = std::make_shared<Primitive>(kGetNextOpName);

  return VectorRef({prim, Xs});
}

const AnfNodePtr InsertMemcpyAsyncForGetNext::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                      const EquivPtr &) const {
  if (func_graph == nullptr || node == nullptr || !AnfAlgo::IsRealKernel(node)) {
    return nullptr;
  }

  if (AnfAlgo::HasNodeAttr(kAttrVisited, node)) {
    MS_LOG(DEBUG) << "Node op_name[" << kGetNextOpName << "] has visited.";
    return nullptr;
  }
  AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);

  return InsertMemcpyAsyncForGetNextOutputs(func_graph, node);
}
}  // namespace opt
}  // namespace mindspore
