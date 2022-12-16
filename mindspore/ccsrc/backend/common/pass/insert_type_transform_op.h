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

#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_PASS_INSERT_TYPE_TRANSFORM_OP_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_PASS_INSERT_TYPE_TRANSFORM_OP_H_

#include <map>
#include <vector>
#include <string>
#include "kernel/kernel_build_info.h"
#include "backend/common/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
using kernel::KernelBuildInfoPtr;
using kernel::KernelObjectType;

// This attribute represents this node's output is already expanded.
constexpr char kTupleUnfoldExpanded[] = "tuple_unfold_expanded";

static std::map<KernelObjectType, std::string> kObjectTypeToString = {{KernelObjectType::UNKNOWN_TYPE, "unknown"},
                                                                      {KernelObjectType::TENSOR, "tensor"},
                                                                      {KernelObjectType::SCALAR, "scalar"},
                                                                      {KernelObjectType::TUPLE, "tuple"},
                                                                      {KernelObjectType::TUPLE_UNFOLD, "tuple_unfold"}};

// Kernel object type pair of:
// 1. One node's input kernel object type.
// 2. The actual kernel object type this node's kernel info stores.
struct ObjectTypePair {
  KernelObjectType current_input_type;
  KernelObjectType needed_input_type;

  std::string to_string() const {
    if (kObjectTypeToString.find(current_input_type) == kObjectTypeToString.end() ||
        kObjectTypeToString.find(needed_input_type) == kObjectTypeToString.end()) {
      MS_LOG(EXCEPTION) << "The current input object type " << current_input_type << " or needed input object type "
                        << needed_input_type << " is not valid.";
    }

    return kObjectTypeToString[current_input_type] + "->" + kObjectTypeToString[needed_input_type];
  }

  bool operator<(const ObjectTypePair &t) const { return to_string() < t.to_string(); }

  bool operator==(const ObjectTypePair &t) const { return to_string() == t.to_string(); }
};

// For each unmatched type pair, a processing method is required to correct the types by inserting type transforming
// ops or replace origin primitive.
// The method returns new input list so a new node could be created and replace the old node.
// If there's no need to change the input, this method returns the old input.
/**
 * @param {FuncGraphPtr} &func_graph: This func_graph.
 * @param {AnfNodePtr} &input: The input which needs to be processed because of its output kernel object type.
 * @param {CNodePtr} &node: The node which uses input but the type is not satisfied.
 * @param {bool} *new_prim: Whether the origin node's primitive should also be replaced. If true, new primitive node is
 * returned as the first element in returned AnfNodePtrList.
 * @return {AnfNodePtrList}: New input list which replaces 'input' and will be handled by caller.
 */
using ProcessTypeTransformFunc = std::function<AnfNodePtrList(const FuncGraphPtr &func_graph, const AnfNodePtr &input,
                                                              const CNodePtr &node, bool *new_prim)>;

// Kernel object type for newly created virtual node should be set.
void SetObjTypeForTupleGetItemNode(const AnfNodePtr &node);

// SplitTupleInputs methods refer to the pass ConvertTupleInputToDynamicInput. It unfolds tuple inputs and returns the
// unfolded inputs nodes.
int64_t SplitTupleInputs(const FuncGraphPtr &graph, const AnfNodePtr &tuple_input,
                         std::vector<AnfNodePtr> *plant_inputs);

// Create the new cnode which will replace the original cnode.
AnfNodePtr CreateNewNode(const FuncGraphPtr &func_graph, const AnfNodePtrList &input_list, const CNodePtr &origin_node);

// The dynamic input size and real input size should match.
// This method checks input size of new_cnode and origin_node.
void CheckDynamicInputSize(const CNodePtr &new_cnode, const CNodePtr &origin_node);

// Update new cnode's kernel build info according to the original cnode.
// Mainly update the inputs/outputs device types and kernel object types.
void UpdateKernelBuildInfo(const CNodePtr &new_cnode, const CNodePtr &origin_node);

// After kernel selection phase, one kernel's acquired input type may not be the same as the actual input type(the input
// node's output type). We need this pass to transform these types to valid types.
class BACKEND_EXPORT InsertTypeTransformOp : public PatternProcessPass {
 public:
  explicit InsertTypeTransformOp(bool multigraph = true);
  ~InsertTypeTransformOp() override = default;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  // This scenario is migrated from the pass ConvertTupleInputToDynamicInput.
  AnfNodePtrList ProcessTupleUnfoldToTupleUnfold(const FuncGraphPtr &func_graph, const AnfNodePtr &input,
                                                 const CNodePtr &node, bool *new_prim);
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_PASS_INSERT_TYPE_TRANSFORM_OP_H_
