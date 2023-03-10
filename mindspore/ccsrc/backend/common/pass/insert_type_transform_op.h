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
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
using kernel::KernelBuildInfoPtr;
using kernel::KernelObjectType;
using KernelBuildInfoBuilder = kernel::KernelBuildInfo::KernelBuildInfoBuilder;

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
    if (kObjectTypeToString.count(current_input_type) == 0 || kObjectTypeToString.count(needed_input_type) == 0) {
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

// SplitTupleInputs methods refer to the pass ConvertTupleInputToDynamicInput. It unfolds tuple inputs and returns the
// unfolded inputs nodes.
int64_t SplitTupleInputsForInsertType(const FuncGraphPtr &graph, const AnfNodePtr &tuple_input,
                                      std::vector<AnfNodePtr> *plant_inputs);

// Create the new cnode which will replace the original cnode.
// This method is called at the last step of this pass specifically.
AnfNodePtr CreateNewNode(const FuncGraphPtr &func_graph, const AnfNodePtrList &input_list, const CNodePtr &origin_node);

// Transforming MakeTuple to RealMakeTuple scenario.
AnfNodePtr CreateRealMakeTupleByMakeTuple(const FuncGraphPtr &func_graph, const CNodePtr &make_tuple_node);

// Node with TupleUnfold output(not MakeTuple) connected to Tuple input scenario.
AnfNodePtr CreateRealMakeTupleByTupleUnfoldInput(const FuncGraphPtr &func_graph,
                                                 const AnfNodePtr &node_with_tuple_unfold_output);

// Set kernel info validation flag according to white list.
void SetBackOffFlag(const KernelBuildInfoPtr &build_info, const CNodePtr &cnode);

// Set kernel info for newly created cnodes. The kernel info will be generated from scratch.
// In some cases, there's no need to set input/output format and type for the node.
void SetKernelInfoForNewCNode(const CNodePtr &cnode, bool set_format_type = true);

// Set kernel info for some value nodes manually.
void SetKernelInfoForValueNode(const ValueNodePtr &value_node);

// Multiplex op infer methods defined under core/ops to generate abstract of new cnode.
abstract::AbstractBasePtr GenerateAbsByOpInfer(const PrimitivePtr &primitive);

// Generate abstract, format and object type for newly created node.
// They can be generated in multiple ways because new node is not processed by kernel selecting method.
std::string GenerateOutputFormatForNewCNode(const CNodePtr &cnode);
void GenerateKernelObjectTypeForNewCNode(const CNodePtr &cnode, std::vector<KernelObjectType> *input_obj_type,
                                         std::vector<KernelObjectType> *output_obj_type);

// After kernel selection phase, one kernel's acquired input type may not be the same as the actual input type(the input
// node's output type). We need this pass to transform these types to valid types.
class BACKEND_EXPORT InsertTypeTransformOp : public PatternProcessPass {
 public:
  explicit InsertTypeTransformOp(bool multigraph = true);
  ~InsertTypeTransformOp() override = default;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  // This method check whether new inputs are generated to replace the old one. If so, new input node list will be
  // returned by method 'Process'.
  bool IsInputUpdated(const AnfNodePtr &origin_input, const AnfNodePtrList &new_input_list) const;

  // This scenario is migrated from the pass ConvertTupleInputToDynamicInput. Please refer to
  // convert_tuple_input_to_dynamic_input.h/cc
  AnfNodePtrList ProcessTupleUnfoldToTupleUnfold(const FuncGraphPtr &func_graph, const AnfNodePtr &input,
                                                 const CNodePtr &node, bool *new_prim);

  // Convert TupleUnfold output to tuple, real tuple with continuous memory.
  AnfNodePtrList ProcessTupleUnfoldToTuple(const FuncGraphPtr &func_graph, const AnfNodePtr &input,
                                           const CNodePtr &node, bool *new_prim);

  // Convert TupleUnfold output to Tensor. Firstly insert TupleToTensor op. Then transform TupleUnfold to Tuple.
  AnfNodePtrList ProcessTupleUnfoldToTensor(const FuncGraphPtr &func_graph, const AnfNodePtr &input,
                                            const CNodePtr &node, bool *new_prim);

  // Convert Tuple output to TupleUnfold. User must be TupleGetItem op and change it to RealTupleGetItem.
  AnfNodePtrList ProcessTupleToTupleUnfold(const FuncGraphPtr &func_graph, const AnfNodePtr &input,
                                           const CNodePtr &node, bool *new_prim);

  // Convert Tuple/Scalar output to Tensor. Simply insert TupleToTensor/ScalarToTensor op.
  AnfNodePtrList ProcessTupleToTensor(const FuncGraphPtr &func_graph, const AnfNodePtr &input, const CNodePtr &node,
                                      bool *new_prim);
  AnfNodePtrList ProcessScalarToTensor(const FuncGraphPtr &func_graph, const AnfNodePtr &input, const CNodePtr &node,
                                       bool *new_prim);

  // Transform Tensor to Tuple/Scalar. Simply insert TensorToTuple/TensorToScalar op.
  AnfNodePtrList ProcessTensorToTuple(const FuncGraphPtr &func_graph, const AnfNodePtr &input, const CNodePtr &node,
                                      bool *new_prim);
  AnfNodePtrList ProcessTensorToScalar(const FuncGraphPtr &func_graph, const AnfNodePtr &input, const CNodePtr &node,
                                       bool *new_prim);
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_PASS_INSERT_TYPE_TRANSFORM_OP_H_
