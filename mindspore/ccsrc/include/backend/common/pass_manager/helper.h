/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_INCLUDE_BACKEND_OPTIMIZER_COMMON_HELPER_H_
#define MINDSPORE_CCSRC_INCLUDE_BACKEND_OPTIMIZER_COMMON_HELPER_H_

#include <vector>
#include <memory>
#include <utility>
#include <string>
#include <set>
#include "base/base.h"
#include "ir/func_graph.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "utils/ms_utils.h"
#include "include/backend/common/pass_manager/pattern_engine.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_build_info.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace opt {
constexpr size_t kTransOpInputTensorNum = 1;
constexpr size_t kCastInputTensorNum = 1;
constexpr size_t kDependInputTensorNum = 2;
constexpr size_t kReluInputTensorNum = 1;
constexpr size_t kReluGradInputTensorNum = 2;
constexpr size_t kAddInputTensorNum = 2;
constexpr size_t kTupleGetItemInputTensorNum = 2;
constexpr size_t kConvInputTensorNum = 2;
constexpr size_t kRealDivInputTensorNum = 2;
constexpr size_t kSqrtInputTensorNum = 1;
constexpr size_t kMatMulInputTensorNum = 2;
constexpr size_t kMulInputTensorNum = 2;
constexpr size_t kSubInputTensorNum = 2;
constexpr size_t kAssignSubInputTensorNum = 2;
constexpr size_t kDropoutInputTensorNum = 4;
constexpr size_t kAssignInputTensorNum = 2;

constexpr size_t kGradIndex = 3;
constexpr size_t kAddNInputNum = 2;

constexpr size_t kConvBn1OutputNum = 3;
constexpr size_t kBn2ReluOutputNum = 4;

constexpr size_t kBnInputTensorNum = 9;
constexpr size_t kSyncBnInputTensorNum = 5;
constexpr size_t kBnOutputNum = 5;

constexpr size_t kBN1OutputNum = 2;
constexpr size_t kBN2OutputNum = 3;
constexpr size_t kBN3OutputNum = 1;

constexpr size_t kBNGradInputTensorNum = 9;
constexpr size_t kSyncBNGradInputTensorNum = 5;
constexpr size_t kBNGradOutputNum = 3;

constexpr size_t kBNGrad1OutputNum = 3;
constexpr size_t kBNGrad2OutputNum = 5;
constexpr size_t kBNGrad3OutputNum = 1;

constexpr size_t kBNTrainingReduceOutputNum = 2;
constexpr size_t kBNTrainingUpdateOutputNum = 5;
constexpr size_t kBNTrainingUpdateV2OutputNum = 3;
constexpr size_t kBNTrainingUpdateV3OutputNum = 5;
constexpr size_t kBNTrainingUpdateGradOutputNum = 2;

constexpr size_t kSingleOutputNum = 1;
constexpr size_t kSumNodeInputTensorNum = 1;
constexpr size_t kSquareNodeInputTensorNum = 1;
constexpr size_t kSquareSumv2OutputNum = 2;
constexpr size_t kMinimumInputTensorNum = 2;

constexpr size_t kLambNextMVWithDecayInputNum = 7;
constexpr size_t kLambNextMVWithDecayConstantMulInputNum = 5;
constexpr size_t kLambNextMVWithDecayOutputNum = 4;
constexpr size_t kLambNextMVWithDecayV1OutputNum = 4;
constexpr size_t kLambNextRightOutputNum = 2;
constexpr size_t kLambUpdateWithLrV2InputNum = 8;
constexpr size_t kLambNextMVRuleInputNum = 14;
constexpr size_t kLambNextMVRuleOutputNum = 4;
constexpr size_t kBackendReshapeInputTensorNum = 1;
constexpr size_t kBackendTransposeInputTensorNum = 1;
constexpr size_t kAdamApplyOneWithDecayOutputNum = 3;
constexpr size_t kLayerNormBetaGammaBackpropInputTensorNum = 4;
constexpr size_t kLayerNormBetaGammaBackpropOutputNum = 2;
constexpr size_t kLayerNormBetaGammaBackpropV2InputTensorNum = 2;
constexpr size_t kLayerNormXBackpropOutputNum = 4;
constexpr size_t kLayerNormXBackpropV2OutputNum = 2;
constexpr size_t kLayerNormGradInputTensorNum = 5;
constexpr size_t kAdamApplyOneOutputNum = 3;
constexpr size_t kApplyMomentumInputTensorNum = 5;
constexpr size_t kBiasAddInputTensorNum = 2;
constexpr size_t kTopkInputTensorNum = 2;
constexpr size_t kLarsV2InputTensorNum = 4;
constexpr size_t kFusedMulApplyMomentumOutputNum = 2;
constexpr size_t kSplitInputTensorNum = 1;
constexpr size_t kGatherV2DynInputTensorNum = 3;
constexpr size_t kUnsortedSegmentSumDInputTensorNum = 2;
constexpr size_t kSoftmaxCrossEntropyWithLogitsOutputNum = 2;
constexpr size_t kSparseSoftmaxCrossEntropyWithLogitsInputTensorNum = 2;
constexpr size_t kOneHotOutputNum = 1;
constexpr size_t kOneHotInputTensorNum = 4;

BACKEND_COMMON_EXPORT bool UnVisited(const BaseRef &n);

BACKEND_COMMON_EXPORT bool Visited(const BaseRef &n);

// Create new cnode with dump flag and trace info maintained
BACKEND_COMMON_EXPORT CNodePtr NewCNode(const std::vector<AnfNodePtr> &inputs, const FuncGraphPtr &fg,
                                        const std::vector<AnfNodePtr> &orig_nodes);

BACKEND_COMMON_EXPORT CNodePtr NewCNode(const CNodePtr &cnode, const KernelGraphPtr &fg,
                                        const std::vector<AnfNodePtr> &orig_nodes);

BACKEND_COMMON_EXPORT void CheckCNodeInputSize(const CNodePtr &cnode, size_t input_tensor_size);

BACKEND_COMMON_EXPORT void CreateMultipleOutputsOfAnfNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                          size_t output_num, std::vector<AnfNodePtr> *outputs);

tensor::TensorPtr CreateTensorWithValueTuple(const ValueTuplePtr &value_tuple_ptr, const TypePtr &type_ptr,
                                             size_t data_length);

BACKEND_COMMON_EXPORT tensor::TensorPtr CreateTupleTensor(const ValueTuplePtr &value_tuple);

BACKEND_COMMON_EXPORT AnfNodePtr CreateTensorInput(const KernelGraphPtr &kernel_graph, const AnfNodePtr &input_node);

BACKEND_COMMON_EXPORT AnfNodePtr CreateTensorMoveOp(const FuncGraphPtr &graph, const AnfNodePtr &node);

BACKEND_COMMON_EXPORT std::vector<AnfNodePtr> InsertTensorMoveForGraphOutput(const FuncGraphPtr &graph,
                                                                             const AnfNodePtr &node);

BACKEND_COMMON_EXPORT CNodePtr CreatTupleGetItemNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                     size_t output_idx);

BACKEND_COMMON_EXPORT CNodePtr CreateMakeTupleNode(const FuncGraphPtr &func_graph,
                                                   const std::vector<AnfNodePtr> &tuple_inputs);

BACKEND_COMMON_EXPORT ValueNodePtr CreateShapeValueNode(const FuncGraphPtr &func_graph,
                                                        const std::vector<int64_t> &shape, bool to_tensor = false);

BACKEND_COMMON_EXPORT CNodePtr CreateReshapeNode(const FuncGraphPtr &graph, const AnfNodePtr &input_node,
                                                 const ShapeVector &shape);

BACKEND_COMMON_EXPORT CNodePtr AddCastNode(const FuncGraphPtr &func_graph, const TypeId dst_type, const CNodePtr &node,
                                           const bool is_input, const size_t input_index = 0);

BACKEND_COMMON_EXPORT AnfNodePtr CreateNodeBase(const FuncGraphPtr &graph,
                                                const std::vector<AnfNodePtr> &new_node_inputs, const AnfNodePtr &node);

BACKEND_COMMON_EXPORT bool IsUsedByOthers(const FuncGraphPtr &graph, const AnfNodePtr &node);

BACKEND_COMMON_EXPORT std::shared_ptr<std::vector<std::pair<AnfNodePtr, int>>> GetRealNodeUsedList(
  const FuncGraphPtr &graph, const AnfNodePtr &node);

BACKEND_COMMON_EXPORT std::shared_ptr<std::vector<std::pair<AnfNodePtr, int>>> GetRealNodeUsedListByOutputIdx(
  const FuncGraphPtr &graph, const AnfNodePtr &node, size_t output_index);

AnfNodePtr SexpToNode(const BaseRef &sexp, const BaseRef &graph, PrimitiveVarMap *primitive_vars,
                      bool multigraph = false);

// Get anf_node from equiv by var_node
BACKEND_COMMON_EXPORT AnfNodePtr GetAnfNodeByVar(const EquivPtr &equiv, const VarPtr &var_node);

// Get tuple getitem's index
BACKEND_COMMON_EXPORT int64_t GetGetitemIndex(const AnfNodePtr &getitem);

// Get attr which is bool from cnode
BACKEND_COMMON_EXPORT bool GetBoolAttr(const AnfNodePtr &node, const std::string &attr_name);

// Check node's data type is in supported data type set
BACKEND_COMMON_EXPORT bool CheckSupportDataType(const AnfNodePtr &node,
                                                const std::set<TypeId> &supported_data_type_set);

// Create a new value node of func graph, not kernel graph
BACKEND_COMMON_EXPORT ValueNodePtr MakeValueNode(const ValueNodePtr &value_node);

// Transfer depend or updatestate to the new node
BACKEND_COMMON_EXPORT void TransferDependOrUpdateState(const CNodePtr &old_node, const FuncGraphPtr &graph,
                                                       const CNodePtr &new_node);

// Infer the shape and type.
BACKEND_COMMON_EXPORT AbstractBasePtr CppInferShapeAndType(const PrimitivePtr &prim,
                                                           const AbstractBasePtrList &args_spec_list);

// Generate kernel build info for created kernel
BACKEND_COMMON_EXPORT kernel::KernelBuildInfoPtr GenerateKernelBuildInfo(const std::vector<AnfNodePtr> &node_list);

BACKEND_COMMON_EXPORT kernel::KernelBuildInfoPtr GenerateKernelBuildInfo(const CNodePtr &node);

BACKEND_COMMON_EXPORT bool IsConstant(const BaseRef &n);

// Get custom operator attr input indexes
BACKEND_COMMON_EXPORT void GetCustomOpAttrIndex(const PrimitivePtr &primitive, mindspore::HashSet<size_t> *indexes);

BACKEND_COMMON_EXPORT size_t GetInputNodeIndex(const AnfNodePtr &input, const CNodePtr &user_node);

BACKEND_COMMON_EXPORT int64_t SplitTupleInputs(const FuncGraphPtr &graph, const AnfNodePtr &tuple_input,
                                               std::vector<AnfNodePtr> *plant_inputs);

BACKEND_COMMON_EXPORT AnfNodePtr ConvertMakeTupleInputToPlantInputs(const FuncGraphPtr &graph,
                                                                    const CNodePtr &cnode_ptr);

using LaunchHandler = abstract::AbstractBasePtr (*)(const PrimitivePtr &,
                                                    const std::vector<abstract::AbstractBase *> &);
BACKEND_COMMON_EXPORT void set_launch_handler(const LaunchHandler &handler);

BACKEND_COMMON_EXPORT abstract::AbstractBasePtr LaunchPy(const PrimitivePtr &primitive,
                                                         const std::vector<abstract::AbstractBase *> &args_abs_list);
BACKEND_COMMON_EXPORT AbstractBasePtr InferAbstract(const PrimitivePtr &primitive,
                                                    const std::vector<AnfNodePtr> &input_list);

BACKEND_COMMON_EXPORT AnfNodePtr CreateValueNodeWithKernelInfo(const FuncGraphPtr &graph, const ValuePtr &value);

BACKEND_COMMON_EXPORT bool CheckStreamAndCoreAttrWithOrigNodes(const FuncGraphPtr &func_graph,
                                                               const std::vector<AnfNodePtr> &orig_nodes);

BACKEND_COMMON_EXPORT void UpdateStreamAndCoreAttrs(const CNodePtr &node, const std::vector<AnfNodePtr> &orig_nodes);
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_BACKEND_OPTIMIZER_COMMON_HELPER_H_
