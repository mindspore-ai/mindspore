/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fission/bn_split.h"

#include <vector>
#include <memory>
#include <string>
#include <limits>

#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/kernel_info.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kReduceOpSum = "sum";
constexpr auto kDeviceNum = "device_num";
constexpr size_t kPositionOffset = 3;
constexpr int64_t kFusionNumThreshold = 2;
}  // namespace

bool BnSplit::CreateOutputsOfBNTrainingReduce(const FuncGraphPtr &graph, const CNodePtr &bn_cnode,
                                              std::vector<AnfNodePtr> *bn_training_reduce_outputs,
                                              bool is_dynamic) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(bn_cnode);
  if (common::AnfAlgo::GetInputTensorNum(bn_cnode) != kBnInputTensorNum) {
    MS_LOG(INFO) << "BatchNorm's input size less than " << kBnInputTensorNum << ". " << bn_cnode->DebugString();
    return false;
  }
  std::vector<AnfNodePtr> bn_training_reduce_inputs = {
    NewValueNode(std::make_shared<Primitive>(kBNTrainingReduceOpName))};
  bn_training_reduce_inputs.push_back(bn_cnode->input(kIndex1));
  auto bn_training_reduce = NewCNode(bn_training_reduce_inputs, graph);
  MS_EXCEPTION_IF_NULL(bn_training_reduce);
  auto kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(kernel_info);
  bn_training_reduce->set_kernel_info(kernel_info);
  auto types = {common::AnfAlgo::GetOutputInferDataType(bn_cnode, 1),
                common::AnfAlgo::GetOutputInferDataType(bn_cnode, 1)};
  auto shapes = {AnfAlgo::GetOutputDetailShape(bn_cnode, 1), AnfAlgo::GetOutputDetailShape(bn_cnode, 1)};
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, bn_training_reduce.get());
  bn_training_reduce->set_scope(bn_cnode->scope());
  if (is_dynamic) {
    common::AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), bn_training_reduce);
  }
  common::AnfAlgo::CopyNodeAttrs(bn_cnode, bn_training_reduce);

  CreateMultipleOutputsOfAnfNode(graph, bn_training_reduce, kBNTrainingReduceOutputNum, bn_training_reduce_outputs);
  return true;
}

AnfNodePtr BnSplit::CreateOutputsOfBNTrainingUpdate(const FuncGraphPtr &graph, const CNodePtr &bn_cnode,
                                                    const std::vector<AnfNodePtr> &bn_training_reduce_outputs,
                                                    bool is_dynamic) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(bn_cnode);
  CheckCNodeInputSize(bn_cnode, kBnInputTensorNum);
  if (bn_training_reduce_outputs.size() != kBNTrainingReduceOutputNum) {
    MS_LOG(EXCEPTION) << "BN1 outputs has wrong input size" << trace::DumpSourceLines(bn_cnode);
  }
  // the inputs of BNTrainingUpdate are from the outputs of BNTrainingReduce and the inputs of BN
  std::vector<AnfNodePtr> bn_training_update_inputs = {
    NewValueNode(std::make_shared<Primitive>(kBNTrainingUpdateOpName))};
  bn_training_update_inputs.push_back(bn_cnode->input(kIndex1));
  bn_training_update_inputs.push_back(bn_training_reduce_outputs[kIndex0]);
  bn_training_update_inputs.push_back(bn_training_reduce_outputs[kIndex1]);
  bn_training_update_inputs.push_back(bn_cnode->input(kIndex2));
  bn_training_update_inputs.push_back(bn_cnode->input(kIndex3));
  bn_training_update_inputs.push_back(bn_cnode->input(kIndex4));
  bn_training_update_inputs.push_back(bn_cnode->input(kIndex5));
  auto bn_training_update = NewCNode(bn_training_update_inputs, graph);
  MS_EXCEPTION_IF_NULL(bn_training_update);
  auto kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(kernel_info);
  bn_training_update->set_kernel_info(kernel_info);
  bn_training_update->set_abstract(bn_cnode->abstract());
  bn_training_update->set_scope(bn_cnode->scope());
  auto factor = common::AnfAlgo::GetNodeAttr<float>(bn_cnode, kAttrMomentum);
  common::AnfAlgo::SetNodeAttr(kAttrFactor, MakeValue<float>(factor), bn_training_update);
  if (is_dynamic) {
    common::AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), bn_training_update);
    common::AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(true), bn_training_update);
  }
  common::AnfAlgo::CopyNodeAttr(kAttrEpsilon, bn_cnode, bn_training_update);
  if (common::AnfAlgo::HasNodeAttr(kAttrFormat, bn_cnode)) {
    common::AnfAlgo::CopyNodeAttr(kAttrFormat, bn_cnode, bn_training_update);
  } else {
    common::AnfAlgo::SetNodeAttr(kAttrFormat, MakeValue(kOpFormat_NCHW), bn_training_update);
  }
  common::AnfAlgo::SetNodeAttr(kAttrIsRef, MakeValue(true), bn_training_update);
  return bn_training_update;
}

AnfNodePtr BnSplit::SplitBatchNormForTBE(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  bool is_dynamic = common::AnfAlgo::IsDynamicShape(cnode);
  if (common::AnfAlgo::GetInputTensorNum(cnode) < kBnInputTensorNum) {
    MS_LOG(INFO) << "Op[" << cnode->DebugString() << "] has less input than " << kBnInputTensorNum << " inputs.";
    return nullptr;
  }
  // Create BNTrainingReduce node and get outputs of BNTrainingReduce
  std::vector<AnfNodePtr> bn_training_reduce_outputs;
  if (!CreateOutputsOfBNTrainingReduce(func_graph, cnode, &bn_training_reduce_outputs, is_dynamic)) {
    MS_LOG(WARNING) << "Create BNTrainingReduce fail, quit split";
    return nullptr;
  }
  if (bn_training_reduce_outputs.size() != kBN1OutputNum) {
    MS_LOG(EXCEPTION) << "Make outputs of op BNTrainingReduce fail" << trace::DumpSourceLines(node);
  }

  // Create BNTrainingUpdate node
  return CreateOutputsOfBNTrainingUpdate(func_graph, cnode, bn_training_reduce_outputs, is_dynamic);
}

AnfNodePtr SyncBnSplit::SyncBNSplitForTBE(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  bool is_dynamic = common::AnfAlgo::IsDynamicShape(cnode);
  if (common::AnfAlgo::GetInputTensorNum(cnode) < kBnInputTensorNum) {
    MS_LOG(INFO) << "Op[" << cnode->DebugString() << "] has less input than " << kBnInputTensorNum << " inputs.";
    return nullptr;
  }
  // Create BNTrainingReduce node and get outputs of BNTrainingReduce
  std::vector<AnfNodePtr> bn_training_reduce_outputs;
  if (!CreateOutputsOfBNTrainingReduce(func_graph, cnode, &bn_training_reduce_outputs, is_dynamic)) {
    MS_LOG(WARNING) << "Create BNTrainingReduce fail, quit split";
    return nullptr;
  }
  if (bn_training_reduce_outputs.size() != kBN1OutputNum) {
    MS_LOG(EXCEPTION) << "Make outputs of op BNTrainingReduce fail" << trace::DumpSourceLines(node);
  }

  std::vector<AnfNodePtr> allreduce_mul_outputs;
  for (size_t i = 0; i < bn_training_reduce_outputs.size(); ++i) {
    auto allreduce_mul_output =
      CreateAllReduceAndMul(func_graph, bn_training_reduce_outputs[i], cnode, *this, is_dynamic);
    (void)allreduce_mul_outputs.emplace_back(allreduce_mul_output);
  }

  // Create BNTrainingUpdate node
  return CreateOutputsOfBNTrainingUpdate(func_graph, cnode, allreduce_mul_outputs, is_dynamic);
}

AnfNodePtr CreateValueNodeOfDeviceNumReciprocal(const FuncGraphPtr &graph, const CNodePtr &sync_bn_cnode) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(sync_bn_cnode);
  if (!common::AnfAlgo::HasNodeAttr(kDeviceNum, sync_bn_cnode)) {
    MS_LOG(EXCEPTION) << "The node [" << sync_bn_cnode->DebugString() << "] does not have attr device_num."
                      << trace::DumpSourceLines(sync_bn_cnode);
  }
  auto device_num = common::AnfAlgo::GetNodeAttr<int64_t>(sync_bn_cnode, kDeviceNum);
  if (device_num == 0) {
    MS_LOG(EXCEPTION) << "The device_num attr of node [" << sync_bn_cnode->DebugString() << "] should not be 0."
                      << trace::DumpSourceLines(sync_bn_cnode);
  }
  MS_LOG(INFO) << "Got device_num value: " << device_num;
  const float device_num_reciprocal = 1.0 / device_num;

  std::vector<int64_t> device_num_shape = {};
  auto device_num_reciprocal_tensor = std::make_shared<tensor::Tensor>(kNumberTypeFloat32, device_num_shape);
  MS_EXCEPTION_IF_NULL(device_num_reciprocal_tensor);
  auto data_ptr = device_num_reciprocal_tensor->data_c();
  MS_EXCEPTION_IF_NULL(data_ptr);
  auto *val = static_cast<float *>(data_ptr);
  *val = device_num_reciprocal;

  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, device_num_shape);
  auto device_num_reciprocal_value = kernel_graph->NewValueNode(abstract, device_num_reciprocal_tensor);
  MS_EXCEPTION_IF_NULL(device_num_reciprocal_value);
  kernel_graph->AddValueNodeToGraph(device_num_reciprocal_value);
  return device_num_reciprocal_value;
}

AnfNodePtr InsertCast(const FuncGraphPtr &graph, const AnfNodePtr &input, const TypeId dst_type) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input);
  if (common::AnfAlgo::GetOutputInferDataType(input, 0) != dst_type) {
    AnfNodePtr cast = graph->NewCNode({NewValueNode(std::make_shared<Primitive>(kCastOpName)), input});
    common::AnfAlgo::SetOutputTypeAndDetailShape({dst_type}, {AnfAlgo::GetOutputDetailShape(input, 0)}, cast.get());
    common::AnfAlgo::SetNodeAttr(kIsBackendCast, MakeValue(true), cast);
    cast->set_scope(input->scope());
    return cast;
  }
  return input;
}

AnfNodePtr CreateAllReduceAndMul(const FuncGraphPtr &graph, const AnfNodePtr &allreduce_input,
                                 const CNodePtr &sync_bn_cnode, const PatternProcessPass &pass, bool is_dynamic) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(allreduce_input);
  MS_EXCEPTION_IF_NULL(sync_bn_cnode);

  // Cast input to fp32, this can reduce the number of cast node. Since the input of AllReduce,
  // BNTrainingReduce/BNTrainingUpdateGrad op only support fp32 output, when inferred output is fp16, it will
  // insert cast: output_fp32->cast_fp16->allreduce&mul->cast_fp32. Add this cast can eliminate above cast.
  // Should be removed if BNTrainingReduce/BNTrainingUpdateGrad op support fp16 output.
  AnfNodePtr input_node = InsertCast(graph, allreduce_input, kNumberTypeFloat32);

  // create AllReduce
  std::vector<AnfNodePtr> allreduce_inputs = {NewValueNode(std::make_shared<Primitive>(kAllReduceOpName)), input_node};
  auto allreduce = pass.NewCNode(allreduce_inputs, graph);
  MS_EXCEPTION_IF_NULL(allreduce);
  allreduce->set_abstract(input_node->abstract());
  allreduce->set_scope(allreduce_input->scope());
  common::AnfAlgo::SetNodeAttr(kAttrOp, MakeValue(kReduceOpSum), allreduce);
  common::AnfAlgo::CopyNodeAttr(kAttrGroup, sync_bn_cnode, allreduce);
  // use SyncBatchNorm's opid as AllReduce's fusion attr
  auto sync_bn_opname = sync_bn_cnode->fullname_with_scope();
  auto opid_pos = sync_bn_opname.rfind("-op");
  if (opid_pos == std::string::npos || opid_pos + kPositionOffset >= sync_bn_opname.size()) {
    MS_LOG(EXCEPTION) << "Op[" << sync_bn_cnode->DebugString() << "] has no opid."
                      << trace::DumpSourceLines(sync_bn_cnode);
  }
  int64_t opid = std::stol(sync_bn_opname.substr(opid_pos + kPositionOffset));
  // user defined fusion should be greater than 1
  if (opid < kFusionNumThreshold) {
    opid = opid - kFusionNumThreshold + std::numeric_limits<int64_t>::max();
  }
  common::AnfAlgo::SetNodeAttr(kAttrFusion, MakeValue(opid), allreduce);

  // Dynamic Shape support for allreduce node
  if (is_dynamic) {
    common::AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), allreduce);
  }

  // create Mul
  auto device_num_reciprocal_vnode = CreateValueNodeOfDeviceNumReciprocal(graph, sync_bn_cnode);
  std::vector<AnfNodePtr> mul_inputs = {NewValueNode(std::make_shared<Primitive>(kMulOpName)), allreduce,
                                        device_num_reciprocal_vnode};
  auto mul = pass.NewCNode(mul_inputs, graph);
  MS_EXCEPTION_IF_NULL(mul);
  mul->set_abstract(input_node->abstract());
  mul->set_scope(allreduce_input->scope());

  // Dynamic Shape support for mul node
  if (is_dynamic) {
    common::AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), mul);
    common::AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(true), mul);
  }

  // Cast output to origin datatype to reduce the number of cast node.
  // Should be removed if BNTrainingReduce/BNTrainingUpdateGrad op support fp16 output.
  return InsertCast(graph, mul, common::AnfAlgo::GetOutputInferDataType(allreduce_input, 0));
}

const BaseRef BnSplit::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  MS_EXCEPTION_IF_NULL(Xs);
  return VectorRef({prim::kPrimBatchNorm, Xs});
}

const AnfNodePtr BnSplit::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const {
  if (!GetBoolAttr(node, kAttrIsTraining)) {
    MS_LOG(INFO) << "Attr is_training should be true if do fusion";
    return nullptr;
  }
  return SplitBatchNormForTBE(func_graph, node);
}

const BaseRef SyncBnSplit::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimSyncBatchNorm, Xs});
}

const AnfNodePtr SyncBnSplit::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const {
  return SyncBNSplitForTBE(func_graph, node);
}
}  // namespace opt
}  // namespace mindspore
