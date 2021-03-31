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
#include "backend/optimizer/ascend/ir_fission/bn_split.h"

#include <vector>
#include <memory>
#include <string>
#include <limits>

#include "utils/utils.h"
#include "utils/ms_context.h"
#include "backend/optimizer/common/helper.h"
#include "runtime/device/kernel_info.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kReduceOpSum = "sum";
constexpr auto kDeviceNum = "device_num";

bool CreateOutputsOfBNTrainingReduce(const FuncGraphPtr &graph, const CNodePtr &bn_cnode,
                                     std::vector<AnfNodePtr> *bn_training_reduce_outputs) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(bn_cnode);
  if (AnfAlgo::GetInputTensorNum(bn_cnode) != kBnInputTensorNum) {
    MS_LOG(INFO) << "BatchNorm's input size less than " << kBnInputTensorNum << ". " << bn_cnode->DebugString();
    return false;
  }
  std::vector<AnfNodePtr> bn_training_reduce_inputs = {
    NewValueNode(std::make_shared<Primitive>(kBNTrainingReduceOpName))};
  bn_training_reduce_inputs.push_back(bn_cnode->input(1));
  auto bn_training_reduce = graph->NewCNode(bn_training_reduce_inputs);
  MS_EXCEPTION_IF_NULL(bn_training_reduce);
  auto kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(kernel_info);
  bn_training_reduce->set_kernel_info(kernel_info);
  std::vector<size_t> bn_shape_i0 = AnfAlgo::GetPrevNodeOutputInferShape(bn_cnode, 0);
  if (bn_shape_i0.size() < kShape2dDims) {
    MS_LOG(INFO) << "The BatchNorm's first input's shape dims less than " << kShape2dDims;
    return false;
  }
  std::vector<size_t> bn_training_reduce_shape = {bn_shape_i0[1]};
  auto types = {kNumberTypeFloat32, kNumberTypeFloat32};
  auto shapes = {bn_training_reduce_shape, bn_training_reduce_shape};
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, bn_training_reduce.get());
  bn_training_reduce->set_scope(bn_cnode->scope());
  AnfAlgo::CopyNodeAttrs(bn_cnode, bn_training_reduce);

  CreateMultipleOutputsOfAnfNode(graph, bn_training_reduce, kBNTrainingReduceOutputNum, bn_training_reduce_outputs);
  return true;
}

AnfNodePtr CreateOutputsOfBNTrainingUpdate(const FuncGraphPtr &graph, const CNodePtr &bn_cnode,
                                           const std::vector<AnfNodePtr> &bn_training_reduce_outputs) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(bn_cnode);
  CheckCNodeInputSize(bn_cnode, kBnInputTensorNum);
  if (bn_training_reduce_outputs.size() != kBNTrainingReduceOutputNum) {
    MS_LOG(EXCEPTION) << "BN1 outputs has wrong input size"
                      << " trace: " << trace::DumpSourceLines(bn_cnode);
  }
  // the inputs of BNTrainingUpdate are from the outputs of BNTrainingReduce and the inputs of BN
  std::vector<AnfNodePtr> bn_training_update_inputs = {
    NewValueNode(std::make_shared<Primitive>(kBNTrainingUpdateOpName))};
  bn_training_update_inputs.push_back(bn_cnode->input(1));
  bn_training_update_inputs.push_back(bn_training_reduce_outputs[0]);
  bn_training_update_inputs.push_back(bn_training_reduce_outputs[1]);
  bn_training_update_inputs.push_back(bn_cnode->input(2));
  bn_training_update_inputs.push_back(bn_cnode->input(3));
  bn_training_update_inputs.push_back(bn_cnode->input(4));
  bn_training_update_inputs.push_back(bn_cnode->input(5));
  auto bn_training_update = graph->NewCNode(bn_training_update_inputs);
  MS_EXCEPTION_IF_NULL(bn_training_update);
  auto kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(kernel_info);
  bn_training_update->set_kernel_info(kernel_info);
  bn_training_update->set_abstract(bn_cnode->abstract());
  bn_training_update->set_scope(bn_cnode->scope());
  auto factor = AnfAlgo::GetNodeAttr<float>(bn_cnode, kAttrMomentum);
  AnfAlgo::SetNodeAttr(kAttrFactor, MakeValue<float>(factor), bn_training_update);
  AnfAlgo::CopyNodeAttr(kAttrEpsilon, bn_cnode, bn_training_update);
  AnfAlgo::SetNodeAttr(kAttrIsRef, MakeValue(true), bn_training_update);
  return bn_training_update;
}

AnfNodePtr SplitBatchNormForTBE(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (AnfAlgo::GetInputTensorNum(cnode) < kBnInputTensorNum) {
    MS_LOG(INFO) << "op[" << cnode->DebugString() << "] has less input than " << kBnInputTensorNum << " inputs.";
    return nullptr;
  }
  // Create BNTrainingReduce node and get outputs of BNTrainingReduce
  std::vector<AnfNodePtr> bn_training_reduce_outputs;
  if (!CreateOutputsOfBNTrainingReduce(func_graph, cnode, &bn_training_reduce_outputs)) {
    MS_LOG(WARNING) << "Create BNTrainingReduce fail, quit split";
    return nullptr;
  }
  if (bn_training_reduce_outputs.size() != kBN1OutputNum) {
    MS_LOG(EXCEPTION) << "make outputs of op BNTrainingReduce fail"
                      << " trace: " << trace::DumpSourceLines(node);
  }

  // Create BNTrainingUpdate node
  return CreateOutputsOfBNTrainingUpdate(func_graph, cnode, bn_training_reduce_outputs);
}

AnfNodePtr SyncBNSplitForTBE(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (AnfAlgo::GetInputTensorNum(cnode) < kBnInputTensorNum) {
    MS_LOG(INFO) << "op[" << cnode->DebugString() << "] has less input than " << kBnInputTensorNum << " inputs.";
    return nullptr;
  }
  // Create BNTrainingReduce node and get outputs of BNTrainingReduce
  std::vector<AnfNodePtr> bn_training_reduce_outputs;
  if (!CreateOutputsOfBNTrainingReduce(func_graph, cnode, &bn_training_reduce_outputs)) {
    MS_LOG(WARNING) << "Create BNTrainingReduce fail, quit split";
    return nullptr;
  }
  if (bn_training_reduce_outputs.size() != kBN1OutputNum) {
    MS_LOG(EXCEPTION) << "make outputs of op BNTrainingReduce fail"
                      << " trace: " << trace::DumpSourceLines(node);
  }

  std::vector<AnfNodePtr> allreduce_mul_outputs;
  for (size_t i = 0; i < bn_training_reduce_outputs.size(); ++i) {
    auto allreduce_mul_output = CreateAllReduceAndMul(func_graph, bn_training_reduce_outputs[i], cnode);
    allreduce_mul_outputs.emplace_back(allreduce_mul_output);
  }

  // Create BNTrainingUpdate node
  return CreateOutputsOfBNTrainingUpdate(func_graph, cnode, allreduce_mul_outputs);
}
}  // namespace

AnfNodePtr CreateValueNodeOfDeviceNumReciprocal(const FuncGraphPtr &graph, const CNodePtr &sync_bn_cnode) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(sync_bn_cnode);
  if (!AnfAlgo::HasNodeAttr(kDeviceNum, sync_bn_cnode)) {
    MS_LOG(EXCEPTION) << "op[" << sync_bn_cnode->DebugString() << "] does not have attr device_num.";
  }
  auto device_num = AnfAlgo::GetNodeAttr<int64_t>(sync_bn_cnode, kDeviceNum);
  MS_LOG(INFO) << "device_num value: " << device_num;
  const float device_num_reciprocal = 1.0 / device_num;

  std::vector<int64_t> device_num_shape = {};
  auto device_num_reciprocal_tensor = std::make_shared<tensor::Tensor>(kNumberTypeFloat32, device_num_shape);
  MS_EXCEPTION_IF_NULL(device_num_reciprocal_tensor);
  auto data_ptr = device_num_reciprocal_tensor->data_c();
  MS_EXCEPTION_IF_NULL(data_ptr);
  auto *val = reinterpret_cast<float *>(data_ptr);
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
  if (AnfAlgo::GetOutputInferDataType(input, 0) != dst_type) {
    AnfNodePtr cast = graph->NewCNode({NewValueNode(std::make_shared<Primitive>(kCastOpName)), input});
    AnfAlgo::SetOutputInferTypeAndShape({dst_type}, {AnfAlgo::GetOutputInferShape(input, 0)}, cast.get());
    AnfAlgo::SetNodeAttr(kIsBackendCast, MakeValue(true), cast);
    cast->set_scope(input->scope());
    return cast;
  }
  return input;
}

AnfNodePtr CreateAllReduceAndMul(const FuncGraphPtr &graph, const AnfNodePtr &allreduce_input,
                                 const CNodePtr &sync_bn_cnode) {
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
  auto allreduce = graph->NewCNode(allreduce_inputs);
  MS_EXCEPTION_IF_NULL(allreduce);
  allreduce->set_abstract(input_node->abstract());
  allreduce->set_scope(allreduce_input->scope());
  AnfAlgo::SetNodeAttr(kAttrOp, MakeValue(kReduceOpSum), allreduce);
  AnfAlgo::CopyNodeAttr(kAttrGroup, sync_bn_cnode, allreduce);
  // use SyncBatchNorm's opid as AllReduce's fusion attr
  auto sync_bn_opname = sync_bn_cnode->fullname_with_scope();
  auto opid_pos = sync_bn_opname.rfind("-op");
  if (opid_pos == std::string::npos || opid_pos + 3 >= sync_bn_opname.size()) {
    MS_LOG(EXCEPTION) << "op[" << sync_bn_cnode->DebugString() << "] has no opid.";
    return nullptr;
  }
  int64_t opid = std::stol(sync_bn_opname.substr(opid_pos + 3));
  // user defined fusion should be greater than 1
  if (opid < 2) {
    opid = opid - 2 + std::numeric_limits<int64_t>::max();
  }
  AnfAlgo::SetNodeAttr(kAttrFusion, MakeValue(opid), allreduce);

  // create Mul
  auto device_num_reciprocal_vnode = CreateValueNodeOfDeviceNumReciprocal(graph, sync_bn_cnode);
  std::vector<AnfNodePtr> mul_inputs = {NewValueNode(std::make_shared<Primitive>(kMulOpName)), allreduce,
                                        device_num_reciprocal_vnode};
  auto mul = graph->NewCNode(mul_inputs);
  MS_EXCEPTION_IF_NULL(mul);
  mul->set_abstract(input_node->abstract());
  mul->set_scope(allreduce_input->scope());

  // Cast output to origin datatype to reduce the number of cast node.
  // Should be removed if BNTrainingReduce/BNTrainingUpdateGrad op support fp16 output.
  return InsertCast(graph, mul, AnfAlgo::GetOutputInferDataType(allreduce_input, 0));
}

const BaseRef BnSplit::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  MS_EXCEPTION_IF_NULL(Xs);
  return VectorRef({prim::kPrimBatchNorm, Xs});
}

const AnfNodePtr BnSplit::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const {
  if (!GetBoolAttr(node, kAttrIsTraining)) {
    MS_LOG(INFO) << "is training should be true if do fusion";
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
