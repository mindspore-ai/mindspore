/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "cxx_api/model/acl/acl_vm/acl_vm.h"
#include <memory>
#include <string>
#include <vector>
#include "cxx_api/model/acl/acl_model_options.h"
#include "cxx_api/model/acl/acl_vm/acl_multi_graph_session.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace {
inline bool IsMonadNode(const AnfNodePtr &node) {
  if (IsPrimitiveCNode(node, prim::kPrimStateSetItem) || IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
    return true;
  }

  if (HasAbstractMonad(node)) {
    return true;
  }

  return false;
}

std::vector<MSTensor> ParseVectorMsTensorRef(const VectorRef &args) {
  std::vector<MSTensor> ms_tensors;
  for (const auto &arg : args) {
    if (utils::isa<VectorRef>(arg)) {
      auto ret = ParseVectorMsTensorRef(utils::cast<VectorRef>(arg));
      (void)ms_tensors.insert(ms_tensors.end(), ret.begin(), ret.end());
    } else if (utils::isa<MSTensorRef>(arg)) {
      auto wrapper = utils::cast<MSTensorRef>(arg);
      (void)ms_tensors.emplace_back(wrapper.GetTensor());
    } else {
      MS_LOG(EXCEPTION) << "Invalid item " << arg.ToString();
    }
  }
  return ms_tensors;
}
}  // namespace
AclBackend::AclBackend(const std::string &name, const std::string &target,
                       const std::shared_ptr<AclModelOptions> &options)
    : MsBackend(name, target, options->GetDeviceID()) {
  auto session = std::dynamic_pointer_cast<session::MultiGraphAclSession>(MsBackend::target_sess_);
  MS_EXCEPTION_IF_NULL(session);
  session->SetOptions(options);
}

VectorRef AclBackend::MsRunGraph(const GraphId &g, const VectorRef &args, const std::string & /* target */) {
  std::vector<MSTensor> inputs = ParseVectorMsTensorRef(args);
  VectorRef outputs;
  MS_EXCEPTION_IF_NULL(target_sess_);
  auto exec_sess = std::dynamic_pointer_cast<session::MultiGraphAclSession>(target_sess_);
  MS_EXCEPTION_IF_NULL(exec_sess);
  exec_sess->RunGraph(g, inputs, &outputs);
  return outputs;
}

bool AclBackend::GetCond(const BaseRef &c, bool *value) {
  MS_EXCEPTION_IF_NULL(value);
  if (!utils::isa<MSTensorRef>(c)) {
    MS_LOG(ERROR) << "Invalid item " << c.ToString() << " must be a MSTensorRef.";
    return false;
  }
  auto wrapper = utils::cast<MSTensorRef>(c);
  if (wrapper.GetTensor().DataType() != DataType::kNumberTypeBool) {
    MS_LOG(ERROR) << "Invalid data type " << wrapper.GetTensor().DataType() << " must be bool.";
    return false;
  }
  auto data = wrapper.GetTensor().Data();
  if (data == nullptr) {
    return false;
  }
  (*value) = *static_cast<const bool *>(data.get());
  return true;
}

bool AclBackend::GetIndex(const BaseRef &c, int64_t *value) {
  MS_EXCEPTION_IF_NULL(value);
  if (!utils::isa<MSTensorRef>(c)) {
    MS_LOG(ERROR) << "Invalid item " << c.ToString() << " must be a MSTensorRef.";
    return false;
  }

  auto wrapper = utils::cast<MSTensorRef>(c);
  if (wrapper.GetTensor().DataType() == DataType::kNumberTypeInt32) {
    auto data = wrapper.GetTensor().Data();
    if (data == nullptr) {
      return false;
    }
    auto value_int32 = *static_cast<const int32_t *>(data.get());
    (*value) = static_cast<int64_t>(value_int32);
    return true;
  } else if (wrapper.GetTensor().DataType() == DataType::kNumberTypeInt64) {
    auto data = wrapper.GetTensor().Data();
    if (data == nullptr) {
      return false;
    }
    (*value) = *static_cast<const int64_t *>(data.get());
    return true;
  } else {
    MS_LOG(ERROR) << "Index must be Int type.";
    return false;
  }
}

AclCompileGraph::AclCompileGraph(const std::shared_ptr<compile::MsBackend> &backend,
                                 const std::vector<PrimitivePtr> &cut_list)
    : CompileGraph(backend, cut_list) {}

void AclCompileGraph::AddInst(const compile::Instruction &inst, const MSTensorRef &arg) {
  VectorRef args;
  args.push_back(arg);
  compile::CompileGraph::AddInst(inst, args);
}

int64_t AclCompileGraph::Ref(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << "Start Ref node " << node->DebugString(true) << " height_: " << height_;
  if (slots_.count(node) == 0 && node->isa<ValueNode>()) {
    if (IsValueNode<FuncGraph>(node)) {
      MS_LOG(DEBUG) << "Push graph.";
      compile::CompileGraph::AddInst(compile::Instruction::kGraph, GetValueNode(node));
    } else {
      MS_LOG(DEBUG) << "Push.";
      if (IsValueNode<Primitive>(node)) {
        MS_LOG(EXCEPTION) << "must not be primitive in here NodeInfo: " << trace::GetDebugInfo(node->debug_info());
      } else if (IsValueNode<tensor::Tensor>(node)) {
        auto tensor_node = std::dynamic_pointer_cast<tensor::Tensor>(node->cast<ValueNodePtr>()->value());
        MS_EXCEPTION_IF_NULL(tensor_node);
        std::string name = "";
        std::vector<int64_t> shape = tensor_node->shape_c();
        DataType type = static_cast<DataType>(tensor_node->data_type_c());
        auto mstensor_node = MSTensor::CreateRefTensor(name, type, shape, tensor_node->data_c(), tensor_node->Size());
        MSTensorRef mstensor_ref(*mstensor_node);
        AddInst(compile::Instruction::kPush, mstensor_ref);
        MSTensor::DestroyTensorPtr(mstensor_node);
      } else {
        compile::CompileGraph::AddInst(compile::Instruction::kPush, GetValueNode(node));
      }
    }
    Push(node);
  } else if (auto const_parameter = dyn_cast<Parameter>(node);
             slots_.count(node) == 0 && const_parameter != nullptr && const_parameter->has_default()) {
    auto value = const_parameter->default_param();
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<tensor::Tensor>()) {
      auto tensor_node = std::dynamic_pointer_cast<tensor::Tensor>(value);
      MS_EXCEPTION_IF_NULL(tensor_node);
      std::vector<int64_t> shape = tensor_node->shape_c();
      DataType type = static_cast<DataType>(tensor_node->data_type_c());
      auto mstensor_node =
        MSTensor::CreateRefTensor(const_parameter->name(), type, shape, tensor_node->data_c(), tensor_node->Size());
      MSTensorRef mstensor_ref(*mstensor_node);
      AddInst(compile::Instruction::kPush, mstensor_ref);
      MSTensor::DestroyTensorPtr(mstensor_node);
    } else {
      compile::CompileGraph::AddInst(compile::Instruction::kPush, value);
    }
    Push(node);
  }
  MS_LOG(DEBUG) << "End Ref node end height_: " << height_ << ", slots: " << slots_[node]
                << ", return: " << slots_[node] - height_;
  return slots_[node] - height_;
}

void AclCompileGraph::AddExternal(const compile::LinConvertResult &result) {
  VectorRef args;
  args.push_back(result.run);
  args.push_back(result.simu_run);
  size_t size = result.inputs.size();
  for (size_t i = 0; i < size; ++i) {
    const auto &input = result.inputs[i];
    MS_EXCEPTION_IF_NULL(input);
    if (auto parameter = dyn_cast<Parameter>(input); parameter != nullptr && parameter->has_default()) {
      MS_LOG(DEBUG) << parameter->DebugString() << " has default value, will not be pushed as inputs.";
      continue;
    }
    if (IsMonadNode(input)) {
      MS_LOG(DEBUG) << input->DebugString() << " is monad node, will not be pushed as inputs.";
      continue;
    }
    args.emplace_back(Ref(input));
  }
  compile::CompileGraph::AddInst(compile::Instruction::kExternal, args);
  size_t out_count = 0;
  for (auto &out : result.outputs) {
    if (IsMonadNode(out)) {
      continue;
    }
    ++out_count;
    Push(out);
  }
  MS_LOG(DEBUG) << "Args size " << args.size() << " out size " << out_count;
}

void AclCompileGraph::AddInput(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (IsMonadNode(node)) {
    return;
  }
  if (slots_.count(node) == 0) {
    MS_LOG(DEBUG) << "Input node is null " << node->DebugString(true);
    (void)Ref(node);
    return;
  }
  compile::CompileGraph::AddInst(compile::Instruction::kInput, Ref(node));
  set_height(height_ + 1);
}

void AclCompileGraph::AddPartial(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto inputs = node->inputs();
  VectorRef args;
  if (inputs.size() <= 1) {
    MS_LOG(EXCEPTION) << "The node:" << node->DebugString() << "do not have two input.";
  }
  auto fn = inputs[1];
  if (!IsValueNode<FuncGraph>(fn)) {
    MS_LOG(EXCEPTION) << "The type of 1st input of node must be FuncGraph";
  }
  for (size_t i = 1; i < inputs.size(); i++) {
    if (IsMonadNode(inputs[i])) {
      continue;
    }
    args.emplace_back(Ref(inputs[i]));
  }
  compile::CompileGraph::AddInst(compile::Instruction::kPartial, args);
}

int64_t AclCompileGraph::AddCall(const FuncGraphPtr &graph, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto inputs = node->inputs();
  AnfNodePtr fn = inputs[0];
  (void)Ref(fn);
  size_t size = inputs.size();
  size_t non_monad_size = size;
  for (size_t i = size - 1; i > 0; --i) {
    if (IsMonadNode(inputs[i])) {
      --non_monad_size;
      continue;
    }
    AddInput(inputs[i]);
  }
  if (node == graph->output()) {
    AddTailCall(fn, non_monad_size);
    return RET_BREAK;
  }
  MS_LOG(DEBUG) << "Call:" << Ref(fn) << ", " << height_ << ", " << (non_monad_size - 1);
  compile::CompileGraph::AddInst(compile::Instruction::kCall, Ref(fn));
  Ret(static_cast<int64_t>(non_monad_size - 1));
  for (size_t i = size - 1; i > 0; i--) {
    const auto iter = slots_.find(inputs[i]);
    if (iter != slots_.end() && iter->second >= height_) {
      slots_.erase(inputs[i]);
    }
  }
  return RET_SUCCESS;
}

void AclCompileGraph::PushParameters(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> parameters = func_graph->parameters();
  for (size_t i = parameters.size(); i != 0; i--) {
    MS_EXCEPTION_IF_NULL(parameters[i - 1]);
    auto param = parameters[i - 1]->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);
    if (param->has_default()) {
      MS_LOG(DEBUG) << "Parameter " << (i - 1) << ": " << param->DebugString() << " has default value, skip.";
      continue;
    }
    if (IsMonadNode(param)) {
      MS_LOG(DEBUG) << "Parameter " << (i - 1) << ": " << param->DebugString() << " has monad type, skip.";
      continue;
    }
    Push(param);
    MS_LOG(DEBUG) << "Push parameter " << (i - 1) << ": " << param->DebugString();
  }
}

AclCompileGraphs::AclCompileGraphs(const std::shared_ptr<compile::MsBackend> &backend,
                                   const std::vector<PrimitivePtr> &cut_list)
    : CompileGraphs(backend, cut_list) {
  MS_EXCEPTION_IF_NULL(backend);
  MS_LOG(DEBUG) << "Start vm: " << backend->name();
  transform_ = std::make_shared<AclCompileGraph>(backend, cut_list);
  Reset();
}
}  // namespace mindspore
