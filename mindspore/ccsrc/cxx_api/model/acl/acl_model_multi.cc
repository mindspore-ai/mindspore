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

#include "cxx_api/model/acl/acl_model_multi.h"
#include <vector>
#include <utility>
#include <map>
#include <string>
#include <algorithm>
#include "backend/session/session_basic.h"
#include "backend/session/session_factory.h"
#include "cxx_api/factory.h"
#include "vm/backend.h"
#include "vm/transform.h"
#include "acl/acl_rt.h"
#include "mindspore/core/load_mindir/infer_mindir.h"
#include "debug/trace.h"

namespace mindspore {
API_FACTORY_REG(ModelImpl, Ascend310, AclModelMulti);

namespace {
class MSTensorRef : public BaseRef {
 public:
  static VectorRef Convert(const std::vector<MSTensor> &tensors) {
    VectorRef res;
    std::transform(tensors.begin(), tensors.end(), std::back_inserter(res),
                   [](const MSTensor &t) { return MSTensorRef(t); });
    return res;
  }

  static std::vector<MSTensor> Convert(const BaseRef &args) {
    std::vector<MSTensor> res;
    if (utils::isa<VectorRef>(args)) {
      VectorRef args_vec = utils::cast<VectorRef>(args);
      for (size_t i = 0; i < args_vec.size(); ++i) {
        const auto &item = args_vec[i];
        if (!utils::isa<MSTensorRef>(item)) {
          MS_LOG(EXCEPTION) << "Invalid item " << item.ToString() << " at index " << i;
        }
        auto wrapper = utils::cast<MSTensorRef>(item);
        res.push_back(wrapper.ms_tensor_);
      }
    } else if (utils::isa<MSTensorRef>(args)) {
      auto wrapper = utils::cast<MSTensorRef>(args);
      res.push_back(wrapper.ms_tensor_);
    } else {
      MS_LOG(EXCEPTION) << "Invalid BaseRef " << args.ToString() << " must be MSTensorRef or VectorRef{MSTensorRef...}";
    }

    return res;
  }

  MS_DECLARE_PARENT(MSTensorRef, BaseRef);
  explicit MSTensorRef(const MSTensor &tensor) : ms_tensor_(tensor) {}
  ~MSTensorRef() override = default;

  const MSTensor &GetTensor() const { return ms_tensor_; }
  std::shared_ptr<Base> copy() const override {
    MSTensor *tensor = ms_tensor_.Clone();
    auto res = std::make_shared<MSTensorRef>(static_cast<const MSTensor &>(*tensor));
    MSTensor::DestroyTensorPtr(tensor);
    return res;
  }

  uint32_t type() const override { return tid(); }
  std::string ToString() const override { return ms_tensor_.Name(); }
  bool operator==(const BaseRef &other) const override {
    if (!utils::isa<MSTensorRef>(other)) {
      return false;
    }
    return *this == utils::cast<MSTensorRef>(other);
  }

  bool operator==(MSTensorRef &other) {
    return (ms_tensor_.Name() == other.ms_tensor_.Name()) && (ms_tensor_.Shape() == other.ms_tensor_.Shape()) &&
           (ms_tensor_.MutableData() == other.ms_tensor_.MutableData()) &&
           (ms_tensor_.DataSize() == other.ms_tensor_.DataSize()) &&
           (ms_tensor_.DataType() == other.ms_tensor_.DataType());
  }

 private:
  MSTensor ms_tensor_;
};

class MultiGraphAclSession : public session::SessionBasic {
 public:
  MultiGraphAclSession() = default;
  ~MultiGraphAclSession() override = default;
  void Init(uint32_t device_id) override;
  GraphId CompileGraphImpl(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) override;
  void RunGraph(GraphId graph_id, const std::vector<MSTensor> &inputs, VectorRef *outputs);
  void SetOptions(const std::shared_ptr<AclModelOptions> &options) { options_ = options; }

 private:
  std::map<GraphId, GraphCell> graphs_ = {};
  std::shared_ptr<AclModelOptions> options_ = nullptr;
};

void MultiGraphAclSession::Init(uint32_t device_id) { InitExecutor(kDavinciMultiGraphInferenceDevice, device_id); }

GraphId MultiGraphAclSession::CompileGraphImpl(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) {
  class FirstGraphModeGuard {
   public:
    explicit FirstGraphModeGuard(const std::shared_ptr<AclModelOptions> &options) : options_(options) {
      if (options_ != nullptr) {
        options_->SetFirstGraph(true);
      }
    }
    ~FirstGraphModeGuard() {
      if (options_ != nullptr) {
        options_->SetFirstGraph(false);
      }
    }

   private:
    std::shared_ptr<AclModelOptions> options_;
  };
  MS_LOG(INFO) << "Start MultiGraph Compile.";
  auto kernel_graph = ConstructKernelGraph(lst, outputs, false);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  ModelConverter model_converter_;
  model_converter_.set_options(options_);
  FirstGraphModeGuard guard(options_);
  auto om_data = model_converter_.LoadMindIR(kernel_graph);
  if (om_data.Data() == nullptr || om_data.DataSize() == 0) {
    MS_LOG(ERROR) << "Load MindIR failed.";
    return kMCFailed;
  }
  std::shared_ptr<Graph> graph = std::make_shared<Graph>(std::make_shared<Graph::GraphData>(om_data, ModelType::kOM));
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_cell = GraphCell(graph);
  auto ret = graph_cell.Load(options_->GetDeviceID());
  if (ret != kSuccess) {
    MS_LOG(EXCEPTION) << "Load failed.";
  }
  graphs_[kernel_graph->graph_id()] = graph_cell;
  MS_LOG(INFO) << "Mulit graph compile success, graph id " << kernel_graph->graph_id();
  return kernel_graph->graph_id();
}

void MultiGraphAclSession::RunGraph(GraphId graph_id, const std::vector<MSTensor> &inputs, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  MS_LOG(INFO) << "Start run graph " << graph_id;
  auto iter = graphs_.find(graph_id);
  if (iter == graphs_.end()) {
    MS_LOG(EXCEPTION) << "Graph id " << graph_id << " not found.";
  }
  std::vector<MSTensor> out_tensors;
  auto ret = iter->second.Run(inputs, &out_tensors);
  if (ret != kSuccess) {
    MS_LOG(EXCEPTION) << "Graph id " << graph_id << " run failed.";
  }
  (*outputs) = MSTensorRef::Convert(out_tensors);
}

class AclBackend : public compile::MsBackend {
 public:
  AclBackend(const std::string &name, const std::string &target, const std::shared_ptr<AclModelOptions> &options)
      : MsBackend(name, target, options->GetDeviceID()) {
    auto session = std::dynamic_pointer_cast<MultiGraphAclSession>(MsBackend::target_sess_);
    MS_EXCEPTION_IF_NULL(session);
    session->SetOptions(options);
  }

  ~AclBackend() override = default;

  VectorRef MsRunGraph(const GraphId &g, const VectorRef &args, const std::string &target) override {
    std::vector<MSTensor> inputs;
    for (const auto &arg : args) {
      if (!utils::isa<MSTensorRef>(arg)) {
        MS_LOG(EXCEPTION) << "Invalid item " << arg.ToString();
      }
      auto wrapper = utils::cast<MSTensorRef>(arg);
      inputs.emplace_back(wrapper.GetTensor());
    }

    VectorRef outputs;
    MS_EXCEPTION_IF_NULL(target_sess_);
    auto exec_sess = std::dynamic_pointer_cast<MultiGraphAclSession>(target_sess_);
    MS_EXCEPTION_IF_NULL(exec_sess);
    exec_sess->RunGraph(g, inputs, &outputs);
    return outputs;
  }

  bool GetCond(const BaseRef &c, bool *value) override {
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
    (*value) = *reinterpret_cast<const bool *>(data.get());
    return true;
  }

  bool GetIndex(const BaseRef &c, int64_t *value) override {
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
      auto value_int32 = *reinterpret_cast<const int32_t *>(data.get());
      (*value) = static_cast<int64_t>(value_int32);
      return true;
    } else if (wrapper.GetTensor().DataType() == DataType::kNumberTypeInt64) {
      auto data = wrapper.GetTensor().Data();
      if (data == nullptr) {
        return false;
      }
      (*value) = *reinterpret_cast<const int64_t *>(data.get());
      return true;
    } else {
      MS_LOG(ERROR) << "Index must be Int type.";
      return false;
    }
  }
};

class AclCompileGraph : public compile::CompileGraph {
 public:
  explicit AclCompileGraph(const std::shared_ptr<compile::MsBackend> &backend,
                           const std::vector<PrimitivePtr> &cut_list)
      : CompileGraph(backend, cut_list) {}
  ~AclCompileGraph() override = default;

  void AddInst(const compile::Instruction &inst, const MSTensorRef &arg) {
    VectorRef args;
    args.push_back(arg);
    compile::CompileGraph::AddInst(inst, args);
  }

  int64_t Ref(const AnfNodePtr &node) override {
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
    }
    MS_LOG(DEBUG) << "End Ref node end height_: " << height_ << ", slots: " << slots_[node]
                  << ", return: " << slots_[node] - height_;
    return slots_[node] - height_;
  }
};

class AclCompileGraphs : public compile::CompileGraphs {
 public:
  explicit AclCompileGraphs(const std::shared_ptr<compile::MsBackend> &backend,
                            const std::vector<PrimitivePtr> &cut_list)
      : CompileGraphs(backend, cut_list) {
    MS_EXCEPTION_IF_NULL(backend);
    MS_LOG(DEBUG) << "Start vm: " << backend->name();
    transform_ = std::make_shared<AclCompileGraph>(backend, cut_list);
    Reset();
  }
  ~AclCompileGraphs() override = default;
  void Compile(const FuncGraphPtr &graph) override {
    MS_LOG(DEBUG) << "Start";
    mapping_[graph] = SizeToLong(insts_.size());
    if (transform_ != nullptr) {
      auto insts = transform_->Run(graph, false);
      if (!insts.empty()) {
        (void)insts_.insert(insts_.end(), insts.begin(), insts.end());
      }
    }
    MS_LOG(DEBUG) << "End";
  }
};

std::shared_ptr<compile::MsBackend> CreateBackend(const std::shared_ptr<AclModelOptions> &options) {
  MS_EXCEPTION_IF_NULL(options);
  return std::make_shared<AclBackend>(kMsConvert, kDavinciMultiGraphInferenceDevice, options);
}

bool HasMultiGraph(const FuncGraphPtr &fg) {
  MS_EXCEPTION_IF_NULL(fg);
  std::vector<AnfNodePtr> all_nodes = TopoSort(fg->get_return());
  for (const auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (IsValueNode<FuncGraph>(node)) {
      MS_LOG(INFO) << fg->ToString() << " has FuncGraph node " << node->DebugString() << " is multi graph.";
      return true;
    }
  }
  return false;
}
}  // namespace
Status AclModelMulti::Build() {
  if (!is_multi_graph_.has_value()) {
    is_multi_graph_ = ModelImpl::GetFuncGraph() == nullptr ? false : HasMultiGraph(ModelImpl::GetFuncGraph());
  }

  if (!is_multi_graph_.value()) {
    return AclModel::Build();
  }

  if (vm_ != nullptr) {
    MS_LOG(INFO) << "Multi graph model has been built, skip.";
    return kSuccess;
  }
  MS_LOG(INFO) << "Start build multi graph model.";
  // perpare func graph
  auto manager = MakeManager();
  manager->AddFuncGraph(ModelImpl::GetFuncGraph());
  ModelImpl::GetFuncGraph()->set_manager(manager);
  // set inputs
  SetInputs();
  // infer mindir
  abstract::AbstractBasePtrList broaded_args;
  auto fg = ModelImpl::GetFuncGraph();
  MS_EXCEPTION_IF_NULL(fg);
  const auto &inputs = fg->get_inputs();
  (void)std::transform(inputs.begin(), inputs.end(), std::back_inserter(broaded_args),
                       [](const AnfNodePtr &n) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(n);
                         auto abstract = n->abstract();
                         MS_EXCEPTION_IF_NULL(abstract);
                         if (abstract->GetValueTrack() != kAnyValue) {
                           return abstract->Broaden();
                         }
                         return abstract;
                       });
  (void)InferMindir(ModelImpl::GetFuncGraph(), broaded_args);
  // create vm
  auto backend = CreateBackend(std::make_shared<AclModelOptions>(model_context_));
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  backend->set_is_multi_graph_sink(false);
  context_ptr->set_param<std::string>(MS_CTX_DEVICE_TARGET, kDavinciMultiGraphInferenceDevice);
  context_ptr->set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, false);
  context_ptr->set_param<bool>(MS_CTX_ENABLE_LOOP_SINK, false);
  auto compile = std::make_shared<AclCompileGraphs>(backend, compile::GetMsNonlinearOps());

  vm_ = compile->CompileAndLink(ModelImpl::GetFuncGraph());
  backend_ = std::move(backend);
  MS_LOG(INFO) << "Build multi graph model success.";
  return kSuccess;
}

Status AclModelMulti::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  if (!is_multi_graph_.has_value()) {
    is_multi_graph_ = ModelImpl::GetFuncGraph() == nullptr ? false : HasMultiGraph(ModelImpl::GetFuncGraph());
  }

  if (!is_multi_graph_.value()) {
    return AclModel::Predict(inputs, outputs);
  }

  Build();
  MS_LOG(INFO) << "Start predict multi graph model.";
  MS_EXCEPTION_IF_NULL(vm_);
  MS_EXCEPTION_IF_NULL(outputs);
  try {
    (*outputs) = MSTensorRef::Convert(vm_->Eval(MSTensorRef::Convert(inputs)));
  } catch (const std::exception &ex) {
    MS_LOG(ERROR) << "Predict Failed, error: " << ex.what();
    return kMCFailed;
  }

  if (inputs_.empty()) {
    inputs_ = inputs;
  } else {
    if (inputs.size() != inputs_.size()) {
      MS_LOG(ERROR) << "Input Size is wrong.";
      return kMCFailed;
    }
    for (size_t i = 0; i < inputs_.size(); ++i) {
      auto input_tensor = MSTensor::CreateTensor(inputs_[i].Name(), inputs_[i].DataType(), inputs_[i].Shape(),
                                                 inputs[i].Data().get(), inputs[i].DataSize());
      inputs_[i] = (*input_tensor);
      MSTensor::DestroyTensorPtr(input_tensor);
    }
  }

  outputs_ = *outputs;
  MS_LOG(INFO) << "Predict multi graph model success.";
  return kSuccess;
}

void AclModelMulti::SetInputs() {
  if (inputs_.empty()) {
    auto fg = ModelImpl::GetFuncGraph();
    MS_EXCEPTION_IF_NULL(fg);
    const auto &inputs = fg->get_inputs();
    for (const auto &in : inputs) {
      auto input_param = std::dynamic_pointer_cast<Parameter>(in);
      MS_EXCEPTION_IF_NULL(input_param);
      MS_EXCEPTION_IF_NULL(input_param->abstract());
      auto input_value = input_param->abstract()->GetValueTrack();
      auto tensor = input_value->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);

      std::vector<int64_t> shape = tensor->shape_c();
      auto input_tensor = MSTensor::CreateTensor(input_param->name(), static_cast<DataType>(tensor->data_type_c()),
                                                 shape, nullptr, tensor->Size());
      inputs_.emplace_back(*input_tensor);
      MSTensor::DestroyTensorPtr(input_tensor);
    }
  } else {
    MS_LOG(DEBUG) << "inputs_ has been set.";
  }
}

std::vector<MSTensor> AclModelMulti::GetInputs() {
  if (!is_multi_graph_.has_value()) {
    is_multi_graph_ = ModelImpl::GetFuncGraph() == nullptr ? false : HasMultiGraph(ModelImpl::GetFuncGraph());
  }

  if (!is_multi_graph_.value()) {
    return AclModel::GetInputs();
  }

  return inputs_;
}

std::vector<MSTensor> AclModelMulti::GetOutputs() {
  if (!is_multi_graph_.has_value()) {
    is_multi_graph_ = ModelImpl::GetFuncGraph() == nullptr ? false : HasMultiGraph(ModelImpl::GetFuncGraph());
  }

  if (!is_multi_graph_.value()) {
    return AclModel::GetOutputs();
  }

  return outputs_;
}

namespace session {
MS_REG_SESSION(kDavinciMultiGraphInferenceDevice, MultiGraphAclSession);
}  // namespace session
}  // namespace mindspore
