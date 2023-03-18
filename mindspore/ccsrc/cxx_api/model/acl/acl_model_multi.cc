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
#include <numeric>
#include <deque>
#include <functional>
#include "cxx_api/factory.h"
#include "acl/acl_rt.h"
#include "mindspore/core/load_mindir/infer_mindir.h"
#include "cxx_api/model/acl/acl_vm/ms_tensor_ref.h"
#include "cxx_api/model/acl/acl_vm/acl_vm.h"

namespace mindspore {
API_MODEL_REG(Ascend310, AclModelMulti);

namespace {
std::map<DataType, size_t> kDtypeMap = {
  {DataType::kNumberTypeBool, sizeof(bool)},       {DataType::kNumberTypeInt8, sizeof(int8_t)},
  {DataType::kNumberTypeInt16, sizeof(int16_t)},   {DataType::kNumberTypeInt32, sizeof(int32_t)},
  {DataType::kNumberTypeInt64, sizeof(int64_t)},   {DataType::kNumberTypeFloat16, sizeof(float16)},
  {DataType::kNumberTypeFloat32, sizeof(float)},   {DataType::kNumberTypeFloat64, sizeof(double)},
  {DataType::kNumberTypeUInt8, sizeof(uint8_t)},   {DataType::kNumberTypeUInt16, sizeof(uint16_t)},
  {DataType::kNumberTypeUInt32, sizeof(uint32_t)}, {DataType::kNumberTypeUInt64, sizeof(uint64_t)}};

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
                         if (abstract->GetValueTrack() != kValueAny) {
                           return abstract->Broaden();
                         }
                         return abstract;
                       });
  try {
    (void)InferMindir(ModelImpl::GetFuncGraph(), broaded_args);
  } catch (const std::runtime_error &e) {
    MS_LOG(ERROR) << "Infer mindir for sub graph failed: " << e.what();
    return kMCFailed;
  }

  // set output
  SetOutput();
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

  auto ret = Build();
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Build multi-graph model as default options failed.";
    return ret;
  }
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
      auto input_abs = input_param->abstract();
      MS_EXCEPTION_IF_NULL(input_abs);
      auto tensor_abs = input_abs->cast<abstract::AbstractTensorPtr>();
      if (tensor_abs == nullptr) {
        MS_LOG(EXCEPTION) << "The graph input type is not a tensor. input args info:" << input_abs->ToString();
      }
      auto shape_ptr = tensor_abs->BuildShape();
      MS_EXCEPTION_IF_NULL(shape_ptr);
      auto tensor_shape = shape_ptr->cast<abstract::ShapePtr>();
      MS_EXCEPTION_IF_NULL(tensor_shape);
      auto elem = tensor_abs->element();
      MS_EXCEPTION_IF_NULL(elem);
      auto type_id = elem->BuildType()->type_id();
      auto tensor = std::make_shared<tensor::Tensor>(type_id, tensor_shape->shape());

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

void AclModelMulti::SetOutput() {
  if (outputs_.empty()) {
    auto fg = ModelImpl::GetFuncGraph();
    MS_EXCEPTION_IF_NULL(fg);
    const auto output = fg->output();
    MS_EXCEPTION_IF_NULL(output);
    auto abs = output->abstract();
    MS_EXCEPTION_IF_NULL(abs);

    // DataType
    DataType type_id;
    if (abs->isa<abstract::AbstractTensor>()) {
      auto abs_tensor = abs->cast<abstract::AbstractTensorPtr>();
      auto ele = abs_tensor->element();
      MS_EXCEPTION_IF_NULL(ele);
      MS_EXCEPTION_IF_NULL(ele->GetTypeTrack());
      type_id = static_cast<DataType>(ele->GetTypeTrack()->type_id());
    } else {
      MS_EXCEPTION_IF_NULL(abs->GetTypeTrack());
      type_id = static_cast<DataType>(abs->GetTypeTrack()->type_id());
    }
    // Shape
    auto shape_track = abs->GetShapeTrack();
    MS_EXCEPTION_IF_NULL(shape_track);
    std::vector<int64_t> shape = {};
    if (shape_track->isa<abstract::Shape>()) {
      auto shapeptr = shape_track->cast<abstract::ShapePtr>();
      shape = static_cast<std::vector<int64_t>>(shapeptr->shape());
    }
    // Size
    size_t ato_size = 0;
    if (kDtypeMap.find(type_id) != kDtypeMap.end()) {
      ato_size = kDtypeMap[type_id];
    }
    int64_t ele_num = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
    size_t size = ato_size * LongToSize(ele_num);
    // create tensor
    auto output_tensor = MSTensor::CreateTensor("", type_id, shape, nullptr, size);
    outputs_.emplace_back(*output_tensor);
    MSTensor::DestroyTensorPtr(output_tensor);
  } else {
    MS_LOG(DEBUG) << "outputs_ has been set.";
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
}  // namespace mindspore
