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

#include <algorithm>
#include <utility>

#include "extendrt/session/default_session.h"
#include "plugin/factory/ms_factory.h"
#include "extendrt/session/factory.h"
#include "extendrt/graph_compiler/factory.h"
#include "extendrt/graph_runtime/factory.h"
#include "backend/graph_compiler/graph_partition.h"

#include "litert/cxx_api/tensor/tensor_impl.h"

namespace mindspore {
static const std::vector<PrimitivePtr> ms_infer_cut_list = {prim::kPrimReturn,   prim::kPrimPartial,
                                                            prim::kPrimSwitch,   prim::kPrimMakeTuple,
                                                            prim::kPrimBpropCut, prim::kPrimSwitchLayer};
Status DefaultInferSession::Init(const std::shared_ptr<Context> &context) {
  MS_LOG(INFO) << "DefaultInferSession::Init";
  context_ = context;

  // Set MSContext::GetInstance param?

  // init compiler and runtime according to context
  compiler_ = GraphCompilerRegistry::GetInstance()->GetCompiler(kDefaultCompiler);
  if (compiler_ == nullptr) {
    MS_LOG(ERROR) << "DefaultInferSession::Init Get Compiler is nullptr";
    return kLiteNullptr;
  }

  runtime_ = GraphRuntimRegistry::GetInstance()->GetRuntime(kDefaultRuntime);
  if (runtime_ == nullptr) {
    MS_LOG(ERROR) << "DefaultInferSession::Init Get Runtime is nullptr";
    return kLiteNullptr;
  }

  return kSuccess;
}
Status DefaultInferSession::CompileGraph(FuncGraphPtr graph, const void *data, size_t size) {
  MS_LOG(INFO) << "DefaultInferSession::CompileGraph";

  MS_LOG(DEBUG) << "DefaultInferSession::CompileGraph Compile Graph Begin";
  auto compiler = this->GetGraphCompiler();
  if (compiler == nullptr) {
    MS_LOG(ERROR) << "DefaultInferSession::CompileGraph Compiler in Infer Session is null";
    return kLiteNullptr;
  }
  auto execution_plan = compiler->Compile(graph);
  if (execution_plan == nullptr) {
    MS_LOG(ERROR) << "DefaultInferSession::CompileGraph Compile Graph Failed, Execution plan is null";
    return kLiteNullptr;
  }
  MS_LOG(DEBUG) << "DefaultInferSession::CompileGraph Compile Graph End";

  MS_LOG(DEBUG) << "DefaultInferSession::CompileGraph Prepare ExecutionPlan Begin";
  auto runtime = this->GetRuntime();
  if (runtime == nullptr) {
    MS_LOG(ERROR) << "DefaultInferSession::CompileGraph Runtime in Infer Session is null";
    return kLiteNullptr;
  }
  auto status = runtime->Prepare(execution_plan);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "DefaultInferSession::CompileGraph Prepare Execution Plan Failed";
    return status;
  }
  MS_LOG(DEBUG) << "DefaultInferSession::CompileGraph Prepare ExecutionPlan End";

  return kSuccess;
}

Status DefaultInferSession::RunGraph(const std::vector<tensor::Tensor> &inputs, std::vector<tensor::Tensor> *outputs,
                                     const MSKernelCallBack &before, const MSKernelCallBack &after) {
  MS_LOG(DEBUG) << "DefaultInferSession::RunGraph Execute ExecutionPlan Begin";
  auto runtime = this->GetRuntime();
  if (runtime_ == nullptr) {
    MS_LOG(ERROR) << "DefaultInferSession::RunGraph Runtime in Infer Session is null";
    return kLiteNullptr;
  }
  // Convert tensor::Tensor to lite::Tensor, see litert cxx_api model
  // auto inner_inputs = xxx(inputs);
  // auto inner_outputs = xxx(outputs);
  auto inner_inputs = runtime->GetInputs();
  auto inner_outputs = runtime->GetOutputs();
  auto status = CopyDataToInnerTensors(inputs, inner_inputs);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "DefaultInferSession::RunGraph Copy Data Pointer to input tensors failed";
    return status;
  }
  status = CopyDataToInnerTensors(outputs, inner_outputs);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "DefaultInferSession::RunGraph Copy Data Pointer to output tensors failed";
    return status;
  }
  status = runtime->Execute(inner_inputs, inner_outputs);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "DefaultInferSession::RunGraph Execute Execution Plan Failed";
    return status;
  }
  *outputs = LiteTensorToTensor(inner_outputs);
  if (outputs->size() != inner_outputs.size()) {
    MS_LOG(ERROR) << "DefaultInferSession::RunGraph Convert output tensors failed";
    return kLiteNullptr;
  }
  MS_LOG(DEBUG) << "DefaultInferSession::RunGraph Execute ExecutionPlan End";
  return kSuccess;
}

Status DefaultInferSession::RunGraph(const std::vector<tensor::Tensor> &inputs, std::vector<tensor::Tensor> *outputs) {
  return RunGraph(inputs, outputs, nullptr, nullptr);
}

std::vector<MutableTensorImplPtr> DefaultInferSession::GetOutputs() {
  auto runtime = this->GetRuntime();
  if (runtime_ == nullptr) {
    MS_LOG(ERROR) << "DefaultInferSession::GetOutputs Runtime in Infer Session is null";
    return kLiteNullptr;
  }
  auto lite_outputs = runtime->GetOutputs();
  MS_LOG(DEBUG) << "DefaultInferSession::GetOutputs end";
  return AbstractTensorsToTensorImpls(lite_outputs);
}

std::vector<MutableTensorImplPtr> DefaultInferSession::GetInputs() {
  auto runtime = this->GetRuntime();
  if (runtime_ == nullptr) {
    MS_LOG(ERROR) << "DefaultInferSession::GetOutputs Runtime in Infer Session is null";
    return kLiteNullptr;
  }
  auto lite_inputs = runtime->GetInputs();
  MS_LOG(DEBUG) << "DefaultInferSession::GetOutputs end";
  return AbstractTensorsToTensorImpls(lite_inputs);
}

std::vector<std::string> DefaultInferSession::GetOutputNames() { return std::vector<std::string>(); }
std::vector<std::string> DefaultInferSession::GetInputNames() { return std::vector<std::string>(); }
MutableTensorImplPtr DefaultInferSession::GetOutputByTensorName(const std::string &tensorName) { return nullptr; }
MutableTensorImplPtr DefaultInferSession::GetInputByTensorName(const std::string &name) { return nullptr; }

Status DefaultInferSession::CopyDataToInnerTensors(const std::vector<tensor::Tensor> &tensors,
                                                   std::vector<abstract::Tensor *> inner_tensors) {
  if (tensors.size() == inner_tensors.size()) {
    MS_LOG(EXCEPTION) << "user input size " << tensors.size() << " is not equal to graphp input size "
                      << inner_tensors.size();
  }
  std::vector<void *> old_data;
  for (size_t i = 0; i < tensors.size(); i++) {
    auto &user_input = tensors.at(i);
    auto input = inner_tensors.at(i);
    if (user_input.data_type() != input->data_type()) {
      // ResetTensorData(old_data, input_tensors);
      MS_LOG(EXCEPTION) << "Tensor " << user_input.id() << " has a different data type from input"
                        << input->tensor_name() << ".";
    }
    if (user_input.data_c() == nullptr) {
      // ResetTensorData(old_data, input_tensors);
      MS_LOG(EXCEPTION) << "Tensor " << user_input.id() << " has no data.";
    }
    old_data.push_back(input->data());
    if (input->data_type() == kObjectTypeString) {
      std::vector<int32_t> shape =
        TruncateShape(user_input.shape_c(), input->data_type(), user_input.DataSize(), false);
      if (shape.empty() && !(user_input.shape_c().empty())) {
        // ResetTensorData(old_data, input_tensors);
        MS_LOG(EXCEPTION) << "Input dims of tensor " << user_input.id() << " is invalid.";
      }
      input->set_shape(shape);
      input->set_data(user_input.data_c(), false);
    } else {
      if (user_input.data_c() != input->data()) {
        if (input->Size() != user_input.Size()) {
          // ResetTensorData(old_data, input_tensors);
#ifndef ENABLE_LITE_ACL
          MS_LOG(EXCEPTION) << "Tensor " << user_input.id() << " has wrong data size.";
#else
          MS_LOG(WARNING) << "Please check tensor " << user_input.id()
                          << " has been modified data size by DVPP method.";
          std::vector<int> truncate_shape = {static_cast<int>(user_input.DataSize())};
          input->set_shape(truncate_shape);
#endif
        }
        input->set_data(user_input.data_c(), false);
      }
    }
  }
  return kSuccess;
}

std::vector<MutableTensorImplPtr> &DefaultInferSession::AbstractTensorsToTensorImpls(
  const std::vector<abstract::Tensor *> &abstract_tensors) {
  std::vector<std::shared_ptr<LiteTensorImpl>> tensorImpls;
  tensorImpls.reserve(abstract_tensors.size());
  (void)std::transform(abstract_tensors.begin(), abstract_tensors.end(), std::back_inserter(tensorImpls),
                       [](abstract::Tensor *tensor) { return std::make_shared<LiteTensorImpl>(tensor); });
  return tensorImpls;
}

std::vector<mindspore::tensor::Tensor> DefaultInferSession::LiteTensorToTensor(
  const std::vector<abstract::Tensor *> &abstract_tensors) {
  std::vector<mindspore::tensor::Tensor> tensors;
  for (auto abstract_tensor : abstract_tensors) {
    if (abstract_tensor == nullptr) {
      MS_LOG(ERROR) << "DefaultInferSession::LiteTensorToTensor get nullptr tensor";
      return std::vector<mindspore::tensor::Tensor>{};
    }
    auto type_id = abstract_tensor->data_type();
    auto shape = abstract_tensor->shape();
    auto data = abstract_tensor->MutableData();
    auto data_size = abstract_tensor->Size();
    auto ref_tensor_data =
      std::make_shared<TensorRefData>(data, abstract_tensor->ElementNum(), data_size, shape.size());
    mindspore::tensor::Tensor tensor(type_id, shape, ref_tensor_data);
    auto device_address = abstract_tensor->device_data();
    if (device_address != nullptr) {
      auto lite_device_address = std::make_shared<LiteDeviceAddress>(device_address, abstract_tensor->DataSize());
      tensor.set_device_address(lite_device_address);
    }
    tensors.emplace_back(std::move(tensor));
  }
  return tensors;
}

static std::shared_ptr<InferSession> DefaultSessionCreator(const std::shared_ptr<Context> &ctx,
                                                           const ConfigInfos &config_infos) {
  auto session = std::make_shared<DefaultInferSession>(ctx);
  session->Init(ctx);
  return session;
}
REG_SESSION(kDefaultSession, DefaultSessionCreator);
}  // namespace mindspore
