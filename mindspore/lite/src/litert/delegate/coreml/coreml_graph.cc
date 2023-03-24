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

#include "src/litert/delegate/coreml/coreml_graph.h"
#include <fstream>
namespace mindspore::lite {
CoreMLGraph::~CoreMLGraph() {
  for (auto *kernel : all_kernels_) {
    delete kernel;
  }
  for (auto *op : coreml_ops_) {
    delete op;
  }
  for (auto tensor : insert_tensors_) {
    MSTensor::DestroyTensorPtr(tensor);
  }
  delete ml_model_;
  delete executor_wrapper_;
}

void CoreMLGraph::set_input(mindspore::MSTensor in_tensor, int index) {
  MS_ASSERT(static_cast<size_t>(index) < inputs_.size());
  auto origin_tensor = this->inputs_[index];
  for (auto kernel : all_kernels_) {
    for (size_t i = 0; i < kernel->inputs().size(); i++) {
      if (kernel->inputs()[i] == origin_tensor) {
        kernel->set_input(in_tensor, i);
      }
    }
  }
  this->inputs_[index] = in_tensor;
}

void CoreMLGraph::set_output(mindspore::MSTensor out_tensor, int index) {
  MS_ASSERT(static_cast<size_t>(index) < outputs_.size());
  auto origin_tensor = this->outputs_[index];
  for (auto kernel : all_kernels_) {
    for (size_t i = 0; i < kernel->outputs().size(); i++) {
      if (kernel->outputs()[i] == origin_tensor) {
        kernel->set_output(out_tensor, i);
      }
    }
  }
  this->outputs_[index] = out_tensor;
}

int CoreMLGraph::Init() {
  ml_model_ = BuildMLModel();
  if (ml_model_ == nullptr) {
    MS_LOG(ERROR) << "Build CoreML model failed.";
    return RET_ERROR;
  }
  auto model_path = SaveMLModel();
  executor_wrapper_ = new (std::nothrow) CoreMLExecutorWrapper();
  if (executor_wrapper_ == nullptr) {
    MS_LOG(ERROR) << "Create CoreML executor wrapper failed.";
    return RET_ERROR;
  }
  auto ret = executor_wrapper_->CompileMLModel(model_path);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Compile coreML model failed!";
    return RET_ERROR;
  }
  return RET_OK;
}

CoreML::Specification::Model *CoreMLGraph::BuildMLModel() {
  auto *model = new (std::nothrow) CoreML::Specification::Model();
  model->set_specificationversion(kCoreMLVersion4);
  model->mutable_neuralnetwork()->set_arrayinputshapemapping(CoreML::Specification::EXACT_ARRAY_MAPPING);
  auto *network = model->mutable_neuralnetwork();
  for (auto &op : coreml_ops_) {
    auto ret = op->BuildLayer();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Failed to build layer for op: " << op->name();
      delete model;
      model = nullptr;
      return nullptr;
    }
    op->SetMLOpInOut();
    auto layers = op->GetLayers();
    if (layers.empty()) {
      MS_LOG(ERROR) << "No layer found for op: " << op->name();
      delete model;
      model = nullptr;
      return nullptr;
    }
    for (auto layer : layers) {
      MS_ASSERT(layer != nullptr);
      network->mutable_layers()->AddAllocated(layer);
    }
  }
  auto ret = SetMLModelInOut(model);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set model input output failed.";
    delete model;
    model = nullptr;
    return nullptr;
  }
  return model;
}

int CoreMLGraph::SetMLModelInOut(CoreML::Specification::Model *model) {
  MS_ASSERT(model != nullptr);
  auto model_desc = model->mutable_description();
  for (const auto &in_tensor : this->inputs_) {
    // add input
    auto input = model_desc->add_input();
    input->set_name(in_tensor.Name());
    auto in_multi_array = input->mutable_type()->mutable_multiarraytype();
    if (in_tensor.DataType() == DataType::kNumberTypeFloat32) {
      in_multi_array->set_datatype(CoreML::Specification::ArrayFeatureType::FLOAT32);
    } else if (in_tensor.DataType() == DataType::kNumberTypeInt32) {
      in_multi_array->set_datatype(CoreML::Specification::ArrayFeatureType::INT32);
    } else {
      MS_LOG(ERROR) << "Unsupported model input data type: " << static_cast<int>(in_tensor.DataType());
      return RET_ERROR;
    }
    for (int64_t i : in_tensor.Shape()) {
      in_multi_array->add_shape(static_cast<uint64_t>(i));
    }
  }
  for (const auto &out_tensor : this->outputs_) {
    // add output
    auto output = model_desc->add_output();
    output->set_name(out_tensor.Name());
    auto out_multi_array = output->mutable_type()->mutable_multiarraytype();
    if (out_tensor.DataType() == DataType::kNumberTypeFloat32) {
      out_multi_array->set_datatype(CoreML::Specification::ArrayFeatureType::FLOAT32);
    } else if (out_tensor.DataType() == DataType::kNumberTypeInt32) {
      out_multi_array->set_datatype(CoreML::Specification::ArrayFeatureType::INT32);
    } else {
      MS_LOG(ERROR) << "Unsupported model output data type: " << static_cast<int>(out_tensor.DataType());
      return RET_ERROR;
    }
    for (int64_t i : out_tensor.Shape()) {
      out_multi_array->add_shape(static_cast<uint64_t>(i));
    }
  }
  return RET_OK;
}

std::string CoreMLGraph::SaveMLModel() {
  MS_ASSERT(ml_model_ != nullptr);
  std::string model_name = this->name() + ".mlmodel";
  auto model_path = std::string(getenv("HOME")) + "/tmp/" + model_name;
  std::ofstream file_stream(model_path, std::ios::out | std::ios::binary);
  ml_model_->SerializeToOstream(&file_stream);
  MS_LOG(INFO) << "Build CoreML model success!";
  return model_path;
}

int CoreMLGraph::Execute() {
  auto ret = executor_wrapper_->Run(inputs(), outputs());
  MS_LOG(INFO) << "run model success!";
  return ret;
}
}  // namespace mindspore::lite
