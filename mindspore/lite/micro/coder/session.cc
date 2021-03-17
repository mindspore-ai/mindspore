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

#include "coder/session.h"
#include <set>
#include <vector>
#include <utility>
#include "coder/context.h"
#include "coder/train.h"
#include "coder/allocator/allocator.h"
#include "coder/generator/generator.h"
#include "coder/generator/inference/inference_generator.h"
#include "coder/generator/train/train_generator.h"
#include "coder/opcoders/op_coder_builder.h"
#include "coder/utils/coder_utils.h"
#include "coder/log.h"
#include "src/ops/populate/populate_register.h"
#include "src/common/version_manager.h"
#include "src/runtime/infer_manager.h"
#include "src/scheduler.h"
#include "include/errorcode.h"
#include "src/common/file_utils.h"
#include "coder/opcoders/nnacl/dequant/de_quant.h"

namespace mindspore::lite::micro {

CoderSession::CoderSession() { allocator_ = MemoryAllocator::GetInstance(); }

int CoderSession::InferShape() {
  const Model *model = coder_graph_->model();
  std::vector<lite::Tensor *> all_tensors = coder_graph_->all_tensors();
  size_t nodes_num = model->all_nodes_.size();
  for (size_t i = 0; i < nodes_num; ++i) {
    auto curr_node = model->all_nodes_.at(i);
    if (!curr_node) {
      MS_LOG(ERROR) << "model's node is null, who's index is " << i << ". InferShape failed ";
      return RET_ERROR;
    }
    std::vector<Tensor *> inputs;
    std::vector<Tensor *> outputs;
    size_t input_nums = curr_node->input_indices_.size();
    inputs.reserve(input_nums);
    for (size_t j = 0; j < input_nums; ++j) {
      inputs.push_back(all_tensors.at(curr_node->input_indices_.at(j)));
    }
    size_t output_nums = curr_node->output_indices_.size();
    outputs.reserve(output_nums);
    for (size_t j = 0; j < output_nums; ++j) {
      outputs.push_back(all_tensors.at(curr_node->output_indices_.at(j)));
    }

    auto primitive = curr_node->primitive_;
    MS_CHECK_PTR(primitive);
    int schema_version = VersionManager::GetInstance()->GetSchemaVersion();
    auto parame_gen = PopulateRegistry::GetInstance()->GetParameterCreator(GetPrimitiveType(primitive), schema_version);
    if (parame_gen == nullptr) {
      MS_LOG(ERROR) << "parameter generator is nullptr.";
      return RET_NULL_PTR;
    }
    auto parameter = parame_gen(primitive);
    if (parameter == nullptr) {
      MS_LOG(ERROR) << "PopulateParameter return nullptr, type: " << PrimitiveTypeName(GetPrimitiveType(primitive));
      return RET_ERROR;
    }
    parameter->infer_flag_ = true;
    auto ret = KernelInferShape(inputs, &outputs, parameter);
    if (ret == RET_INFER_INVALID) {
      MS_LOG(INFO) << "InferShape shouldn't be done before runtime, name: " << curr_node->name_
                   << ", type: " << PrimitiveTypeName(GetPrimitiveType(primitive)) << "flag set to false.";
      parameter->infer_flag_ = false;
    } else if (ret != RET_OK) {
      MS_LOG(ERROR) << "InferShape failed, name: " << curr_node->name_
                    << ", type: " << PrimitiveTypeName(GetPrimitiveType(primitive));
      return RET_ERROR;
    }
  }
  return RET_OK;
}

void CoderSession::EndCode() {
  context_->set_tensor_map(allocator_->tensors_map());
  context_->set_saved_weights(allocator_->saved_weights());
  size_t de_quant_max_workspace_size = nnacl::Dequant::GetInstance()->de_quant_max_workspace();
  size_t final_total_size = allocator_->total_buffer_size() > de_quant_max_workspace_size
                              ? allocator_->total_buffer_size()
                              : de_quant_max_workspace_size;
  context_->set_total_buffer_size(final_total_size);
  context_->set_graph_inputs(coder_graph_->input_tensors());
  context_->set_graph_outputs(coder_graph_->output_tensors());
  Configurator *config = Configurator::GetInstance();
  if (config->debug_mode()) {
    std::vector<std::string> blocks;
    blocks = AddDumpDataInfo(context_->code_blocks(), op_coders_);
    context_->set_code_blocks(blocks);
  }
  if (config->code_mode() == Train) {
    Train::TransformGraphForTrain(context_.get(), op_coders_);
  }
}

int CoderSession::Run() {
  MS_LOG(INFO) << "start run opcoders";
  // 1. assign memory
  std::vector<lite::Tensor *> inputs = coder_graph_->input_tensors();
  int ret = allocator_->Assign(inputs, op_coders_);
  MS_CHECK_RET_CODE(ret, "assign memory failed");
  // 2. prepare, init model parameters
  for (const auto &op_coder : op_coders_) {
    MS_CHECK_PTR(op_coder);
    MS_LOG(DEBUG) << "prepare: " << op_coder->name();
    ret = op_coder->Prepare(context_.get());
    MS_CHECK_RET_CODE(ret, "prepare coder " << op_coder->name() << " failed");
    allocator_->enable_is_next();
  }
  // 3. docode, write operator code
  for (const auto &op_coder : op_coders_) {
    MS_CHECK_PTR(op_coder);
    MS_LOG(DEBUG) << "code: " << op_coder->name();
    ret = op_coder->DoCode(this->context_.get());
    MS_CHECK_RET_CODE(ret, "do coder " << op_coder->name() << " failed");
  }

  this->EndCode();
  MS_LOG(INFO) << "run opcoders success";
  return RET_OK;
}

int CoderSession::GenerateCode() {
  MS_LOG(INFO) << "CoderSession::GenerateCode start";
  std::shared_ptr<Generator> generator;
  Configurator *config = Configurator::GetInstance();
  CodeMode code_mode = config->code_mode();
  switch (code_mode) {
    case Inference:
      MS_LOG(INFO) << "generate code for Inference";
      generator = std::make_shared<InferenceGenerator>(std::move(context_));
      break;
    case Train:
      MS_LOG(INFO) << "generate code for Train";
      generator = std::make_shared<TrainGenerator>(std::move(context_));
      break;
    default:
      MS_LOG(ERROR) << "unsupported generator code mode, " << code_mode;
      return RET_ERROR;
  }
  // when use file, coder context need to remove initial parameters from tensors info
  // we use tmp_tensor_list to storage
  MS_CHECK_PTR(generator);
  int ret = generator->GenerateCode();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "generate code failed";
  }
  MS_LOG(INFO) << "CoderSession::GenerateCode done";
  return ret;
}

int CoderSession::Init(const std::string &model_path) {
  MS_LOG(INFO) << "CoderSession::Init start";
  // Load graph
  MS_LOG(DEBUG) << "start reading model file";
  size_t size = 0;
  char *graph_buf = ReadFile(model_path.c_str(), &size);
  if (graph_buf == nullptr) {
    MS_LOG(ERROR) << "read model file from path \"" << model_path << "\" failed.";
    return RET_ERROR;
  }
  // new a context for session
  if (size >= UINT_MAX) {
    MS_LOG(ERROR) << "the size is invalid";
    delete[] graph_buf;
    return RET_ERROR;
  }
  Model *model = lite::Model::Import(graph_buf, size);
  delete[] graph_buf;
  MS_CHECK_PTR(model);
  coder_graph_ = std::make_unique<CoderGraph>(model);
  context_ = std::make_unique<CoderContext>();
  allocator_->RecordRuntimeAddrs(context_->input_name(), context_->buffer_name(), context_->weight_name());
  MS_LOG(INFO) << "CoderSession::Init done";
  return RET_OK;
}

int CoderSession::Build() {
  if (coder_graph_ == nullptr) {
    return RET_ERROR;
  }
  int ret = this->CompileGraph();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CompileGraph failed: " << ret;
    return ret;
  }
  return RET_OK;
}

int CoderSession::InitOpcodersInputsAndOutputs() {
  std::map<Tensor *, OperatorCoder *> input_node_map;
  std::map<Tensor *, OperatorCoder *> output_node_map;
  for (const auto &op_coder : op_coders_) {
    std::vector<Tensor *> inputs = op_coder->input_tensors();
    std::for_each(inputs.begin(), inputs.end(),
                  [&](Tensor *t) { input_node_map.insert(std::make_pair(t, op_coder.get())); });
    std::vector<Tensor *> outputs = op_coder->input_tensors();
    std::for_each(outputs.begin(), outputs.end(),
                  [&](Tensor *t) { output_node_map.insert(std::make_pair(t, op_coder.get())); });
  }
  for (const auto &op_coder : op_coders_) {
    std::vector<Tensor *> inputs = op_coder->input_tensors();
    for (const auto &tensor : inputs) {
      auto item = output_node_map.find(tensor);
      if (item != output_node_map.end()) {
        op_coder->AddInputOp(item->second);
      }
    }
    std::vector<Tensor *> outputs = op_coder->output_tensors();
    for (const auto &tensor : outputs) {
      auto item = input_node_map.find(tensor);
      if (item != input_node_map.end()) {
        op_coder->AddOutputOp(item->second);
      }
    }
  }
  return RET_OK;
}

int CoderSession::InitTensorsRef() {
  auto all_tensors = coder_graph_->all_tensors();
  for (auto &tensor : all_tensors) {
    size_t refcount = 0;
    for (const auto &node : this->op_coders_) {
      auto inputs = node->input_tensors();
      auto iter = std::find(inputs.begin(), inputs.end(), tensor);
      if (iter != inputs.end()) {
        refcount++;
      }
    }
    tensor->set_ref_count(refcount);
  }
  return RET_OK;
}

int CoderSession::CreateOpCoders() {
  const Model *model = coder_graph_->model();
  if (model == nullptr) {
    MS_LOG(ERROR) << "Graph model is nullptr";
    return RET_ERROR;
  }
  Configurator *config = Configurator::GetInstance();
  Target code_target = config->target();
  CodeMode code_mode = config->code_mode();
  bool support_parallel = config->support_parallel();
  uint32_t nodes_size = model->all_nodes_.size();
  OpCoderBuilder builder;
  for (uint32_t i = 0; i < nodes_size; ++i) {
    const auto *node = model->all_nodes_.at(i);
    if (node == nullptr) {
      MS_LOG(ERROR) << "node is nullptr";
      return RET_ERROR;
    }
    std::vector<lite::Tensor *> all_tensors = coder_graph_->all_tensors();
    if (all_tensors.empty()) {
      MS_LOG(ERROR) << "coder_graph has no any tensors";
      return RET_ERROR;
    }
    // set op_coder's inputs && outputs info
    std::vector<uint32_t> input_indices;
    Uint32Vector node_input_indices = node->input_indices_;
    input_indices.insert(input_indices.end(), node_input_indices.begin(), node_input_indices.end());
    std::vector<uint32_t> output_indices;
    Uint32Vector node_output_indices = node->output_indices_;
    output_indices.insert(output_indices.end(), node_output_indices.begin(), node_output_indices.end());

    std::vector<lite::Tensor *> inputs;
    std::vector<lite::Tensor *> outputs;
    for (auto in_index : input_indices) {
      in_index = static_cast<size_t>(in_index);
      if (in_index > all_tensors.size()) {
        MS_LOG(ERROR) << "in_index is invalid";
        return RET_ERROR;
      }
      inputs.push_back(all_tensors.at(in_index));
    }
    for (auto ou_index : output_indices) {
      ou_index = static_cast<size_t>(ou_index);
      if (ou_index > all_tensors.size()) {
        MS_LOG(ERROR) << "ou_index is invalid";
        return RET_ERROR;
      }
      outputs.push_back(all_tensors.at(ou_index));
    }
    if (inputs.empty()) {
      MS_LOG(ERROR) << "node: " << node->name_ << "has  no inputs tensor";
      return RET_ERROR;
    }
    if (outputs.empty()) {
      MS_LOG(ERROR) << "node: " << node->name_ << "has  no outputs tensor";
      return RET_ERROR;
    }

    TypeId tensor_data_type = inputs.at(0)->data_type();
    std::unique_ptr<OperatorCoder> op_coder = builder.inputs(inputs)
                                                .outputs(outputs)
                                                .node(node)
                                                .target(code_target)
                                                .support_parallel(support_parallel)
                                                .data_type(tensor_data_type)
                                                .mode(code_mode)
                                                .input_indices(input_indices)
                                                .output_indices(output_indices)
                                                .build();
    MS_CHECK_PTR(op_coder);
    op_coders_.push_back(std::move(op_coder));
    builder.Reset();
  }
  InitOpcodersInputsAndOutputs();
  return RET_OK;
}

int CoderSession::InitCodeGraph() {
  MS_CHECK_RET_CODE(coder_graph_->ConvertTensors(), "convert tensors failed");
  MS_CHECK_RET_CODE(coder_graph_->InitGraphInOutTensors(), "init graph inputs and outputs failed");
  return RET_OK;
}

int CoderSession::CompileGraph() {
  MS_LOG(INFO) << "CompileGraph";
  MS_CHECK_RET_CODE(InitCodeGraph(), "InitGraphInOutTensors failed");
  MS_CHECK_RET_CODE(InferShape(), "do infershape failed!");
  MS_CHECK_RET_CODE(CreateOpCoders(), "CreateOpCoders failed!");
  MS_CHECK_RET_CODE(InitTensorsRef(), "InitTensorsRefcount failed!");
  return RET_OK;
}

std::shared_ptr<CoderSession> CreateCoderSession() {
  auto session = std::make_shared<CoderSession>();
  return session;
}

CoderSession::~CoderSession() { allocator_->Free(); }

}  // namespace mindspore::lite::micro
