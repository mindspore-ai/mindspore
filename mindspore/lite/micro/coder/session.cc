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
#include <queue>
#include <vector>
#include <utility>
#include "coder/allocator/allocator.h"
#include "coder/context.h"
#include "coder/generator/generator.h"
#include "coder/generator/inference/inference_generator.h"
#include "coder/opcoders/op_coder_builder.h"
#include "coder/utils/coder_utils.h"
#include "coder/log.h"
#include "include/errorcode.h"
#include "src/common/file_utils.h"

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

    PrimitiveC *primitive = curr_node->primitive_;
    if (primitive == nullptr) {
      MS_LOG(ERROR) << "Op " << curr_node->name_ << " should exist in model!";
      return RET_ERROR;
    }
    primitive->set_infer_flag(true);
    int ret = primitive->InferShape(inputs, outputs);
    if (ret == RET_INFER_INVALID) {
      MS_LOG(INFO) << "InferShape shouldn't be done before runtime, name: " << curr_node->name_
                   << ", type: " << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(primitive->Type()))
                   << "flag set to false.";
      primitive->set_infer_flag(false);
    } else if (ret != RET_OK) {
      MS_LOG(ERROR) << "InferShape failed, name: " << curr_node->name_ << ", type: "
                    << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(primitive->Type()));
      return RET_ERROR;
    }
  }
  return RET_OK;
}

void CoderSession::EndCode() {
  context_->set_tensor_map(allocator_->tensors_map());
  context_->set_saved_weights(allocator_->saved_weights());
  context_->set_total_buffer_size(allocator_->total_buffer_size());
  context_->set_graph_inputs(coder_graph_->input_tensors());
  context_->set_graph_outputs(coder_graph_->output_tensors());
  Configurator *config = Configurator::GetInstance();
  if (config->debug_mode()) {
    std::vector<std::string> blocks;
    blocks = AddDumpDataInfo(context_->code_blocks(), op_coders_);
    context_->set_code_blocks(blocks);
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
    ret = op_coder->Prepare(context_.get());
    MS_CHECK_RET_CODE(ret, "prepare coder " << op_coder->ID() << " failed");
    allocator_->enable_is_next();
  }
  // 3. docode, write operator code
  for (const auto &op_coder : op_coders_) {
    MS_CHECK_PTR(op_coder);
    ret = op_coder->DoCode(this->context_.get());
    MS_CHECK_RET_CODE(ret, "do coder " << op_coder->ID() << " failed");
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
    case Code_Normal:
    case Code_Android:
      MS_LOG(INFO) << "generate code for Android";
      generator = std::make_shared<InferenceGenerator>(std::move(context_));
      break;
    default:
      MS_LOG(ERROR) << "unsupported generator code mode, " << code_mode;
      return RET_ERROR;
  }
  // when use file, coder context need to remove initial parameters from tensors info
  // we use tmp_tensor_list to storage
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
  MS_CHECK_PTR(graph_buf);
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

int CoderSession::ConvertTensors() {
  auto model = coder_graph_->model();
  if (model == nullptr) {
    MS_LOG(ERROR) << "Graph model is nullptr";
    return RET_ERROR;
  }
  std::vector<Tensor *> all_tensors;
  auto clear_tensors = [&all_tensors]() {
    std::for_each(all_tensors.begin(), all_tensors.end(), [](Tensor *&t) {
      delete t;
      t = nullptr;
    });
    all_tensors.clear();
  };
  auto check_dim = [](int dim) -> int {
    MS_CHECK_TRUE(dim > 0, "invalid dim value!");
    return RET_OK;
  };

  // deal with allTensors
  uint32_t tensorCount = model->all_tensors_.size();
  for (uint32_t i = 0; i < tensorCount; ++i) {
    schema::Tensor *origin_tensor = model->all_tensors_.at(i);
    MS_CHECK_PTR_WITH_EXE(origin_tensor, clear_tensors());
    // tensor dims
    std::vector<int> shape;
    if (origin_tensor->nodeType() == schema::NodeType_ValueNode) {
      MS_CHECK_PTR_WITH_EXE(origin_tensor->dims(), clear_tensors());
      for (uint32_t j = 0; j < origin_tensor->dims()->size(); j++) {
        MS_CHECK_PTR(origin_tensor->dims()->data());
        int dim = static_cast<int>(origin_tensor->dims()->data()[j]);
        MS_CHECK_RET_CODE_WITH_EXE(check_dim(dim), "parse shape failed!", clear_tensors());
        shape.push_back(dim);
      }
    }
    // tensor Datatype
    int origin_data_type = static_cast<int>(origin_tensor->dataType());
    Tensor *dstTensor = new (std::nothrow)
      lite::Tensor(TypeId(origin_data_type), shape, origin_tensor->format(), TensorCategory(origin_tensor));
    MS_CHECK_PTR(dstTensor);
    if (origin_tensor->nodeType() == schema::NodeType_ValueNode && origin_tensor->data() != nullptr &&
        origin_tensor->data()->size() > 0) {
      if (shape.empty()) {
        shape.push_back(1);
      }
      // copy data, this is weight && bias
      MS_CHECK_TRUE_WITH_EXE(origin_tensor->data()->size() > 0, "invalid meta_tensor data size.", delete dstTensor);
      auto data_size = static_cast<size_t>(origin_tensor->data()->size());
      MS_CHECK_RET_CODE_WITH_EXE(dstTensor->MallocData(), "dst tensor malloc data failed!", delete dstTensor);
      void *dst_data = dstTensor->data_c();
      MS_CHECK_RET_CODE_WITH_EXE(memcpy_s(dst_data, data_size, origin_tensor->data()->data(), data_size),
                                 "memcpy_s copy data failed!", delete dstTensor);
      dstTensor->set_data(dst_data);
    }
    auto quant_params = origin_tensor->quantParams();
    if (quant_params != nullptr) {
      for (int j = 0; j < static_cast<int>(quant_params->size()); j++) {
        QuantArg quant_arg{};
        quant_arg.scale = quant_params->Get(j)->scale();
        quant_arg.zeroPoint = quant_params->Get(j)->zeroPoint();
        dstTensor->AddQuantParam(quant_arg);
      }
    }
    all_tensors.emplace_back(dstTensor);
  }
  coder_graph_->SetAllTensors(all_tensors);
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

int CoderSession::InitGraphInOutTensors() {
  const Model *model = coder_graph_->model();
  if (model == nullptr) {
    return RET_ERROR;
  }
  std::vector<size_t> graph_input_node_indexes = lite::GetGraphInputNodes(model);
  std::vector<uint32_t> input_indices;
  for (auto in_node_index : graph_input_node_indexes) {
    in_node_index = static_cast<uint32_t>(in_node_index);
    auto *in_node = model->all_nodes_.at(in_node_index);
    if (in_node == nullptr) {
      return RET_ERROR;
    }
    for (uint32_t i = 0; i < in_node->input_indices_.size(); i++) {
      auto in_tensor_index = size_t(in_node->input_indices_.at(i));
      bool is_graph_input = false;
      for (uint32_t j = 0; j < model->sub_graphs_.at(0)->input_indices_.size(); j++) {
        if (in_tensor_index == size_t(model->sub_graphs_.at(0)->input_indices_.at(j))) {
          input_indices.push_back(static_cast<uint32_t>(in_tensor_index));
          is_graph_input = true;
          break;
        }
      }
      if (!is_graph_input) {
        continue;
      }
      if (in_tensor_index < coder_graph_->all_tensors().size()) {
        lite::Tensor *in_tensor = this->coder_graph_->all_tensors().at(in_tensor_index);
        coder_graph_->AddInputMap(in_node->name_, in_tensor);
      }
    }
  }
  coder_graph_->SetInputIndices(input_indices);
  std::vector<uint32_t> output_indices;
  auto graph_output_node_indexes = lite::GetGraphOutputNodes(model);
  for (auto out_node_index : graph_output_node_indexes) {
    out_node_index = static_cast<uint32_t>(out_node_index);
    auto *out_node = model->all_nodes_.at(out_node_index);
    for (uint32_t i = 0; i < out_node->output_indices_.size(); i++) {
      auto out_tensor_index = size_t(out_node->output_indices_.at(i));
      bool is_graph_output = false;
      for (uint32_t j = 0; j < model->sub_graphs_.at(0)->output_indices_.size(); j++) {
        if (out_tensor_index == size_t(model->sub_graphs_.at(0)->output_indices_.at(j))) {
          output_indices.push_back(static_cast<uint32_t>(out_tensor_index));
          is_graph_output = true;
          break;
        }
      }
      if (!is_graph_output) {
        continue;
      }
      if (out_tensor_index < coder_graph_->all_tensors().size()) {
        lite::Tensor *out_tensor = this->coder_graph_->all_tensors().at(out_tensor_index);
        if (out_tensor == nullptr) {
          MS_LOG(ERROR) << "can not find any output tensor in all_tensors";
          return RET_ERROR;
        }
        coder_graph_->AddOutputMap(out_node->name_, out_tensor);
      }
    }
  }
  coder_graph_->SetOutputIndices(output_indices);
  coder_graph_->InitInputs();
  coder_graph_->InitOutputs();
  return RET_OK;
}

int CoderSession::CompileGraph() {
  MS_LOG(INFO) << "CompileGraph";
  MS_CHECK_RET_CODE(ConvertTensors(), "ConvertTensors failed");
  MS_CHECK_RET_CODE(InitGraphInOutTensors(), "InitGraphInOutTensors failed");
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
