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
#include "coder/allocator/allocator.h"
#include "coder/generator/inference/inference_generator.h"
#include "coder/opcoders/op_coder_builder.h"
#include "coder/opcoders/kernel_registry.h"
#include "coder/utils/coder_utils.h"
#include "coder/log.h"
#include "src/common/ops/populate/populate_register.h"
#include "src/litert/infer_manager.h"
#include "src/litert/lite_model.h"
#include "include/errorcode.h"
#include "include/model.h"
#include "src/common/file_utils.h"
#include "src/common/prim_util.h"
#include "coder/opcoders/nnacl/dequant/de_quant.h"
#include "coder/opcoders/parallel.h"

namespace mindspore::lite::micro {
CoderSession::CoderSession() { allocator_ = MemoryAllocator::GetInstance(); }

int CoderSession::PassArgsToContext(const std::string model_name) {
  context_->set_tensor_map(allocator_->tensors_map());
  context_->set_saved_weights(allocator_->saved_weights());
  size_t de_quant_max_workspace_size = nnacl::Dequant::GetInstance()->de_quant_max_workspace();
  size_t final_total_size = allocator_->total_buffer_size() > de_quant_max_workspace_size
                              ? allocator_->total_buffer_size()
                              : de_quant_max_workspace_size;
  context_->set_total_buffer_size(final_total_size);
  context_->set_graph_inputs(coder_graph_->input_tensors());
  context_->set_graph_outputs(coder_graph_->output_tensors());
  if (Configurator::GetInstance()->debug_mode()) {
    std::vector<std::string> blocks;
    blocks = AddDumpDataInfo(context_->code_blocks(), op_coders_);
    if (blocks.size() == 0) {
      MS_LOG(ERROR) << "AddDumpDataInfo failed.";
      return RET_ERROR;
    }
    context_->set_code_blocks(blocks);
  }
  context_->set_model_name(model_name);
  return RET_OK;
}

int CoderSession::Preprocess() {
  // assign memory
  std::vector<lite::Tensor *> inputs = coder_graph_->input_tensors();
  int ret = allocator_->Assign(inputs, op_coders_);
  MS_CHECK_RET_CODE(ret, "assign memory failed");

  // prepare, init model parameters
  for (const auto &op_coder : op_coders_) {
    MS_CHECK_PTR(op_coder);
    MS_LOG(DEBUG) << "prepare: " << op_coder->name();
    ret = op_coder->Prepare(context_.get());
    MS_CHECK_RET_CODE(ret, "prepare coder " << op_coder->name() << " failed");
    allocator_->enable_is_next();
  }
  return RET_OK;
}

int CoderSession::DoCode() {
  int ret = RET_OK;
  for (const auto &op_coder : op_coders_) {
    MS_CHECK_PTR(op_coder);
    MS_LOG(DEBUG) << "code: " << op_coder->name();
    ret = op_coder->DoCode(this->context_.get());
    MS_CHECK_RET_CODE(ret, "do coder " << op_coder->name() << " failed");
  }
  return ret;
}
int CoderSession::Run(const std::string model_name) {
  MS_LOG(INFO) << "start run opcoders";

  int ret = Preprocess();
  MS_CHECK_RET_CODE(ret, "preprocess failed");

  ret = DoCode();
  MS_CHECK_RET_CODE(ret, "do code failed");

  ret = PassArgsToContext(model_name);
  MS_CHECK_RET_CODE(ret, "PassArgsToContext failed");
  MS_LOG(INFO) << "run opcoders success";
  return RET_OK;
}

int CoderSession::GenerateCode() {
  MS_LOG(INFO) << "CoderSession::GenerateCode start";
  auto generator = std::make_shared<InferenceGenerator>(std::move(context_));
  MS_CHECK_PTR(generator);

  // when use file, coder context need to remove initial parameters from tensors info
  // we use tmp_tensor_list to storage
  int ret = generator->GenerateCode();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "generate code failed";
  }
  MS_LOG(INFO) << "CoderSession::GenerateCode done";
  return ret;
}

int CoderSession::Init(const void *content, int size, const int model_index) {
  MS_LOG(INFO) << "CoderSession::Init start";
  Model *model = lite::Model::Import(static_cast<const char *>(content), size);
  MS_CHECK_PTR(model);
  coder_graph_ = std::make_unique<CoderGraph>(model);
  InitGlobalVariable(model_index);
  InitThread(model_index);
  context_ = std::make_unique<CoderContext>(model_index);
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
  std::map<Tensor *, OperatorCoder *> tensor_pre_coders;                // a tensor is a certain coder's output
  std::map<Tensor *, std::vector<OperatorCoder *>> tensor_post_coders;  // a tensor is many coder's input
  for (const auto &op_coder : op_coders_) {
    for (auto *in_tensor : op_coder->input_tensors()) {
      tensor_post_coders[in_tensor].emplace_back(op_coder.get());
    }
    for (auto *output_tensor : op_coder->output_tensors()) {
      tensor_pre_coders[output_tensor] = op_coder.get();
    }
  }
  for (const auto &op_coder : op_coders_) {
    op_coder->SetInputOps({});
    std::vector<Tensor *> inputs = op_coder->input_tensors();
    for (const auto &tensor : inputs) {
      auto item = tensor_pre_coders.find(tensor);
      if (item != tensor_pre_coders.end() && item->second != op_coder.get()) {
        op_coder->AddInputOp(item->second);
      }
    }

    op_coder->SetOutputOps({});
    std::vector<Tensor *> outputs = op_coder->output_tensors();
    for (const auto &tensor : outputs) {
      auto item = tensor_post_coders.find(tensor);
      if (item != tensor_post_coders.end()) {
        for (auto *find_coder : item->second) {
          if (find_coder == op_coder.get()) {
            continue;
          }
          op_coder->AddOutputOp(find_coder);
        }
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

OpParameter *CoderSession::GenParameterAndInfer(const LiteGraph::Node *node, const std::vector<lite::Tensor *> &inputs,
                                                std::vector<lite::Tensor *> *outputs) const {
  auto primitive = node->primitive_;
  MS_CHECK_PTR_RET_NULL(primitive);
  auto parame_gen =
    PopulateRegistry::GetInstance()->GetParameterCreator(GetPrimitiveType(primitive, schema_version_), schema_version_);
  MS_CHECK_PTR_RET_NULL(parame_gen);
  auto parameter = parame_gen(primitive);
  MS_CHECK_PTR_RET_NULL(parameter);
  auto ret = KernelInferShape(inputs, *outputs, parameter);
  if (ret == RET_INFER_INVALID) {
    MS_LOG(INFO) << "InferShape shouldn't be done before runtime, name: " << node->name_
                 << ", type: " << GetPrimitiveTypeName(primitive, schema_version_) << "flag set to false.";
  } else if (ret != RET_OK) {
    MS_LOG(ERROR) << "InferShape failed, name: " << node->name_
                  << ", type: " << GetPrimitiveTypeName(primitive, schema_version_);
    return nullptr;
  }
  return parameter;
}

int CoderSession::CreateOpCoders() {
  const Model *model = coder_graph_->model();
  if (model == nullptr) {
    MS_LOG(ERROR) << "Graph model is nullptr";
    return RET_ERROR;
  }
  schema_version_ = reinterpret_cast<const lite::LiteModel *>(model)->GetSchemaVersion();
  Configurator *config = Configurator::GetInstance();
  Target code_target = config->target();
  CodeMode code_mode = config->code_mode();
  bool support_parallel = config->support_parallel();
  uint32_t nodes_size = model->graph_.all_nodes_.size();
  std::vector<lite::Tensor *> all_tensors = coder_graph_->all_tensors();
  MS_CHECK_TRUE_MSG(!all_tensors.empty(), RET_ERROR, "coder_graph has no any tensors");
  OpCoderBuilder builder;
  for (uint32_t i = 0; i < nodes_size; ++i) {
    const auto *node = model->graph_.all_nodes_.at(i);
    if (node == nullptr) {
      MS_LOG(ERROR) << "node is nullptr";
      return RET_ERROR;
    }
    // set op_coder's inputs && outputs info
    std::vector<uint32_t> input_indices(node->input_indices_);
    std::vector<uint32_t> output_indices(node->output_indices_);
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
    for (auto out_index : output_indices) {
      out_index = static_cast<size_t>(out_index);
      if (out_index > all_tensors.size()) {
        MS_LOG(ERROR) << "out_index is invalid";
        return RET_ERROR;
      }
      outputs.push_back(all_tensors.at(out_index));
    }
    if (inputs.empty()) {
      MS_LOG(ERROR) << "node: " << node->name_ << "has no inputs tensor";
      return RET_ERROR;
    }
    if (outputs.empty()) {
      MS_LOG(ERROR) << "node: " << node->name_ << "has no outputs tensor";
      return RET_ERROR;
    }

    OpParameter *parameter = nullptr;
    if (IsCustomNode(node->primitive_, schema_version_)) {
      KernelRegistry::GetInstance()->RegisterKernel(schema::PrimitiveType_Custom);
    } else {
      parameter = GenParameterAndInfer(node, inputs, &outputs);  // built-in ops infer
      MS_CHECK_PTR(parameter);
    }

    TypeId tensor_data_type = inputs.at(0)->data_type();
    std::unique_ptr<OperatorCoder> op_coder = builder.inputs(inputs)
                                                .outputs(outputs)
                                                .node(node)
                                                .parameter(parameter)
                                                .target(code_target)
                                                .support_parallel(support_parallel)
                                                .data_type(tensor_data_type)
                                                .mode(code_mode)
                                                .input_indices(input_indices)
                                                .output_indices(output_indices)
                                                .build(schema_version_);
    if (op_coder == nullptr) {
      coder_graph_->DumpUnSupportLayer(code_target);
      return RET_ERROR;
    }
    op_coders_.push_back(std::move(op_coder));
    builder.Reset();
  }
  (void)InitOpcodersInputsAndOutputs();
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
  MS_CHECK_RET_CODE(CreateOpCoders(), "CreateOpCoders failed!");
  MS_CHECK_RET_CODE(InitTensorsRef(), "InitTensorsRefcount failed!");
  return RET_OK;
}
CoderSession::~CoderSession() { allocator_->Free(); }
}  // namespace mindspore::lite::micro
