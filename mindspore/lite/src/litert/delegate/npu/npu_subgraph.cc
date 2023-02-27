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

#include "src/litert/delegate/npu/npu_subgraph.h"
#include <set>
#include <unordered_map>
#include <utility>
#include "include/errorcode.h"
#include "include/graph/operator.h"
#include "include/graph/graph.h"
#include "include/graph/op/const_defs.h"
#include "include/graph/model.h"
#include "include/hiai_ir_build.h"
#include "src/common/utils.h"
#include "src/litert/delegate/npu/npu_converter_utils.h"
namespace mindspore::lite {
static std::set<mindspore::schema::PrimitiveType> npu_specific_weight_nodes = {
  schema::PrimitiveType_Conv2DFusion, schema::PrimitiveType_Conv2dTransposeFusion, schema::PrimitiveType_PadFusion,
  schema::PrimitiveType_BatchNorm,    schema::PrimitiveType_FullConnection,        schema::PrimitiveType_InstanceNorm,
  schema::PrimitiveType_TileFusion};

NPUSubGraph::~NPUSubGraph() {
  subgraph_input_ops_.clear();
  subgraph_output_ops_.clear();
  out_tensor_sorted_.clear();
  all_tensors_from_out_ops_.clear();
  for (auto op : op_buffer_) {
    delete op;
  }
  if (executor_ != nullptr) {
    delete executor_;
  }
  op_buffer_.clear();
}

void NPUSubGraph::set_input(mindspore::MSTensor in_tensor, int index) {
  MS_ASSERT(index < inputs_.size());
  auto origin_tensor = inputs_[index];
  // only in_ops_ input tensors list used in execute function
  for (auto op : in_ops_) {
    for (size_t i = 0; i < op->inputs().size(); i++) {
      if (op->inputs()[i] == origin_tensor) {
        op->set_input(in_tensor, i);
      }
    }
  }
  this->inputs_[index] = in_tensor;
}

void NPUSubGraph::set_output(mindspore::MSTensor out_tensor, int index) {
  MS_ASSERT(index < outputs_.size());
  auto origin_tensor = outputs_[index];
  for (size_t i = 0; i < all_tensors_from_out_ops_.size(); i++) {
    if (all_tensors_from_out_ops_[i] == origin_tensor) {
      all_tensors_from_out_ops_[i] = out_tensor;
    }
  }
  outputs_[index] = out_tensor;
}

int NPUSubGraph::GetGraphInOutOps() {
  for (const auto &in_tensor : this->inputs()) {
    for (auto op : npu_ops_) {
      if (find(op->inputs().begin(), op->inputs().end(), in_tensor) != op->inputs().end() &&
          find(in_ops_.begin(), in_ops_.end(), op) == in_ops_.end()) {
        in_ops_.push_back(op);
      }
    }
  }
  if (in_ops_.empty()) {
    MS_LOG(ERROR) << "Can't find the input ops for npu sub graph.";
    return RET_ERROR;
  }

  for (const auto &out_tensor : this->outputs()) {
    for (auto op : npu_ops_) {
      if (find(op->outputs().begin(), op->outputs().end(), out_tensor) != op->outputs().end() &&
          find(out_ops_.begin(), out_ops_.end(), op) == out_ops_.end()) {
        out_ops_.push_back(op);
      }
    }
  }
  if (out_ops_.empty()) {
    MS_LOG(ERROR) << "Can't find the output ops for npu sub graph.";
    return RET_ERROR;
  }
  return RET_OK;
}

std::vector<NPUOp *> NPUSubGraph::FindPreOps(NPUOp *cur_op) {
  std::vector<NPUOp *> in_ops;
  for (const auto &in_tensor : cur_op->inputs()) {
    for (auto op : npu_ops_) {
      if (find(op->outputs().begin(), op->outputs().end(), in_tensor) != op->outputs().end()) {
        in_ops.push_back(op);
      }
    }
  }
  return in_ops;
}

std::shared_ptr<domi::ModelBufferData> NPUSubGraph::BuildIRModel() {
  ge::Graph graph("NPUGraph");

  auto ret = BuildNPUInputOp();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build NPU input operator failed.";
    return nullptr;
  }
  ret = BuildNPUOutputOp();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build NPU output operator failed.";
    return nullptr;
  }
  graph.SetInputs(subgraph_input_ops_).SetOutputs(subgraph_output_ops_);
  ge::Model model(GetOMModelName(), mindspore::Version());
  model.SetGraph(graph);
  domi::HiaiIrBuild ir_build;
  auto om_model_buff = std::make_shared<domi::ModelBufferData>();
  if (om_model_buff == nullptr) {
    MS_LOG(ERROR) << "OM model buffer is nullptr.";
    return nullptr;
  }
  if (!ir_build.CreateModelBuff(model, *om_model_buff)) {
    MS_LOG(ERROR) << "Create model buffer failed.";
    return nullptr;
  }
  if (!ir_build.BuildIRModel(model, *om_model_buff)) {
    MS_LOG(ERROR) << "Build IR model failed.";
    ir_build.ReleaseModelBuff(*om_model_buff);
    return nullptr;
  }
  return om_model_buff;
}

int NPUSubGraph::Execute() { return executor_->Run(inputs(), outputs(), all_tensors_from_out_ops_, out_ops_); }

int NPUSubGraph::BuildNPUInputOp() {
  int count = 0;
  subgraph_input_ops_.clear();
  op_buffer_.clear();
  for (auto op : this->npu_ops_) {
    std::vector<ge::Operator *> input_ops;
    std::unordered_map<int, std::pair<ge::Operator *, int>> index2_multi_out_index;
    for (int i = 0; i < op->inputs().size(); ++i) {
      auto in_tensor = op->inputs()[i];
      if (IsSubGraphInputTensor(in_tensor)) {
        auto tensor_name = "Input_" + std::to_string(count++) + '_' + op->name();
        hiai::op::Data *data = ConverterToNPUData(in_tensor, tensor_name);
        subgraph_input_ops_.push_back(*data);
        input_ops.push_back(data);
        op_buffer_.push_back(data);
        continue;
      }

      bool is_weight_tensor = true;
      auto pre_ops = FindPreOps(op);
      for (auto pre_op : pre_ops) {
        if (find(pre_op->outputs().begin(), pre_op->outputs().end(), in_tensor) != pre_op->outputs().end()) {
          // input come from npu
          auto npu_op = reinterpret_cast<NPUOp *>(pre_op)->GetNPUOp();
          if (npu_op == nullptr) {
            MS_LOG(ERROR) << pre_op->name() << "'s npu operator is nullptr.";
            return RET_ERROR;
          }
          input_ops.push_back(npu_op);
          if (pre_op->outputs().size() != 1) {  // in_op has multi output, we record which output we want.
            int out_index =
              std::find(pre_op->outputs().begin(), pre_op->outputs().end(), in_tensor) - pre_op->outputs().begin();
            index2_multi_out_index[i] = {npu_op, out_index};
          }
          is_weight_tensor = false;
          break;
        }
      }

      // weight tensor
      if (is_weight_tensor) {
        if (npu_specific_weight_nodes.find(op->type()) == npu_specific_weight_nodes.end()) {
          auto name = op->name() + "_" + std::to_string(count++);
          auto weight_const = new (std::nothrow) hiai::op::Const(op->name() + "_" + std::to_string(count++));
          if (weight_const == nullptr) {
            MS_LOG(ERROR) << "New weight const failed.";
            return RET_ERROR;
          }
          auto weight_tensor = ConverterToNPUTensor(in_tensor);
          weight_const->set_attr_value(weight_tensor);
          input_ops.push_back(weight_const);
          op_buffer_.push_back(weight_const);
        }
      }
    }
    // set input to NPU
    int ret =
      reinterpret_cast<NPUOp *>(op)->SetNPUInputs(op->inputs(), op->outputs(), input_ops, index2_multi_out_index);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << op->name() << " set npu inputs failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

bool NPUSubGraph::IsSubGraphInputTensor(const mindspore::MSTensor &input) {
  if (find(this->inputs().begin(), this->inputs().end(), input) != this->inputs().end()) {
    return true;
  }
  return false;
}

int NPUSubGraph::GetNPUOperators(const std::vector<NPUOp *> &ops) {
  subgraph_output_ops_.reserve(ops.size());
  for (int i = 0; i < ops.size(); i++) {
    auto npu_op = reinterpret_cast<NPUOp *>(ops[i])->GetNPUOp();
    if (npu_op == nullptr) {
      MS_LOG(ERROR) << "Get NPU operator for " << ops[i]->name() << " failed.";
      return RET_ERROR;
    }
    subgraph_output_ops_.push_back(*npu_op);
  }
  return RET_OK;
}

int NPUSubGraph::BuildNPUOutputOp() {
  subgraph_output_ops_.clear();
  auto ret = GetNPUOperators(out_ops_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get NPU operators failed.";
    return RET_ERROR;
  }
  for (auto node : out_ops_) {
    for (const auto &tensor : node->outputs()) {
      all_tensors_from_out_ops_.emplace_back(tensor);
    }
  }
  if (subgraph_output_ops_.empty()) {
    MS_LOG(ERROR) << "NPU subgraph output op is empty.";
    return RET_ERROR;
  }
  return RET_OK;
}

std::string NPUSubGraph::GetOMModelName() { return this->name_ + ".om"; }

int NPUSubGraph::Init() {
  auto ret = GetGraphInOutOps();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get NPU subgraph input and output ops failed.";
    return RET_ERROR;
  }
  name_ = "kNpuSubGraph" + std::to_string(npu_manager_->SubGraphIndex());
  auto model_buffer_data = BuildIRModel();
  if (model_buffer_data == nullptr) {
    MS_LOG(ERROR) << "Build IR model failed.";
    return RET_ERROR;
  }

  MS_ASSERT(npu_manager_ != nullptr);
  npu_manager_->AddModel(model_buffer_data, GetOMModelName(), npu_manager_->GetFrequency());

  executor_ = new (std::nothrow) NPUExecutor(GetOMModelName(), npu_manager_);
  if (executor_ == nullptr) {
    MS_LOG(ERROR) << "Create NPUExecutor failed.";
    return RET_ERROR;
  }
  executor_->InitInputMappingRelationShip(input_relationship_);
  return RET_OK;
}

int NPUSubGraph::Prepare() {
  if (executor_->Prepare() != RET_OK) {
    MS_LOG(ERROR) << "NPU executor prepare failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::lite
