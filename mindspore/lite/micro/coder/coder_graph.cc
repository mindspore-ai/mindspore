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

#include "micro/coder/coder_graph.h"
#include <queue>
#include <deque>
#include <string>
#include <memory>
#include <algorithm>
#include <set>
#include "schema/inner/model_generated.h"
#include "src/ops/primitive_c.h"
namespace mindspore::lite::micro {
CoderGraph::~CoderGraph() {
  model_->Free();
  delete model_;
  for (auto &tensor : all_tensors_) {
    delete tensor;
  }
}

std::vector<lite::Tensor *> CoderGraph::input_tensors() const { return input_tensors_; }

std::vector<lite::Tensor *> CoderGraph::output_tensors() const { return output_tensors_; }

void CoderGraph::InitInputs() {
  for (const auto &pair : inputs_map_) {
    std::vector<Tensor *> tensors = pair.second;
    input_tensors_.insert(input_tensors_.end(), tensors.begin(), tensors.end());
  }
  // remove duplicate tensors
  std::set<lite::Tensor *> unique;
  unique.insert(input_tensors_.begin(), input_tensors_.end());
  input_tensors_.clear();
  input_tensors_.insert(input_tensors_.end(), unique.begin(), unique.end());
}

void CoderGraph::InitOutputs() {
  std::transform(output_indices_.begin(), output_indices_.end(), std::back_inserter(output_tensors_),
                 [&](uint32_t a) { return this->all_tensors_.at(a); });
}

void CoderGraph::SetAllTensors(const std::vector<Tensor *> &all_tensors) {
  all_tensors_.insert(all_tensors_.end(), all_tensors.begin(), all_tensors.end());
}

void CoderGraph::SetInputIndices(const std::vector<uint32_t> &input_indices) {
  input_indices_.insert(input_indices_.end(), input_indices.begin(), input_indices.end());
}

void CoderGraph::SetOutputIndices(const std::vector<uint32_t> &output_indices) {
  output_indices_.insert(output_indices_.end(), output_indices.begin(), output_indices.end());
}

void CoderGraph::AddInputMap(const std::string &node_id, Tensor *input_tensor) {
  if (!input_tensor) {
    MS_LOG(ERROR) << "input tensor is nullptr, can not added to coder_graph";
    return;
  }
  this->inputs_map_[node_id].emplace_back(input_tensor);
}

void CoderGraph::AddOutputMap(const std::string &node_id, Tensor *output_tensor) {
  if (!output_tensor) {
    MS_LOG(ERROR) << "output tensor is nullptr, can not added to coder_graph";
    return;
  }
  this->outputs_map_[node_id].emplace_back(output_tensor);
}

std::vector<lite::Tensor *> CoderGraph::all_tensors() const { return this->all_tensors_; }

const std::map<std::string, std::vector<lite::Tensor *>> &CoderGraph::GetOutputsMap() const { return outputs_map_; }

std::vector<uint32_t> CoderGraph::input_indices() const { return this->input_indices_; }

std::vector<uint32_t> CoderGraph::output_indices() const { return this->output_indices_; }
}  // namespace mindspore::lite::micro
