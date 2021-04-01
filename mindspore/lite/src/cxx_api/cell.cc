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

#include "include/api/cell.h"
#include "src/common/log_adapter.h"

namespace mindspore {
class GraphImpl {};

std::vector<Output> CellBase::operator()(const std::vector<Input> &inputs) const {
  std::vector<Output> empty;
  MS_LOG(ERROR) << "Unsupported feature.";
  return empty;
}

ParameterCell::ParameterCell(const ParameterCell &cell) { MS_LOG(ERROR) << "Unsupported feature."; }
ParameterCell &ParameterCell::operator=(const ParameterCell &cell) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return *this;
}

ParameterCell::ParameterCell(ParameterCell &&cell) { MS_LOG(ERROR) << "Unsupported feature."; }

ParameterCell &ParameterCell::operator=(ParameterCell &&cell) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return *this;
}

ParameterCell::ParameterCell(const MSTensor &tensor) { MS_LOG(ERROR) << "Unsupported feature."; }

ParameterCell &ParameterCell::operator=(const MSTensor &tensor) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return *this;
}

ParameterCell::ParameterCell(MSTensor &&tensor) : tensor_(tensor) { MS_LOG(ERROR) << "Unsupported feature."; }

ParameterCell &ParameterCell::operator=(MSTensor &&tensor) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return *this;
}

GraphCell::GraphCell(const Graph &graph) : graph_(std::shared_ptr<Graph>(new (std::nothrow) Graph(graph))) {
  if (graph_ == nullptr) {
    MS_LOG(ERROR) << "Invalid graph.";
  }
}

GraphCell::GraphCell(const std::shared_ptr<Graph> &graph) : graph_(graph) {
  if (graph_ == nullptr) {
    MS_LOG(ERROR) << "Invalid graph.";
  }
}

GraphCell::GraphCell(Graph &&graph) : graph_(std::shared_ptr<Graph>(new (std::nothrow) Graph(graph))) {
  if (graph_ == nullptr) {
    MS_LOG(ERROR) << "Invalid graph.";
  }
}

Status GraphCell::Run(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return kLiteError;
}

Status GraphCell::Load(uint32_t device_id) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return kLiteError;
}

InputAndOutput::InputAndOutput() { MS_LOG(ERROR) << "Unsupported feature."; }

InputAndOutput::InputAndOutput(const MSTensor &tensor) { MS_LOG(ERROR) << "Unsupported feature."; }
InputAndOutput::InputAndOutput(MSTensor &&tensor) { MS_LOG(ERROR) << "Unsupported feature."; }

InputAndOutput::InputAndOutput(const std::shared_ptr<CellBase> &cell, const std::vector<InputAndOutput> &prev,
                               int32_t index) {
  MS_LOG(ERROR) << "Unsupported feature.";
}
}  // namespace mindspore
