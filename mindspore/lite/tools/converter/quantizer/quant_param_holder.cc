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

#include "tools/converter/quantizer/quant_param_holder.h"
#include <utility>
#include <vector>
#include <memory>
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
void QuantParamHolder::set_input_quant_param(const size_t &index,
                                             const std::vector<schema::QuantParamT> &input_quant_param) {
  if (index >= this->input_quant_params_.size()) {
    std::vector<schema::QuantParamT> place_quant(1);
    this->input_quant_params_.insert(this->input_quant_params_.end(), index + 1 - input_quant_params_.size(),
                                     place_quant);
  }

  this->input_quant_params_.at(index) = input_quant_param;
}

void QuantParamHolder::set_output_quant_param(const size_t &index,
                                              const std::vector<schema::QuantParamT> &output_quant_param) {
  if (index >= this->output_quant_params_.size()) {
    std::vector<schema::QuantParamT> place_quant(1);
    this->output_quant_params_.insert(this->output_quant_params_.end(), index + 1 - output_quant_params_.size(),
                                      place_quant);
  }
  this->output_quant_params_.at(index) = output_quant_param;
}

bool QuantParamHolder::IsInputQuantParamsInited() {
  if (this->input_quant_params_.empty()) {
    return false;
  }
  bool is_quant_params_inited =
    std::all_of(this->input_quant_params_.begin(), this->input_quant_params_.end(),
                [](const std::vector<schema::QuantParamT> &quant_params) { return quant_params.front().inited; });
  return is_quant_params_inited;
}

bool QuantParamHolder::IsOutputQuantParamsInited() {
  if (this->output_quant_params_.empty()) {
    return false;
  }
  bool is_quant_params_inited =
    std::all_of(this->output_quant_params_.begin(), this->output_quant_params_.end(),
                [](const std::vector<schema::QuantParamT> &quant_params) { return quant_params.front().inited; });
  return is_quant_params_inited;
}

bool QuantParamHolder::IsInputExistInited() {
  if (this->input_quant_params_.empty()) {
    return false;
  }
  bool is_exist_param_inited =
    std::any_of(this->input_quant_params_.begin(), this->input_quant_params_.end(),
                [](const std::vector<schema::QuantParamT> &quant_params) { return quant_params.front().inited; });
  return is_exist_param_inited;
}

bool QuantParamHolder::IsOutputExistInited() {
  if (this->output_quant_params_.empty()) {
    return false;
  }
  bool is_exist_param_inited =
    std::any_of(this->output_quant_params_.begin(), this->output_quant_params_.end(),
                [](const std::vector<schema::QuantParamT> &quant_params) { return quant_params.front().inited; });
  return is_exist_param_inited;
}

void QuantParamHolder::ClearQuantParams() {
  quant_type_ = schema::QuantType_QUANT_NONE;
  input_quant_params_.clear();
  output_quant_params_.clear();
}

bool QuantParamHolder::CheckInit(size_t index, bool is_input) {
  std::vector<schema::QuantParamT> param;
  if (is_input) {
    if (input_quant_params_.size() <= index) {
      return false;
    }
    param = input_quant_params_.at(index);
  } else {
    if (output_quant_params_.size() <= index) {
      return false;
    }
    param = output_quant_params_.at(index);
  }
  return (!param.empty() && param.front().inited);
}

void QuantParamHolder::SetQuantClusters(size_t index, const std::vector<float> &quant_cluster) {
  quant_clusters.insert({index, quant_cluster});
}

std::vector<float> QuantParamHolder::GetQuantClusters(size_t index) {
  auto iter = quant_clusters.find(index);
  if (iter == quant_clusters.end()) {
    return {};
  } else {
    return iter->second;
  }
}
}  // namespace lite
}  // namespace mindspore
