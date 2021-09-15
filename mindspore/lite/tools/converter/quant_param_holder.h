/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANT_PARAM_CONTEXT_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANT_PARAM_CONTEXT_H

#include <utility>
#include <vector>
#include <memory>
#include "ir/anf.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
using QuantParamsVector = std::vector<std::vector<schema::QuantParamT>>;
class QuantParamHolder : public Value {
 public:
  QuantParamHolder(size_t input_size, size_t output_size) {
    input_quant_params_.resize(input_size);
    output_quant_params_.resize(output_size);
    for (size_t i = 0; i < input_size; i++) {
      std::vector<schema::QuantParamT> notinited_quant_params(1);
      set_input_quant_param(i, notinited_quant_params);
    }

    for (size_t i = 0; i < output_size; i++) {
      std::vector<schema::QuantParamT> notinited_quant_params(1);
      set_output_quant_param(i, notinited_quant_params);
    }
  }

  QuantParamHolder(const QuantParamsVector &input_quant_params, const QuantParamsVector &output_quant_params) {
    input_quant_params_ = input_quant_params;
    output_quant_params_ = output_quant_params;
  }

  ~QuantParamHolder() override = default;

  MS_DECLARE_PARENT(QuantParamHolder, Value);

  bool operator==(const Value &rhs) const override {  // unused
    if (rhs.isa<QuantParamHolder>()) {
      auto other_holder = dynamic_cast<const QuantParamHolder &>(rhs);
      auto input_quant_params_rhs = other_holder.get_input_quant_params();
      auto output_quant_params_rhs = other_holder.get_output_quant_params();
      if (input_quant_params_rhs.size() != this->input_quant_params_.size() ||
          output_quant_params_rhs.size() != this->output_quant_params_.size()) {
        return false;
      }
      for (size_t i = 0; i < input_quant_params_rhs.size(); ++i) {
        if (input_quant_params_rhs.at(i).size() != this->input_quant_params_.at(i).size()) {
          return false;
        }
        auto *params = reinterpret_cast<const char *>(this->input_quant_params_.at(i).data());
        auto *params_rhs = reinterpret_cast<const char *>(input_quant_params_rhs.at(i).data());
        for (size_t j = 0; j < input_quant_params_rhs.at(i).size() * sizeof(schema::QuantParamT); ++j) {
          if (params[j] != params_rhs[j]) {
            return false;
          }
        }
      }
      for (size_t i = 0; i < output_quant_params_rhs.size(); ++i) {
        if (output_quant_params_rhs.at(i).size() != this->output_quant_params_.at(i).size()) {
          return false;
        }
        auto *params = reinterpret_cast<const char *>(this->output_quant_params_.at(i).data());
        auto *params_rhs = reinterpret_cast<const char *>(output_quant_params_rhs.at(i).data());
        for (size_t j = 0; j < output_quant_params_rhs.at(i).size() * sizeof(schema::QuantParamT); ++j) {
          if (params[j] != params_rhs[j]) {
            return false;
          }
        }
      }
    } else {
      return false;
    }
    return true;
  }

  void set_quant_type(const schema::QuantType &quant_type) { quant_type_ = quant_type; }

  schema::QuantType quant_type() const { return quant_type_; }

  void set_input_quant_param(const size_t &index, const std::vector<schema::QuantParamT> &input_quant_param) {
    if (index >= this->input_quant_params_.size()) {
      std::vector<schema::QuantParamT> place_quant(1);
      this->input_quant_params_.insert(this->input_quant_params_.end(), index + 1 - input_quant_params_.size(),
                                       place_quant);
    }
    this->input_quant_params_.at(index) = input_quant_param;
  }

  void set_output_quant_param(const size_t &index, const std::vector<schema::QuantParamT> &output_quant_param) {
    if (index >= this->output_quant_params_.size()) {
      std::vector<schema::QuantParamT> place_quant(1);
      this->output_quant_params_.insert(this->output_quant_params_.end(), index + 1 - output_quant_params_.size(),
                                        place_quant);
    }
    this->output_quant_params_.at(index) = output_quant_param;
  }

  void set_enable_huffman_code(bool enable_huffman_code) { enable_huffman_code_ = enable_huffman_code; }

  bool enable_huffman_code() const { return enable_huffman_code_; }

  std::vector<std::vector<schema::QuantParamT>> get_input_quant_params() const { return this->input_quant_params_; }

  std::vector<std::vector<schema::QuantParamT>> get_output_quant_params() const { return this->output_quant_params_; }

  // deprecated
  void ClearInputOutputQuantParam() {
    input_quant_params_.clear();
    output_quant_params_.clear();
  }

  bool IsInputQuantParamsInited() {
    if (this->input_quant_params_.empty()) {
      return false;
    }
    for (auto &quant_param : this->input_quant_params_) {
      if (!quant_param.front().inited) {
        return false;
      }
    }
    return true;
  }

  bool IsOutputQuantParamsInited() {
    if (this->output_quant_params_.empty()) {
      return false;
    }
    for (auto &quant_param : this->output_quant_params_) {
      if (!quant_param.front().inited) {
        return false;
      }
    }
    return true;
  }

 private:
  schema::QuantType quant_type_{schema::QuantType_QUANT_NONE};
  QuantParamsVector input_quant_params_;
  QuantParamsVector output_quant_params_;
  bool enable_huffman_code_ = false;
};
using QuantParamHolderPtr = std::shared_ptr<QuantParamHolder>;
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANT_PARAM_CONTEXT_H
