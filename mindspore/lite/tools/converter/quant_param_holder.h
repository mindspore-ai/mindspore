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

#include <vector>
#include <memory>
#include "ir/anf.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
using QuantParamsVector = std::vector<std::vector<schema::QuantParamT>>;
class QuantParamHolder : public Value {
 public:
  QuantParamHolder() = default;

  ~QuantParamHolder() override = default;

  MS_DECLARE_PARENT(QuantParamHolder, Value);

  bool operator==(const Value &rhs) const override {  // unused
    if (rhs.isa<QuantParamHolder>()) {
      auto other_holder = dynamic_cast<const QuantParamHolder &>(rhs);
      auto input_quant_params_rhs = other_holder.input_quant_params();
      auto output_quant_params_rhs = other_holder.output_quant_params();
      if (input_quant_params_rhs.size() != this->input_quant_param_.size() ||
          output_quant_params_rhs.size() != this->output_quant_param_.size()) {
        return false;
      }
      for (size_t i = 0; i < input_quant_params_rhs.size(); ++i) {
        if (input_quant_params_rhs.at(i).size() != this->input_quant_param_.at(i).size()) {
          return false;
        }
        auto *params = reinterpret_cast<const char *>(this->input_quant_param_.at(i).data());
        auto *params_rhs = reinterpret_cast<const char *>(input_quant_params_rhs.at(i).data());
        for (size_t j = 0; j < input_quant_params_rhs.at(i).size() * sizeof(schema::QuantParamT); ++j) {
          if (params[j] != params_rhs[j]) {
            return false;
          }
        }
      }
      for (size_t i = 0; i < output_quant_params_rhs.size(); ++i) {
        if (output_quant_params_rhs.at(i).size() != this->output_quant_param_.at(i).size()) {
          return false;
        }
        auto *params = reinterpret_cast<const char *>(this->output_quant_param_.at(i).data());
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

  void set_input_quant_params(const QuantParamsVector &input_quant_param) {
    this->input_quant_param_ = input_quant_param;
  }

  void set_input_quant_param(const size_t &index, const std::vector<schema::QuantParamT> &input_quant_param) {
    if (index >= this->input_quant_param_.size()) {
      std::vector<schema::QuantParamT> place_quant(1);
      this->input_quant_param_.insert(this->input_quant_param_.end(), index + 1 - input_quant_param_.size(),
                                      place_quant);
    }
    this->input_quant_param_.at(index) = input_quant_param;
  }

  void set_output_quant_params(const std::vector<std::vector<schema::QuantParamT>> &output_quant_param) {
    this->output_quant_param_ = output_quant_param;
  }

  void set_output_quant_param(const size_t &index, const std::vector<schema::QuantParamT> &output_quant_param) {
    if (index >= this->output_quant_param_.size()) {
      std::vector<schema::QuantParamT> place_quant(1);
      this->output_quant_param_.insert(this->output_quant_param_.end(), index + 1 - output_quant_param_.size(),
                                       place_quant);
    }
    this->output_quant_param_.at(index) = output_quant_param;
  }

  void set_enable_huffman_code(bool enable_huffman_code) { enable_huffman_code_ = enable_huffman_code; }

  bool enable_huffman_code() const { return enable_huffman_code_; }

  void AddInputQuantParam(const std::vector<schema::QuantParamT> &quant_param) {
    this->input_quant_param_.emplace_back(quant_param);
  }

  std::vector<std::vector<schema::QuantParamT>> input_quant_params() const { return this->input_quant_param_; }

  void AddOutputQuantParam(const std::vector<schema::QuantParamT> &quant_param) {
    this->output_quant_param_.emplace_back(quant_param);
  }

  std::vector<std::vector<schema::QuantParamT>> output_quant_params() const { return this->output_quant_param_; }

  void ClearInputOutputQuantParam() {
    input_quant_param_.clear();
    output_quant_param_.clear();
  }

  bool IsInputQuantParamsInited() {
    if (this->input_quant_param_.empty()) {
      return false;
    }
    for (auto &quant_param : this->input_quant_param_) {
      if (!quant_param.front().inited) {
        return false;
      }
    }
    return true;
  }

  bool IsOutputQuantParamsInited() {
    if (this->output_quant_param_.empty()) {
      return false;
    }
    for (auto &quant_param : this->output_quant_param_) {
      if (!quant_param.front().inited) {
        return false;
      }
    }
    return true;
  }

 private:
  schema::QuantType quant_type_{schema::QuantType_QUANT_NONE};
  QuantParamsVector input_quant_param_;
  QuantParamsVector output_quant_param_;
  bool enable_huffman_code_ = false;
};
using QuantParamHolderPtr = std::shared_ptr<QuantParamHolder>;
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANT_PARAM_CONTEXT_H
