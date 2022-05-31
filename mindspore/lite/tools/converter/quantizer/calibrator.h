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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_CALIBRATOR_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_CALIBRATOR_H_
#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <memory>
#include "tools/converter/quantizer/quant_params.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/converter/quantizer/data_distribution.h"

namespace mindspore::lite::quant {
enum CollectType {
  MIN_MAX,
  KL_BIN,
};
class Calibrator {
 public:
  Calibrator(size_t bit_num, int quant_max, int quant_min, ActivationQuantizedMethod activation_quant_method,
             const preprocess::DataPreProcessParam &data_pre_process_param, bool symmetric)
      : bit_num_(bit_num),
        quant_max_(quant_max),
        quant_min_(quant_min),
        symmetric_(symmetric),
        activation_quant_method_(activation_quant_method),
        data_pre_process_param_(data_pre_process_param) {}

  ~Calibrator() = default;

  int GenerateInputData(const std::string &input_name, size_t image_index, mindspore::MSTensor *tensor) const;

  int AddQuantizedOp(const CNodePtr &cnode);

  int RecordMaxMinValue(const std::vector<float> &data, const std::unique_ptr<DataDistribution> &diverg_info);

  int UpdateDivergInterval();

  int UpdateDataFrequency(const std::vector<float> &data, const std::unique_ptr<DataDistribution> &diverg_info);

  int ComputeThreshold();

  size_t GetBatchNum() const { return data_pre_process_param_.calibrate_size; }

  size_t GetInputNum() const { return data_pre_process_param_.calibrate_path_vector.size(); }

  std::unordered_map<std::string, std::map<int, std::unique_ptr<DataDistribution>>> *GetInputDivergInfo() {
    return &this->inputs_diverg_info_;
  }

  std::unordered_map<std::string, std::map<int, std::unique_ptr<DataDistribution>>> *GetOutputDivergInfo() {
    return &this->outputs_diverg_info_;
  }

  int CollectDataDistribution(
    const std::string &node_name, const std::vector<mindspore::MSTensor> &tensors,
    std::unordered_map<std::string, std::map<int, std::unique_ptr<DataDistribution>>> *diverg_info_map,
    CollectType collect_type);

 private:
  // {node_name,{tensor_index,DataDistribution}}
  std::unordered_map<std::string, std::map<int, std::unique_ptr<DataDistribution>>> inputs_diverg_info_;
  // {node_name,{tensor_index,DataDistribution}}
  std::unordered_map<std::string, std::map<int, std::unique_ptr<DataDistribution>>> outputs_diverg_info_;
  size_t bit_num_;
  int quant_max_;
  int quant_min_;
  bool symmetric_;
  ActivationQuantizedMethod activation_quant_method_;
  preprocess::DataPreProcessParam data_pre_process_param_;
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER__CALIBRATOR_H
