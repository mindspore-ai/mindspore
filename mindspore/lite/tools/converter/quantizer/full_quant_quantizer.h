/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_FULL_QUANT_QUANTIZER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_FULL_QUANT_QUANTIZER_H

#include <string>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cfloat>
#include <map>
#include "ops/primitive_c.h"
#include "schema/inner/model_generated.h"
#include "src/lite_session.h"
#include "tools/converter/quantizer/quantizer.h"
#include "tools/converter/converter.h"
#include "include/ms_tensor.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/converter/quantizer/quant_params.h"
#include "tools/converter/preprocess/preprocess_param.h"

namespace mindspore::lite::quant {
class Calibrator;
constexpr int kDefaultBinNumber = 2048;
struct DivergInfo {
  std::vector<float> histogram;
  CNodePtr cnode;
  int bin_num = 0;
  float interval = 0;
  float max = 0.0f;
  float min = 0.0f;
  float best_T = 0.0f;
  size_t bit_num = 0;
  int quant_max = 255;
  int quant_min = 0;
  ActivationQuantizedMethod activation_quant_method = MAX_MIN;
  std::vector<float> min_datas;
  std::vector<float> max_datas;
  std::pair<float, float> percent_result{0.0, 0.0};
  float scale_tmp = 0;
  DivergInfo() = default;
  DivergInfo(CNodePtr cnode, int bins, size_t bits, int quant_max, int quant_min,
             ActivationQuantizedMethod activation_quant_method) {
    this->activation_quant_method = activation_quant_method;
    this->cnode = std::move(cnode);
    this->bin_num = bins;
    this->bit_num = bits;
    histogram.resize(bin_num);
    max = -FLT_MAX;
    min = FLT_MAX;
    this->quant_max = quant_max;
    this->quant_min = quant_min;
    std::fill(histogram.begin(), histogram.end(), 1.0e-7);
  }

  STATUS RecordMaxMinValue(const std::vector<float> &data);

  STATUS RecordMaxMinValueArray(const std::vector<float> &data);

  void UpdateInterval();

  STATUS UpdateHistogram(const std::vector<float> &data);

  void DumpHistogram();

  void HandleBinForKL(int quant_bint_nums, int bin_index, std::vector<float> *quantized_histogram,
                      std::vector<float> *expanded_histogram);

  STATUS ComputeThreshold();

  std::pair<CNodePtr, float> GetScale();

  std::pair<CNodePtr, int32_t> GetZeropoint();
};

class FullQuantQuantizer : public Quantizer {
 public:
  FullQuantQuantizer(FuncGraphPtr graph, int bit_num, TypeId target_type = kNumberTypeInt8, bool per_channel = true);
  ~FullQuantQuantizer() override;

  STATUS DoQuantize(FuncGraphPtr func_graph) override;

  size_t bit_num;
  int quant_max{INT8_MAX};
  int quant_min{INT8_MIN};

 private:
  bool per_channel_{true};
  TypeId target_type_{kNumberTypeInt8};
  std::unique_ptr<Calibrator> calibrator_{nullptr};

  session::LiteSession *fp32_session_{nullptr};
  Model *fp32_model_{nullptr};
  session::LiteSession *int8_session_{nullptr};
  Model *int8_model_{nullptr};

  std::map<std::string, std::vector<float>> fp32_op_input_map;           // concurrency
  std::map<std::string, std::vector<float>> fp32_op_output_ch_mean_map;  // concurrency
  std::map<std::string, std::vector<float>> op_bias_diff_map;            // only use by int8 model
  std::mutex mutex_op_input;
  std::mutex mutex_op_output;

  enum OperationType {
    STORE,
    FETCH,
  };

  bool OpInputDataHandle(OperationType type, const string &op_name, std::vector<float> *data);
  bool OpOutputChMeanDataHandle(OperationType type, const string &op_name, std::vector<float> *data);

  const std::string kTypeConv2D = schema::EnumNamePrimitiveType(schema::PrimitiveType_Conv2DFusion);
  const std::string kTypeDepthwiseConv2D = schema::EnumNamePrimitiveType(schema::PrimitiveType_Conv2DFusion);
  const std::string kTypeConcat = schema::EnumNamePrimitiveType(schema::PrimitiveType_Concat);

  STATUS PreProcess();

  static STATUS CheckFp32TensorVec(const std::string &node_name,
                                   const std::vector<mindspore::tensor::MSTensor *> &tensor_vec);

  STATUS DoInference();

  STATUS UpdateDivergeInterval();

  STATUS CollectDataFrequency();

  STATUS ComputeThreshold();

  STATUS QuantNodeSimpleOp(const CNodePtr &cnode);

  STATUS QuantNode();

  STATUS SetInOutQuantParam(const AnfNodePtr &input_node, const std::unique_ptr<DivergInfo> &info,
                            const PrimitivePtr &primitive, bool is_input, size_t index) const;

  STATUS DoWeightQuant(const std::string &op_name, const AnfNodePtr &weight, const PrimitivePtr &primitive,
                       bool per_channel, int input_index) const;

  STATUS DoParameterNodeQuant(const CNodePtr &cnode, const AnfNodePtr &input_node, size_t input_index);

  static STATUS DoBiasQuant(const AnfNodePtr &bias, const PrimitivePtr &primitive);
  STATUS Int8Inference();
  STATUS BiasCorrection(const FuncGraphPtr &func_graph);
  STATUS BiasCorrection(const FuncGraphPtr &func_graph, const CNodePtr &cnode);
  KernelCallBack GetBeforeCallBack(bool int8_op);
  KernelCallBack GetAfterCallBack(bool int8_op);
  KernelCallBack GetInt8AfterCallBack();
  KernelCallBack GetFloatAfterCallBack();
};

class Calibrator {
 public:
  explicit Calibrator(size_t bit_num, int quant_max, int quant_min)
      : bit_num_(bit_num), quant_max_(quant_max), quant_min_(quant_min) {}

  ~Calibrator() = default;

  STATUS GenerateInputData(const std::string &input_name, size_t image_index,
                           mindspore::tensor::MSTensor *tensor) const;

  size_t GetBatchNum() const { return data_pre_process_param_.calibrate_size; }

  uint32_t GetThreadNum() const { return full_quant_param_.thread_num; }

  bool GetBiasCorrection() const { return full_quant_param_.bias_correction; }

  size_t GetInputNum() const { return data_pre_process_param_.calibrate_path_vector.size(); }

  STATUS AddQuantizedOp(const CNodePtr &cnode);

  STATUS RecordMaxMinValue(const std::vector<float> &data, const std::unique_ptr<DivergInfo> &diverg_info);

  STATUS UpdateDivergInterval(std::unordered_map<std::string, std::vector<std::unique_ptr<DivergInfo>>> *diverg_info);

  STATUS UpdateDataFrequency(const std::vector<float> &data, const std::unique_ptr<DivergInfo> &diverg_info);

  STATUS ComputeThreshold();

  std::unordered_map<std::string, std::vector<std::unique_ptr<DivergInfo>>> *GetInputDivergInfo();

  std::unordered_map<std::string, std::vector<std::unique_ptr<DivergInfo>>> *GetOutputDivergInfo();

  FullQuantParam full_quant_param_;

  preprocess::DataPreProcessParam data_pre_process_param_;

 private:
  std::unordered_map<std::string, std::vector<std::unique_ptr<DivergInfo>>> inputs_diverg_info_;

  std::unordered_map<std::string, std::vector<std::unique_ptr<DivergInfo>>> outputs_diverg_info_;

  size_t bit_num_;
  int quant_max_;
  int quant_min_;
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_FULL_QUANT_QUANTIZER_H
