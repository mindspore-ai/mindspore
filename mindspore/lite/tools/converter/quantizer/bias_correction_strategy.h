/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_BIAS_CORRECTION_STRATEGY_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_BIAS_CORRECTION_STRATEGY_H_

#include <memory>
#include <map>
#include <string>
#include <vector>
#include "base/base.h"
#include "ir/anf.h"
#include "tools/converter/quantizer/calibrator.h"
#include "tools/converter/quantizer/quant_strategy.h"

namespace mindspore::lite::quant {
enum OperationType {
  STORE,
  FETCH,
};
enum CallBackType {
  CPUFP32,
  CPUInt8,
  NVGPUInt8,
};

class BiasCorrectionStrategy {
 public:
  BiasCorrectionStrategy(const std::shared_ptr<ConverterPara> &param, const std::shared_ptr<Calibrator> &calibrator,
                         const std::shared_ptr<QuantStrategy> &quant_strategy,
                         const std::shared_ptr<mindspore::Model> &fp32_ms_model, int activation_q_min,
                         int activation_q_max)
      : param_(param),
        calibrator_(calibrator),
        quant_strategy_(quant_strategy),
        fp32_ms_model_(fp32_ms_model),
        activation_q_min_(activation_q_min),
        activation_q_max_(activation_q_max) {}

  ~BiasCorrectionStrategy() = default;

  int DoBiasCorrection(const FuncGraphPtr &quant_func_graph);

 private:
  int DoCPUBiasCorrection(const FuncGraphPtr &quant_func_graph);

  int DoNVGPUBiasCorrection(const FuncGraphPtr &quant_func_graph);

 private:
  int DoBiasCorrection(const FuncGraphPtr &quant_func_graph, bool int32_bias);

  int DoCNodeBiasCorrection(const FuncGraphPtr &quant_func_graph, const CNodePtr &cnode, bool int32_bias);

  int Int8Inference(const MSKernelCallBack &before_call_back, const MSKernelCallBack &after_call_back,
                    const FuncGraphPtr &quant_func_graph);

  int Fp32Inference(const MSKernelCallBack &before_call_back, const MSKernelCallBack &after_call_back);

  bool OpInputDataHandle(OperationType type, const string &op_name, std::vector<float> *data);

  bool OpOutputChMeanDataHandle(OperationType type, const string &op_name, std::vector<float> *data);

  void CalcAccumulativeError(const MSCallBackParam &call_param, const std::vector<float> &fp32_op_output_ch_mean,
                             const std::vector<float> &dequant_op_output_ch_mean);

  MSKernelCallBack GetBeforeCallBack(CallBackType call_back_flag);

  MSKernelCallBack GetCPUFloatBeforeCallBack();

  MSKernelCallBack GetCPUInt8BeforeCallBack();

  MSKernelCallBack GetNVGPUInt8BeforeCallBack();

  MSKernelCallBack GetAfterCallBack(CallBackType call_back_flag);

  MSKernelCallBack GetCPUInt8AfterCallBack();

  MSKernelCallBack GetCPUFloatAfterCallBack();

  MSKernelCallBack GetNVGPUInt8AfterCallBack();

  int QuantOriginFeatureMap(const float *origin_feature_map_data, size_t origin_feature_map_data_size,
                            const std::vector<mindspore::QuantParam> &feature_map_quant_params, size_t quant_size,
                            std::vector<int8_t> *quant_datas);

  int CreateFp32BiasTensor(const FuncGraphPtr &quant_func_graph, const CNodePtr &cnode, const ParameterPtr &parameter,
                           const std::vector<float> &bias_diff);

  int AddBiasToInt32Tensor(const CNodePtr &cnode, const tensor::TensorPtr &bias_tensor,
                           const std::vector<schema::QuantParamT> &bias_quant_params,
                           const std::vector<float> &bias_diff);

  int AddBiasToFp32Tensor(const CNodePtr &cnode, const tensor::TensorPtr &bias_tensor,
                          const std::vector<float> &bias_diff);

  template <typename T>
  int CalculatePerChannelMeans(const T *tensor_data, size_t elem_count, std::vector<int64_t> shapes,
                               std::vector<float> *per_channel_mean) const {
    CHECK_NULL_RETURN(tensor_data);
    MS_CHECK_TRUE_RET(elem_count != 0, RET_PARAM_INVALID);
    // suppose the activation format: NHWC
    MS_CHECK_TRUE_RET(!shapes.empty(), RET_ERROR);
    auto channels = shapes[shapes.size() - 1];
    MS_CHECK_GT(channels, 0, RET_ERROR);
    per_channel_mean->resize(channels);
    auto bucket_size = elem_count / channels;
    for (int i = 0; i < channels; i++) {
      float sum = 0;
      for (size_t j = 0; j < bucket_size; j++) {
        auto index = j * channels + i;
        if (index >= elem_count) {
          MS_LOG(ERROR) << "over flow!";
          return RET_ERROR;
        }
        sum += tensor_data[index];
      }
      MS_CHECK_TRUE_RET(bucket_size != 0, RET_PARAM_INVALID);
      sum = sum / bucket_size;
      MS_CHECK_GT(static_cast<int>(per_channel_mean->size()), i, RET_ERROR);
      per_channel_mean->at(i) = sum;
    }
    return RET_OK;
  }

 private:
  const std::shared_ptr<ConverterPara> param_;
  std::shared_ptr<Calibrator> calibrator_{nullptr};
  std::shared_ptr<QuantStrategy> quant_strategy_{nullptr};
  std::shared_ptr<mindspore::Model> fp32_ms_model_{nullptr};
  int activation_q_min_{INT8_MIN};
  int activation_q_max_{INT8_MAX};

  MSKernelCallBack int8_before_call_back_;
  MSKernelCallBack int8_after_call_back_;
  MSKernelCallBack fp32_before_call_back_;
  MSKernelCallBack fp32_after_call_back_;

  std::map<std::string, std::vector<float>> fp32_op_input_map_;           // concurrency
  std::map<std::string, std::vector<float>> fp32_op_output_ch_mean_map_;  // concurrency
  std::map<std::string, std::vector<float>> op_bias_diff_sum_map_;        // Record the sum of diffs in tensor
  std::mutex mutex_op_input_;
  std::mutex mutex_op_output_;
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_BIAS_CORRECTION_STRATEGY_H_
