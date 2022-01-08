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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_BIASCORRECTION_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_BIASCORRECTION_H
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
  BiasCorrectionStrategy(const converter::Flags &flags, const std::shared_ptr<Calibrator> &calibrator,
                         const std::shared_ptr<QuantStrategy> &quant_strategy, session::LiteSession *fp32_session,
                         Model *fp32_model, int activation_q_min, int activation_q_max)
      : flags_(flags),
        calibrator_(calibrator),
        quant_strategy_(quant_strategy),
        fp32_session_(fp32_session),
        fp32_model_(fp32_model),
        activation_q_min_(activation_q_min),
        activation_q_max_(activation_q_max) {}
  ~BiasCorrectionStrategy() {
    if (int8_session_ != nullptr) {
      delete int8_session_;
    }
    if (int8_model_ != nullptr) {
      delete int8_model_;
    }
  }
  int DoCPUBiasCorrection(const FuncGraphPtr &quant_func_graph);

  int DoNVGPUBiasCorrection(const FuncGraphPtr &quant_func_graph);

 private:
  int CreateQuantModel(const FuncGraphPtr &quant_func_graph);
  int DoBiasCorrection(const FuncGraphPtr &quant_func_graph, bool int32_bias);
  int DoCNodeBiasCorrection(const FuncGraphPtr &quant_func_graph, const CNodePtr &cnode, bool int32_bias);
  int Int8Inference(const KernelCallBack &before_call_back, const KernelCallBack &after_call_back);
  int Fp32Inference(const KernelCallBack &before_call_back, const KernelCallBack &after_call_back);
  bool OpInputDataHandle(OperationType type, const string &op_name, std::vector<float> *data);
  bool OpOutputChMeanDataHandle(OperationType type, const string &op_name, std::vector<float> *data);
  void CalcAccumulativeError(const CallBackParam &call_param, const std::vector<float> &fp32_op_output_ch_mean,
                             const std::vector<float> &dequant_op_output_ch_mean);
  KernelCallBack GetBeforeCallBack(CallBackType call_back_flag);
  KernelCallBack GetCPUFloatBeforeCallBack();
  KernelCallBack GetCPUInt8BeforeCallBack();
  KernelCallBack GetNVGPUInt8BeforeCallBack();
  KernelCallBack GetAfterCallBack(CallBackType call_back_flag);
  KernelCallBack GetCPUInt8AfterCallBack();
  KernelCallBack GetCPUFloatAfterCallBack();
  KernelCallBack GetNVGPUInt8AfterCallBack();

  int QuantOriginFeatureMap(const float *origin_feature_map_data, size_t origin_feature_map_data_size,
                            const std::vector<lite::LiteQuantParam> &feature_map_quant_params, size_t quant_size,
                            std::vector<int8_t> *quant_datas);

  int CreateFp32BiasTensor(const FuncGraphPtr &quant_func_graph, const CNodePtr &cnode, const ParameterPtr &parameter,
                           const std::vector<float> &bias_diff);

  int AddBiasToInt32Tensor(const CNodePtr &cnode, const tensor::TensorPtr &bias_tensor,
                           const std::vector<schema::QuantParamT> &bias_quant_params,
                           const std::vector<float> &bias_diff);

  int AddBiasToFp32Tensor(const CNodePtr &cnode, const tensor::TensorPtr &bias_tensor,
                          const std::vector<float> &bias_diff);
  template <typename T>
  int CalculatePerChannelMeans(const T *tensor_data, size_t elem_count, std::vector<int> shapes,
                               std::vector<float> *per_channel_mean) {
    CHECK_NULL_RETURN(tensor_data);
    MS_CHECK_GT(elem_count, 0, RET_ERROR);
    // suppose the activation format: NHWC
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
      MS_CHECK_GT(bucket_size, 0, RET_ERROR);
      sum = sum / bucket_size;
      per_channel_mean->at(i) = sum;
    }
    return RET_OK;
  }

 private:
  converter::Flags flags_;
  std::shared_ptr<Calibrator> calibrator_{nullptr};
  std::shared_ptr<QuantStrategy> quant_strategy_{nullptr};
  session::LiteSession *fp32_session_{nullptr};
  Model *fp32_model_{nullptr};
  int activation_q_min_{INT8_MIN};
  int activation_q_max_{INT8_MAX};

  session::LiteSession *int8_session_{nullptr};
  Model *int8_model_{nullptr};

  KernelCallBack int8_before_call_back_;
  KernelCallBack int8_after_call_back_;
  KernelCallBack fp32_before_call_back_;
  KernelCallBack fp32_after_call_back_;

  std::map<std::string, std::vector<float>> fp32_op_input_map_;           // concurrency
  std::map<std::string, std::vector<float>> fp32_op_output_ch_mean_map_;  // concurrency
  std::map<std::string, std::vector<float>> op_bias_diff_sum_map_;        // Record the sum of diffs in tensor
  std::mutex mutex_op_input_;
  std::mutex mutex_op_output_;
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_BIASCORRECTION_H
