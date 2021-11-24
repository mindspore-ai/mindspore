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
#include "tools/converter/quantizer/calibrator.h"
#include "tools/converter/quantizer/data_distribution.h"

namespace mindspore::lite::quant {
enum OperationType {
  STORE,
  FETCH,
};
class FullQuantQuantizer : public Quantizer {
 public:
  FullQuantQuantizer(FuncGraphPtr graph, int bit_num, TypeId target_type = kNumberTypeInt8);
  ~FullQuantQuantizer() override;

  int DoQuantize(FuncGraphPtr func_graph) override;

 private:
  bool OpInputDataHandle(OperationType type, const string &op_name, std::vector<float> *data);
  bool OpOutputChMeanDataHandle(OperationType type, const string &op_name, std::vector<float> *data);

  int PreProcess();

  int CheckFp32TensorVec(const std::string &node_name, const std::vector<mindspore::tensor::MSTensor *> &tensor_vec);

  int DoInference(CollectType collect_type);

  int UpdateDivergeInterval();

  int ComputeThreshold();

  int QuantNodeSimpleOp(const CNodePtr &cnode);

  int QuantNode();

  int SetInOutQuantParam(const AnfNodePtr &input_node, const std::unique_ptr<DataDistribution> &info,
                         const PrimitivePtr &primitive, bool is_input, size_t index) const;

  int DoWeightQuant(const std::string &op_name, const AnfNodePtr &weight, const PrimitivePtr &primitive,
                    bool per_channel, int input_index) const;

  int DoParameterNodeQuant(const CNodePtr &cnode, const AnfNodePtr &input_node, size_t input_index);

  int DoBiasQuant(const AnfNodePtr &bias, const PrimitivePtr &primitive);
  int Int8Inference();
  int BiasCorrection(const FuncGraphPtr &func_graph);
  int BiasCorrection(const FuncGraphPtr &func_graph, const CNodePtr &cnode);
  KernelCallBack GetBeforeCallBack(bool int8_op);
  KernelCallBack GetAfterCallBack(bool int8_op);
  KernelCallBack GetInt8AfterCallBack();
  KernelCallBack GetFloatAfterCallBack();

 private:
  TypeId target_data_type_{kNumberTypeInt8};
  std::unique_ptr<Calibrator> calibrator_{nullptr};
  session::LiteSession *fp32_session_{nullptr};
  Model *fp32_model_{nullptr};
  session::LiteSession *int8_session_{nullptr};
  Model *int8_model_{nullptr};

  std::map<std::string, std::vector<float>> fp32_op_input_map_;           // concurrency
  std::map<std::string, std::vector<float>> fp32_op_output_ch_mean_map_;  // concurrency
  std::map<std::string, std::vector<float>> op_bias_diff_map_;            // only use by int8 model
  std::mutex mutex_op_input_;
  std::mutex mutex_op_output_;

  size_t bit_num_;
  int q_max_{INT8_MAX};
  int q_min_{INT8_MIN};
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_FULL_QUANT_QUANTIZER_H
