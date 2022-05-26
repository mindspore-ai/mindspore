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
#define USE_DEPRECATED_API

#include "tools/converter/quantizer/parameter_tunner.h"
#include <set>
#include <functional>
#include <map>
#include <memory>
#include <vector>
#include <algorithm>
#include "tools/converter/preprocess/image_preprocess.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/converter/quantizer/weight_quantizer.h"
#include "tools/converter/export_model.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/parser/parser_utils.h"

namespace mindspore::lite::quant {
MinMax ParameterOptimizer::GetFineTuneRange(std::vector<float> *candidate_scales) {
  const int top_3 = 3;
  if (candidate_scales == nullptr || candidate_scales->empty()) {
    MS_LOG(ERROR) << "candidate_scales is nullptr.";
    return {0, 0};
  }
  MinMax min_max;
  std::sort(candidate_scales->begin(), candidate_scales->end(), std::greater<>());
  min_max.max = candidate_scales->front();
  min_max.min = candidate_scales->size() >= top_3 ? candidate_scales->at(top_3 - 1)
                                                  : candidate_scales->at(candidate_scales->size() - 1);
  return min_max;
}

int ParameterOptimizer::CloneFuncGraph(const FuncGraphPtr &func_graph, const std::shared_ptr<ConverterPara> &param,
                                       FuncGraphPtr *func_graph_bak) {
  CHECK_NULL_RETURN(func_graph_bak);
  CHECK_NULL_RETURN(param);
  std::map<FuncGraphPtr, FuncGraphPtr> cloned_func_graph;
  *func_graph_bak = lite::CloneFuncGraph(func_graph, param, &cloned_func_graph);
  CHECK_NULL_RETURN(*func_graph_bak);
  static auto root_func_manager = Manage(*func_graph_bak);
  std::set<FuncGraphPtr> all_func_graphs = {};
  lite::GetAllFuncGraph(*func_graph_bak, &all_func_graphs);
  for (const auto &graph : all_func_graphs) {
    graph->set_manager(root_func_manager);
  }
  return RET_OK;
}

int ParameterOptimizer::WeightQuantModelInference(const FuncGraphPtr &func_graph,
                                                  const std::shared_ptr<ConverterPara> &param,
                                                  std::shared_ptr<mindspore::Model> origin_model, int origin_model_size,
                                                  const InferenceParam &infer_param, double *init_scale,
                                                  std::vector<float> *candidate_scales, bool is_run_all) {
  CHECK_NULL_RETURN(param);
  CHECK_NULL_RETURN(origin_model);
  CHECK_NULL_RETURN(init_scale);
  CHECK_NULL_RETURN(candidate_scales);
  auto origin_out_tensor = origin_model->GetOutputs();
  const float threshold = 0.995f;
  float best_compress_ratio = 0.0f;
  float best_compress_mean_error = 0.0f;
  float best_compress_cos_sim = 0.0f;
  int best_compress_model_size = 0;
  size_t over_error_count = 0;
  for (size_t round = 0; round < infer_param.rounds; round++) {
    auto scale = infer_param.start_scale + round * infer_param.step;
    param->commonQuantParam.quant_type = schema::QuantType_QUANT_WEIGHT;
    FuncGraphPtr func_graph_bak;
    auto ret = CloneFuncGraph(func_graph, param, &func_graph_bak);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Clone FuncGraph failed.";
      return ret;
    }

    // quant
    auto quantizer = std::make_unique<quant::WeightQuantizer>(param);
    CHECK_NULL_RETURN(quantizer);
    auto status = quantizer->DoQuantize(func_graph_bak, scale);
    if (status != RET_OK) {
      MS_LOG(WARNING) << "DoQuantization failed " << status;
      continue;
    }

    MS_LOG(INFO) << "create quant session";
    int weight_quant_size;
    auto weight_quant_model = std::make_shared<mindspore::Model>();
    CHECK_NULL_RETURN(weight_quant_model);
    auto build_status = BuildModelByFuncGraph(weight_quant_model, func_graph_bak, param, &weight_quant_size);
    if (build_status != kSuccess) {
      MS_LOG(WARNING) << "build model failed!";
      continue;
    }
    auto weight_quant_inputs = weight_quant_model->GetInputs();
    for (auto input : weight_quant_inputs) {
      auto origin_tensor = origin_model->GetInputByTensorName(input.Name());
      auto weight_quant_tensor_data = input.MutableData();
      if (memcpy_s(weight_quant_tensor_data, input.DataSize(), origin_tensor.Data().get(), origin_tensor.DataSize()) !=
          EOK) {
        MS_LOG(ERROR) << "memcpy data failed.";
        return RET_ERROR;
      }
    }
    auto weight_quant_outputs = weight_quant_model->GetOutputs();
    auto model_status = weight_quant_model->Predict(weight_quant_inputs, &weight_quant_outputs);
    if (model_status != kSuccess) {
      MS_LOG(ERROR) << "Run origin session failed.";
      return RET_ERROR;
    }
    auto cos_sim = CompareDataByCosineDistance<float>(origin_model, weight_quant_model);
    auto mean_error = CompareData<float>(origin_model, weight_quant_model);

    if (!is_run_all) {
      const int tolerate_round = 3;
      if (cos_sim < threshold) {
        over_error_count++;
      }
      if (over_error_count <= 1) {
        candidate_scales->push_back(scale);
      }
      if (over_error_count >= tolerate_round) {
        MS_LOG(INFO) << "error count is over tolerate, the round is " << round;
        break;
      }
    }
    MS_CHECK_TRUE_MSG(weight_quant_size > 0, RET_ERROR, "weight quant size must large 0");
    const auto compress_ratio = 1.0 * origin_model_size / weight_quant_size;
    std::cout << " round:" << round << " scale:" << scale << " cos_sim:" << cos_sim << " mean_error:" << mean_error
              << " ratio:" << compress_ratio << std::endl;
    if (cos_sim >= threshold && compress_ratio > best_compress_ratio) {
      best_compress_ratio = compress_ratio;
      best_compress_mean_error = mean_error;
      best_compress_cos_sim = cos_sim;
      best_compress_model_size = weight_quant_size;
      *init_scale = scale;
    }
  }
  if (is_run_all) {
    std::cout << " best compress ratio:" << best_compress_ratio << " compressed model size:" << best_compress_model_size
              << " cos sim:" << best_compress_cos_sim << " mean error:" << best_compress_mean_error
              << " init scale:" << *init_scale << std::endl;
  }
  return RET_OK;
}

int ParameterOptimizer::OriginModelInference(const FuncGraphPtr &func_graph,
                                             const std::shared_ptr<ConverterPara> &param,
                                             std::shared_ptr<mindspore::Model> origin_model, int *origin_model_size) {
  CHECK_NULL_RETURN(param);
  CHECK_NULL_RETURN(origin_model);
  CHECK_NULL_RETURN(origin_model_size);
  FuncGraphPtr func_graph_bak;
  auto ret = CloneFuncGraph(func_graph, param, &func_graph_bak);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Clone FuncGraph failed.";
    return RET_ERROR;
  }
  param->commonQuantParam.quant_type = schema::QuantType_QUANT_NONE;
  *origin_model_size = 0;
  auto status = BuildModelByFuncGraph(origin_model, func_graph_bak, param, origin_model_size);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "build model failed!";
    return RET_ERROR;
  }
  auto origin_inputs = origin_model->GetInputs();
  for (auto input : origin_inputs) {
    if (param->dataPreProcessParam.calibrate_size > 0) {
      ret = preprocess::PreProcess(param->dataPreProcessParam, input.Name(), 0, &input);
    } else {
      ret = GenerateRandomData(&input);
    }
    if (ret != RET_OK) {
      MS_LOG(ERROR) << input.Name() << ":"
                    << "Generate random data failed.";
      return ret;
    }
  }
  auto origin_outputs = origin_model->GetOutputs();
  auto model_status = origin_model->Predict(origin_inputs, &origin_outputs);
  if (model_status != kSuccess) {
    MS_LOG(ERROR) << "Run origin predict failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ParameterOptimizer::GridSearchForScale(const FuncGraphPtr &func_graph, const std::shared_ptr<ConverterPara> &param,
                                           double *init_scale) {
  CHECK_NULL_RETURN(param);
  CHECK_NULL_RETURN(init_scale);

  double default_init_scale = *init_scale;

  auto origin_model = std::make_shared<mindspore::Model>();
  int origin_model_size;
  auto ret = OriginModelInference(func_graph, param, origin_model, &origin_model_size);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Origin Model Inference failed.";
    return ret;
  }

  float start_scale = 0.005f;
  const int giant_rounds = 10;
  float step = 0.005f;
  std::vector<float> candidate_scales;

  InferenceParam infer_param{};
  infer_param.rounds = giant_rounds;
  infer_param.start_scale = start_scale;
  infer_param.step = step;
  infer_param.thread_num = param->commonQuantParam.thread_num;

  std::cout << "==========Search with giant step==============\n";
  ret = WeightQuantModelInference(func_graph, param, origin_model, origin_model_size, infer_param, init_scale,
                                  &candidate_scales, false);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Weight quant graph inference failed.";
    return ret;
  }

  auto min_max = GetFineTuneRange(&candidate_scales);
  MS_LOG(INFO) << " min:" << min_max.min << " max:" << min_max.max;
  start_scale = min_max.min;
  if (min_max.max - min_max.min <= 0) {
    MS_LOG(WARNING) << "search reach max step, init_scale return default " << *init_scale;
    *init_scale = default_init_scale;
    return RET_OK;
  }
  const int baby_step_rounds = 25;
  step = (min_max.max - min_max.min) / baby_step_rounds;

  infer_param.rounds = baby_step_rounds;
  infer_param.start_scale = start_scale;
  infer_param.step = step;
  infer_param.thread_num = param->commonQuantParam.thread_num;
  std::cout << "==========Search with baby step==============\n";
  ret = WeightQuantModelInference(func_graph, param, origin_model, origin_model_size, infer_param, init_scale,
                                  &candidate_scales, true);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Weight quant graph inference failed.";
    return ret;
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
