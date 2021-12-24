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

int ParameterOptimizer::CloneFuncGraph(const FuncGraphPtr &func_graph, converter::Flags *flags,
                                       FuncGraphPtr *func_graph_bak) {
  CHECK_NULL_RETURN(func_graph_bak);
  CHECK_NULL_RETURN(flags);
  std::map<FuncGraphPtr, FuncGraphPtr> cloned_func_graph;
  *func_graph_bak = lite::CloneFuncGraph(func_graph, flags, &cloned_func_graph);
  CHECK_NULL_RETURN(*func_graph_bak);
  static auto root_func_manager = Manage(*func_graph_bak);
  std::set<FuncGraphPtr> all_func_graphs = {};
  lite::GetAllFuncGraph(*func_graph_bak, &all_func_graphs);
  for (const auto &graph : all_func_graphs) {
    graph->set_manager(root_func_manager);
  }
  return RET_OK;
}

int ParameterOptimizer::WeightQuantModelInference(const FuncGraphPtr &func_graph, converter::Flags *flags,
                                                  session::LiteSession *origin_session, int origin_model_size,
                                                  const InferenceParam &param, double *init_scale,
                                                  std::vector<float> *candidate_scales, bool is_run_all) {
  CHECK_NULL_RETURN(flags);
  CHECK_NULL_RETURN(origin_session);
  CHECK_NULL_RETURN(init_scale);
  CHECK_NULL_RETURN(candidate_scales);
  auto origin_out_tensor = origin_session->GetOutputs();
  const float threshold = 0.995f;
  float best_compress_ratio = 0.0f;
  float best_compress_mean_error = 0.0f;
  float best_compress_cos_sim = 0.0f;
  int best_compress_model_size = 0;
  size_t over_error_count = 0;
  for (size_t round = 0; round < param.rounds; round++) {
    auto scale = param.start_scale + round * param.step;
    flags->commonQuantParam.quant_type = schema::QuantType_QUANT_WEIGHT;
    FuncGraphPtr func_graph_bak;
    auto ret = CloneFuncGraph(func_graph, flags, &func_graph_bak);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Clone FuncGraph failed.";
      return ret;
    }

    // quant
    auto quantizer = std::make_unique<quant::WeightQuantizer>(*flags);
    CHECK_NULL_RETURN(quantizer);
    auto status = quantizer->DoQuantize(func_graph_bak, scale);
    if (status != RET_OK) {
      MS_LOG(WARNING) << "DoQuantization failed " << status;
      continue;
    }

    MS_LOG(INFO) << "create quant session";
    int weight_quant_size;
    auto weight_quant_sm = CreateSessionByFuncGraph(func_graph_bak, *flags, param.thread_num, &weight_quant_size);
    auto weight_quant_session = weight_quant_sm.session;
    auto weight_quant_model = weight_quant_sm.model;
    if (weight_quant_session == nullptr || weight_quant_model == nullptr) {
      MS_LOG(WARNING) << "create session failed!";
      continue;
    }
    auto weight_quant_inputs = weight_quant_session->GetInputs();
    for (auto input : weight_quant_inputs) {
      auto origin_tensor = origin_session->GetInputsByTensorName(input->tensor_name());
      auto weight_quant_tensor_data = input->MutableData();
      if (memcpy_s(weight_quant_tensor_data, input->Size(), origin_tensor->data(), origin_tensor->Size()) != EOK) {
        MS_LOG(ERROR) << "memcpy data failed.";
        delete weight_quant_session;
        delete weight_quant_model;
        return RET_ERROR;
      }
    }
    weight_quant_session->BindThread(true);
    ret = weight_quant_session->RunGraph();
    weight_quant_session->BindThread(false);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Run origin session failed.";
      delete weight_quant_session;
      delete weight_quant_model;
      return ret;
    }
    auto weight_quant_tensor = weight_quant_session->GetOutputs();
    auto cos_sim = CompareDataByCosineDistance<float>(origin_out_tensor, weight_quant_tensor);
    auto mean_error = CompareData<float>(origin_out_tensor, weight_quant_tensor);

    delete weight_quant_session;
    delete weight_quant_model;

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
    auto compress_ratio = 1.0 * origin_model_size / weight_quant_size;
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

int ParameterOptimizer::OriginModelInference(const FuncGraphPtr &func_graph, converter::Flags *flags, SessionModel *sm,
                                             int *origin_model_size) {
  FuncGraphPtr func_graph_bak;
  auto ret = CloneFuncGraph(func_graph, flags, &func_graph_bak);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Clone FuncGraph failed.";
    return RET_ERROR;
  }
  flags->commonQuantParam.quant_type = schema::QuantType_QUANT_NONE;
  *origin_model_size = 0;
  *sm = CreateSessionByFuncGraph(func_graph_bak, *flags, flags->commonQuantParam.thread_num, origin_model_size);
  auto origin_session = sm->session;
  auto origin_model = sm->model;
  if (origin_session == nullptr || origin_model == nullptr) {
    MS_LOG(ERROR) << "create session failed!";
    return RET_ERROR;
  }
  auto origin_inputs = origin_session->GetInputs();
  for (auto input : origin_inputs) {
    if (flags->dataPreProcessParam.calibrate_size > 0) {
      ret = preprocess::PreProcess(flags->dataPreProcessParam, input->tensor_name(), 0, input);
    } else {
      ret = GenerateRandomData(input);
    }
    if (ret != RET_OK) {
      MS_LOG(ERROR) << input->tensor_name() << ":"
                    << "Generate random data failed.";
      return ret;
    }
  }
  origin_session->BindThread(true);
  ret = origin_session->RunGraph();
  origin_session->BindThread(false);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run origin session failed.";
    return ret;
  }
  return RET_OK;
}

int ParameterOptimizer::GridSearchForScale(const FuncGraphPtr &func_graph, converter::Flags *flags,
                                           double *init_scale) {
  CHECK_NULL_RETURN(flags);
  CHECK_NULL_RETURN(init_scale);

  double default_init_scale = *init_scale;

  SessionModel sm;
  int origin_model_size;
  auto ret = OriginModelInference(func_graph, flags, &sm, &origin_model_size);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Origin Model Inference failed.";
    return ret;
  }
  auto origin_session = sm.session;
  auto origin_model = sm.model;

  float start_scale = 0.005f;
  const int giant_rounds = 10;
  float step = 0.005f;
  std::vector<float> candidate_scales;

  InferenceParam param{};
  param.rounds = giant_rounds;
  param.start_scale = start_scale;
  param.step = step;
  param.thread_num = flags->commonQuantParam.thread_num;

  std::cout << "==========Search with giant step==============\n";
  ret = WeightQuantModelInference(func_graph, flags, origin_session, origin_model_size, param, init_scale,
                                  &candidate_scales, false);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Weight quant graph inference failed.";
    delete origin_session;
    delete origin_model;
    return ret;
  }

  auto min_max = GetFineTuneRange(&candidate_scales);
  MS_LOG(INFO) << " min:" << min_max.min << " max:" << min_max.max;
  start_scale = min_max.min;
  if (min_max.max - min_max.min <= 0) {
    MS_LOG(WARNING) << "search reach max step, init_scale return default " << *init_scale;
    *init_scale = default_init_scale;
    delete origin_session;
    delete origin_model;
    return RET_OK;
  }
  int babysitting_rounds = 25;
  step = (min_max.max - min_max.min) / babysitting_rounds;

  param.rounds = babysitting_rounds;
  param.start_scale = start_scale;
  param.step = step;
  param.thread_num = flags->commonQuantParam.thread_num;
  std::cout << "==========Search with babysitting step==============\n";
  ret = WeightQuantModelInference(func_graph, flags, origin_session, origin_model_size, param, init_scale,
                                  &candidate_scales, true);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Weight quant graph inference failed.";
    delete origin_session;
    delete origin_model;
    return ret;
  }
  delete origin_session;
  delete origin_model;
  return RET_OK;
}
}  // namespace mindspore::lite::quant
