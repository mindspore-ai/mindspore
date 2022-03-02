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
#include "tools/converter/preprocess/cv_calib_data.h"
#include "tools/converter/preprocess/image_preprocess.h"
#include "tools/converter/preprocess/opencv_utils.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/converter/quantizer/weight_quantizer.h"
#include "tools/converter/export_model.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/parser/parser_utils.h"

namespace mindspore::lite::quant {
static const int kOneChannel = 1;
static const int kThreeChannels = 3;
static const int kSixChannels = 6;
static const int kDefaultRangeStart = 1;
static const int kDefaultRangeEnd = 30000;
static const int kDefaultExtendFactor = 4;
static const int kMaxStepCoarseSearch = 173;
static const int kMaxStepFineSearch = 20;
static const int kMaxFineSearchIterations = 3;
static const int kFinestGranularity = 2;
static const int kCompressToInt8Ratio = 4.0;
static const int kTwo = 2;

static int ExtendBatchSize(mindspore::session::LiteSession *session, Vector<tensor::MSTensor *> *inputs, int batch) {
  Vector<Vector<int>> dims(inputs->size());
  int i = 0;
  for (auto input : *inputs) {
    dims.at(i) = input->shape();
    dims.at(i).at(0) = batch;
    input->set_shape(dims.at(i));
    i++;
  }
  return session->Resize(*inputs, dims);
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

int ParameterOptimizer::CopyDataAndRun(session::LiteSession *origin_session, session::LiteSession *quant_session) {
  auto weight_quant_inputs = quant_session->GetInputs();
  for (auto input : weight_quant_inputs) {
    auto origin_tensor = origin_session->GetInputsByTensorName(input->tensor_name());
    auto weight_quant_tensor_data = input->MutableData();
    if (memcpy_s(weight_quant_tensor_data, input->Size(), origin_tensor->data(), origin_tensor->Size()) != EOK) {
      MS_LOG(ERROR) << "memcpy data failed.";
      return RET_ERROR;
    }
  }
  quant_session->BindThread(true);
  auto ret = quant_session->RunGraph();
  quant_session->BindThread(false);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run origin session failed.";
    return ret;
  }
  return RET_OK;
}

int ParameterOptimizer::WeightQuantModelInference(const FuncGraphPtr &func_graph, converter::Flags *flags,
                                                  session::LiteSession *origin_session, int origin_model_size,
                                                  InferenceParam *param, int *ret_scale, float *best_compress_ratio,
                                                  bool *found_valid_scale) {
  CHECK_NULL_RETURN(flags);
  CHECK_NULL_RETURN(origin_session);
  CHECK_NULL_RETURN(ret_scale);
  CHECK_NULL_RETURN(best_compress_ratio);
  CHECK_NULL_RETURN(found_valid_scale);
  auto origin_out_tensor = origin_session->GetOutputs();
  const float threshold = 0.995f;
  *best_compress_ratio = 0.0f;
  *found_valid_scale = false;
  for (int scale = param->range_start; scale <= param->range_end; scale += param->step) {
    flags->commonQuantParam.quant_type = schema::QuantType_QUANT_WEIGHT;
    FuncGraphPtr func_graph_bak;
    auto ret = CloneFuncGraph(func_graph, flags, &func_graph_bak);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Clone FuncGraph failed.";
      return ret;
    }
    auto quantizer = std::make_unique<quant::WeightQuantizer>(*flags);
    CHECK_NULL_RETURN(quantizer);
    auto status = quantizer->DoQuantize(func_graph_bak, 1.0f / scale);
    if (status != RET_OK) {
      MS_LOG(WARNING) << "DoQuantization failed " << status;
      continue;
    }
    if (scale == 1) {
      float inv_min_scale = quantizer->GetMinScale();
      if (inv_min_scale != 0) {
        if ((1.0 / inv_min_scale) > param->range_end) {
          // extend scale end
          int num_of_steps = (param->range_end - param->range_start) / param->step;
          param->range_end = static_cast<int>(1.0 / inv_min_scale);
          param->step = param->range_end / num_of_steps;
        }
      }
      std::cout << "=== Basic search in range [1," << param->range_end << "] ===\n";
    }

    MS_LOG(INFO) << "create quant session";
    int weight_quant_size;
    auto weight_quant_sm = CreateSessionByFuncGraph(func_graph_bak, *flags, param->thread_num, &weight_quant_size);
    auto weight_quant_session = weight_quant_sm.session;
    auto weight_quant_model = weight_quant_sm.model;
    if (weight_quant_session == nullptr || weight_quant_model == nullptr) {
      MS_LOG(WARNING) << "create session failed!";
      continue;
    }
    auto weight_quant_inputs = weight_quant_session->GetInputs();
    if ((flags->dataPreProcessParam.calibrate_size == 0) && (flags->mixedBitWeightQuantParam.use_cv_data)) {
      if (ExtendBatchSize(weight_quant_session, &weight_quant_inputs, kNumOfCalibrationImages) != RET_OK) {
        MS_LOG(ERROR) << "Resize session for CV calibration failed!";
        return RET_ERROR;
      }
    }
    ret = CopyDataAndRun(origin_session, weight_quant_session);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Run origin session failed.";
      delete weight_quant_session;
      delete weight_quant_model;
      return ret;
    }
    auto weight_quant_tensor = weight_quant_session->GetOutputs();
    auto cos_sim = CompareDataByCosineDistance<float>(origin_out_tensor, weight_quant_tensor);
    delete weight_quant_session;
    delete weight_quant_model;
    MS_CHECK_TRUE_MSG(weight_quant_size > 0, RET_ERROR, "weight quant size must be larger than 0");
    const auto compress_ratio = 1.0 * origin_model_size / weight_quant_size;
    std::cout << " scale:" << scale << " cos_sim:" << cos_sim << " compression ratio:" << compress_ratio << std::endl;
    if (cos_sim >= threshold) {
      if (compress_ratio > *best_compress_ratio) {
        *best_compress_ratio = compress_ratio;
        *found_valid_scale = true;
        *ret_scale = scale;
        return RET_OK;
      }
    }
  }
  *found_valid_scale = false;
  MS_LOG(DEBUG) << "Couldn't reach cosine similarity constraint";
  return RET_OK;
}

static int PrepareSingleImage(const uint8_t *buf, int len, const Vector<int> &shape, uint8_t *out_buf,
                              size_t *out_len) {
  cv::Mat mat;
  const int HEIGHT_INDEX = 1;
  const int WIDTH_INDEX = 2;
  const std::vector<double> mean = {127.5, 127.5, 127.5};
  const std::vector<double> standard_deviation = {127.5, 127.5, 127.5};
  auto ret = preprocess::DecodeBuffer(buf, len, &mat);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PrepareSingleImage error in decode: " << ret;
    return ret;
  }
  ret = preprocess::Resize(&mat, shape.at(WIDTH_INDEX), shape.at(HEIGHT_INDEX), cv::INTER_LINEAR);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PrepareSingleImage error in Resize: " << ret;
    return ret;
  }
  ret = preprocess::Normalize(&mat, mean, standard_deviation);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PrepareSingleImage error in Normalize: " << ret;
    return ret;
  }
  if (shape.at(kNHWC_C) == kOneChannel) {
    ret = preprocess::ConvertImageFormat(&mat, cv::COLOR_BGR2GRAY);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "PrepareSingleImage error in Gray Scaling: " << ret;
      return ret;
    }
  }

  if (shape.at(kNHWC_C) == kSixChannels) {  // in case of 2 stacked images
    std::vector<cv::Mat> channels(kThreeChannels);
    cv::split(mat, channels);
    std::vector<cv::Mat> mat_6_channels;
    for (int i = 0; i < kThreeChannels; i++) {
      mat_6_channels.push_back(channels[i]);
    }
    for (int i = 0; i < kThreeChannels; i++) {
      mat_6_channels.push_back(channels[i]);
    }
    cv::merge(mat_6_channels, mat);
  }

  uint8_t *data = nullptr;
  size_t size = 0;
  ret = preprocess::GetMatData(mat, reinterpret_cast<void **>(&data), &size);
  if (data == nullptr || size == 0) {
    MS_LOG(ERROR) << "GetMatData data is nullptr or size == 0";
    return RET_ERROR;
  }
  if (ret != RET_OK) {
    delete[] data;
    MS_LOG(ERROR) << "Get mat data failed.";
    return ret;
  }

  if (size > *out_len) {
    delete[] data;
    MS_LOG(ERROR) << "Buffer Size mismatch " << size << " vs " << *out_len;
    return ret;
  }
  std::copy(data, data + size, out_buf);
  delete[] data;
  *out_len = size;
  return RET_OK;
}

static int GenerateCvData(mindspore::tensor::MSTensor *tensor) {
  MS_ASSERT(tensor != nullptr);
  auto input_data = tensor->MutableData();
  if (input_data == nullptr) {
    MS_LOG(ERROR) << "MallocData for tensor failed";
    return RET_ERROR;
  }

  const int num_of_images = kNumOfCalibrationImages;
  const uint8_t *ims[num_of_images] = {COCO_train2014_0581821, COCO_train2014_0581821, COCO_train2014_0581821};
  int im_sizes[num_of_images] = {sizeof(COCO_train2014_0581882), sizeof(COCO_train2014_0581909),
                                 sizeof(COCO_train2014_0581909)};
  uint8_t *t_data = reinterpret_cast<uint8_t *>(tensor->MutableData());
  size_t t_size = tensor->Size();
  size_t loc = 0;

  for (int i = 0; i < num_of_images; i++) {
    size_t o_size = t_size - loc;
    auto ret = PrepareSingleImage(ims[i], im_sizes[i], tensor->shape(), t_data + loc, &o_size);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Preparing Single image error";
      return ret;
    }
    loc += o_size;
  }
  return RET_OK;
}

int ParameterOptimizer::OriginModelInference(const FuncGraphPtr &func_graph, converter::Flags *flags, SessionModel *sm,
                                             int *origin_model_size) {
  CHECK_NULL_RETURN(flags);
  CHECK_NULL_RETURN(sm);
  CHECK_NULL_RETURN(origin_model_size);
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

  if ((flags->dataPreProcessParam.calibrate_size == 0) && (flags->mixedBitWeightQuantParam.use_cv_data)) {
    if (ExtendBatchSize(origin_session, &origin_inputs, kNumOfCalibrationImages) != RET_OK) {
      MS_LOG(ERROR) << "Resize session for CV calibration failed!";
      return RET_ERROR;
    }
  }

  for (auto input : origin_inputs) {
    if (flags->dataPreProcessParam.calibrate_size > 0) {
      ret = preprocess::PreProcess(flags->dataPreProcessParam, input->tensor_name(), 0, input);
    } else {
      if (flags->mixedBitWeightQuantParam.use_cv_data && (input->shape().size() == DIMENSION_4D) &&
          (input->shape().at(0) == kNumOfCalibrationImages) &&
          ((input->shape().at(kNHWC_C) == kOneChannel) || (input->shape().at(kNHWC_C) == kThreeChannels) ||
           (input->shape().at(kNHWC_C) == kSixChannels)) &&
          ((input->data_type() == kNumberTypeFloat32) || input->data_type() == kNumberTypeFloat)) {
        ret = GenerateCvData(input);
      } else {
        ret = GenerateRandomData(input);
      }
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

  SessionModel sm;
  int origin_model_size;
  int search_param = kDefaultRangeStart;
  auto ret = OriginModelInference(func_graph, flags, &sm, &origin_model_size);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Origin Model Inference failed.";
    return ret;
  }
  auto origin_session = sm.session;
  auto origin_model = sm.model;

  float best_compress_ratio = 0;
  bool found_valid_scale = false;
  int steps_per_stage = flags->mixedBitWeightQuantParam.max_iterations / (kMaxFineSearchIterations + 1);
  if (steps_per_stage > kMaxStepCoarseSearch) {
    steps_per_stage = kMaxStepCoarseSearch;
  }

  int range_start = kDefaultRangeStart;
  int range_end = kDefaultRangeEnd;
  int step = kDefaultRangeEnd / steps_per_stage;
  InferenceParam param = {range_start, range_end, step, flags->commonQuantParam.thread_num};

  std::cout << "====== Search for the best scale =======\n";
  ret = WeightQuantModelInference(func_graph, flags, origin_session, origin_model_size, &param, &search_param,
                                  &best_compress_ratio, &found_valid_scale);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Weight quant graph inference failed.";
    delete origin_session;
    delete origin_model;
    return ret;
  }

  if (found_valid_scale == false ||
      (search_param == kDefaultRangeStart && (best_compress_ratio < kCompressToInt8Ratio))) {
    range_start = param.range_end;
    range_end = kDefaultExtendFactor * param.range_end;
    InferenceParam wider_range_param = {range_start, range_end, step, flags->commonQuantParam.thread_num};
    std::cout << "=== Couldn't find proper compression, extending the search range ===\n";
    ret = WeightQuantModelInference(func_graph, flags, origin_session, origin_model_size, &wider_range_param,
                                    &search_param, &best_compress_ratio, &found_valid_scale);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Weight quant graph inference failed.";
      delete origin_session;
      delete origin_model;
      return ret;
    }
    if (found_valid_scale == false) {
      std::cout << "=== Couldn't find compression that will match similarity constraints. Aborting! ===\n";
      std::cout << "======================= You may try fixed 8bit quantization =======================\n";
      return RET_ERROR;
    }
  }

  if (steps_per_stage > kMaxStepFineSearch) {
    steps_per_stage = kMaxStepFineSearch;
  }
  for (int search_cnt = 0; search_cnt < kMaxFineSearchIterations; search_cnt++) {
    int prev_prev_val = search_param - kTwo * step;
    range_end = search_param;
    range_start = std::max(1, prev_prev_val);
    step = (range_end - range_start) / steps_per_stage;
    if (step < static_cast<int>(sqrt(static_cast<float>(range_end - range_start))) / kTwo) {
      step = static_cast<int>(sqrt(static_cast<float>(range_end - range_start))) / kTwo;
    }
    range_start = search_param - ((search_param - range_start) / step) * step;  // align search to meet prev scale
    if ((range_start == param.range_start) || (range_start == prev_prev_val) || (range_start == 1)) {
      range_start += step;
    }

    param.range_start = range_start;
    param.range_end = range_end;
    param.step = step;
    std::cout << "=== Fine search " << search_cnt << " in range [" << range_start << "," << range_end << "] ===\n";
    ret = WeightQuantModelInference(func_graph, flags, origin_session, origin_model_size, &param, &search_param,
                                    &best_compress_ratio, &found_valid_scale);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Weight quant graph inference failed.";
      delete origin_session;
      delete origin_model;
      return ret;
    }
    if (step <= kFinestGranularity) {
      break;
    }
  }

  std::cout << "best compression is " << best_compress_ratio << " at scale " << search_param << std::endl;
  *init_scale = 1.0 / (search_param);
  delete origin_session;
  delete origin_model;
  return RET_OK;
}
}  // namespace mindspore::lite::quant
