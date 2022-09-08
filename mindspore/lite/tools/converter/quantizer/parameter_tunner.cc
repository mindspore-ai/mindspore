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
#include "tools/converter/preprocess/cv_calib_data.h"
#include "tools/converter/preprocess/image_preprocess.h"
#include "tools/converter/preprocess/opencv_utils.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/converter/quantizer/weight_quantizer.h"
#include "tools/converter/export_model.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/parser/parser_utils.h"

namespace mindspore::lite::quant {
namespace {
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
static const float kScaleFactor = (0.01 * 0.01 * 0.01 * 24.0);
}  // namespace
float GetMinScale(const std::set<tensor::TensorPtr> &weight_quantized_tensors) {
  size_t max_tensor_size = 1;
  for (const auto &tensor : weight_quantized_tensors) {
    max_tensor_size = std::max(max_tensor_size, tensor->DataSize());
  }
  return (max_tensor_size > 0) ? std::sqrt(kScaleFactor / max_tensor_size) : kScaleFactor;
}

static Status ExtendBatchSize(const std::shared_ptr<mindspore::Model> &model, std::vector<MSTensor> *inputs,
                              int batch) {
  std::vector<std::vector<int64_t>> dims(inputs->size());
  int i = 0;
  for (auto input : *inputs) {
    dims.at(i) = input.Shape();
    dims.at(i).at(0) = batch;
    input.SetShape(dims.at(i));
    i++;
  }
  return model->Resize(*inputs, dims);
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

int ParameterOptimizer::CopyDataAndRun(const std::shared_ptr<mindspore::Model> &origin_model,
                                       const std::shared_ptr<mindspore::Model> &quant_model) {
  auto weight_quant_inputs = quant_model->GetInputs();
  for (auto input : weight_quant_inputs) {
    auto origin_tensor = origin_model->GetInputByTensorName(input.Name());
    auto weight_quant_tensor_data = input.MutableData();
    if (memcpy_s(weight_quant_tensor_data, input.DataSize(), origin_tensor.MutableData(), origin_tensor.DataSize()) !=
        EOK) {
      MS_LOG(ERROR) << "memcpy data failed.";
      return RET_ERROR;
    }
  }
  auto weight_quant_outputs = quant_model->GetOutputs();
  auto model_status = quant_model->Predict(weight_quant_inputs, &weight_quant_outputs);
  if (model_status != kSuccess) {
    MS_LOG(ERROR) << "Run origin session failed.";
    return RET_ERROR;
  }

  return RET_OK;
}

int ParameterOptimizer::WeightQuantModelInference(const FuncGraphPtr &func_graph,
                                                  const std::shared_ptr<ConverterPara> &param,
                                                  const std::shared_ptr<mindspore::Model> &origin_model,
                                                  size_t origin_model_size, SearchParams *s_param, int *ret_scale,
                                                  float *best_compress_ratio, bool *found_valid_scale) {
  CHECK_NULL_RETURN(param);
  CHECK_NULL_RETURN(origin_model);
  CHECK_NULL_RETURN(ret_scale);
  CHECK_NULL_RETURN(best_compress_ratio);
  CHECK_NULL_RETURN(found_valid_scale);
  CHECK_NULL_RETURN(s_param);
  const float threshold = 0.995f;
  *best_compress_ratio = 0.0f;
  *found_valid_scale = false;
  for (int scale = s_param->range_start; scale <= s_param->range_end; scale += s_param->step) {
    param->commonQuantParam.quant_type = schema::QuantType_QUANT_WEIGHT;
    FuncGraphPtr func_graph_bak;
    auto ret = CloneFuncGraph(func_graph, param, &func_graph_bak);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Clone FuncGraph failed.";
      return ret;
    }

    auto quantizer = std::make_unique<quant::WeightQuantizer>(param, 1.0f / scale);
    CHECK_NULL_RETURN(quantizer);
    auto status = quantizer->DoQuantize(func_graph_bak);
    if (status != RET_OK) {
      MS_LOG(WARNING) << "DoQuantization failed " << status;
      continue;
    }

    if (scale == 1) {
      float inv_min_scale = GetMinScale(quantizer->GetWeightQuantizedTensors());
      if (inv_min_scale != 0) {
        if ((1.0 / inv_min_scale) > s_param->range_end) {
          // extend scale end
          int num_of_steps = (s_param->range_end - s_param->range_start) / s_param->step;
          s_param->range_end = static_cast<int>(1.0 / inv_min_scale);
          s_param->step = s_param->range_end / num_of_steps;
        }
      }
      std::cout << "=== Basic search in range [1," << s_param->range_end << "] === " << s_param->step << "\n";
    }

    MS_LOG(INFO) << "create quant session";
    size_t weight_quant_size = 0;
    auto weight_quant_model = std::make_shared<mindspore::Model>();
    CHECK_NULL_RETURN(weight_quant_model);
    auto build_status = BuildModelByFuncGraph(weight_quant_model, func_graph_bak, param, &weight_quant_size);
    if (build_status != kSuccess) {
      MS_LOG(WARNING) << "build model failed!";
      continue;
    }
    auto weight_quant_inputs = weight_quant_model->GetInputs();
    if ((param->dataPreProcessParam.calibrate_size == 0) && (param->mixedBitWeightQuantParam.use_cv_data)) {
      if (ExtendBatchSize(weight_quant_model, &weight_quant_inputs, kNumOfCalibrationImages) != kSuccess) {
        MS_LOG(ERROR) << "Resize session for CV calibration failed!";
        return RET_ERROR;
      }
    }
    auto model_status = CopyDataAndRun(origin_model, weight_quant_model);
    if (model_status != kSuccess) {
      MS_LOG(ERROR) << "Copy Input Data to model failed.";
      return RET_ERROR;
    }
    auto cos_sim = CompareDataByCosineDistance<float>(origin_model, weight_quant_model);
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

static int PrepareSingleImage(const uint8_t *buf, int len, const std::vector<int64_t> &shape, uint8_t *out_buf,
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

static int GenerateCvData(mindspore::MSTensor *tensor) {
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
  size_t t_size = tensor->DataSize();
  size_t loc = 0;

  for (int i = 0; i < num_of_images; i++) {
    size_t o_size = t_size - loc;
    auto ret = PrepareSingleImage(ims[i], im_sizes[i], tensor->Shape(), t_data + loc, &o_size);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Preparing Single image error";
      return ret;
    }
    loc += o_size;
  }
  return RET_OK;
}

int ParameterOptimizer::OriginModelInference(const FuncGraphPtr &func_graph,
                                             const std::shared_ptr<ConverterPara> &param,
                                             const std::shared_ptr<mindspore::Model> &origin_model,
                                             size_t *origin_model_size) {
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
  if ((param->dataPreProcessParam.calibrate_size == 0) && (param->mixedBitWeightQuantParam.use_cv_data)) {
    if (ExtendBatchSize(origin_model, &origin_inputs, kNumOfCalibrationImages) != kSuccess) {
      MS_LOG(ERROR) << "Resize session for CV calibration failed!";
      return RET_ERROR;
    }
  }

  for (auto input : origin_inputs) {
    if (param->dataPreProcessParam.calibrate_size > 0) {
      ret = preprocess::PreProcess(param->dataPreProcessParam, input.Name(), 0, &input);
    } else {
      if (param->mixedBitWeightQuantParam.use_cv_data && (input.Shape().size() == DIMENSION_4D) &&
          (input.Shape().at(0) == kNumOfCalibrationImages) &&
          ((input.Shape().at(kNHWC_C) == kOneChannel) || (input.Shape().at(kNHWC_C) == kThreeChannels) ||
           (input.Shape().at(kNHWC_C) == kSixChannels)) &&
          (input.DataType() == DataType::kNumberTypeFloat32)) {
        ret = GenerateCvData(&input);
      } else {
        ret = GenerateRandomData(&input);
      }
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

  auto origin_model = std::make_shared<mindspore::Model>();
  size_t origin_model_size;
  int ord_param = kDefaultRangeStart;
  auto ret = OriginModelInference(func_graph, param, origin_model, &origin_model_size);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Origin Model Inference failed.";
    return ret;
  }

  float best_compress_ratio = 0;
  bool found_valid_scale = false;
  int steps_per_stage = param->mixedBitWeightQuantParam.max_iterations / (kMaxFineSearchIterations + 1);
  if (steps_per_stage > kMaxStepCoarseSearch) {
    steps_per_stage = kMaxStepCoarseSearch;
  }

  int range_start = kDefaultRangeStart;
  int range_end = kDefaultRangeEnd;
  int step = (range_end - range_start) / steps_per_stage;
  SearchParams search_param = {range_start, range_end, step};

  std::cout << "====== Search for the best scale =======\n";
  ret = WeightQuantModelInference(func_graph, param, origin_model, origin_model_size, &search_param, &ord_param,
                                  &best_compress_ratio, &found_valid_scale);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Weight quant graph inference failed.";
    return ret;
  }
  step = search_param.step;

  if (found_valid_scale == false || (ord_param == kDefaultRangeStart && (best_compress_ratio < kCompressToInt8Ratio))) {
    range_start = search_param.range_end;
    range_end = kDefaultExtendFactor * search_param.range_end;
    SearchParams wider_range_param = {range_start, range_end, step};
    std::cout << "=== Couldn't find proper compression, extending the search range ===\n";
    ret = WeightQuantModelInference(func_graph, param, origin_model, origin_model_size, &wider_range_param, &ord_param,
                                    &best_compress_ratio, &found_valid_scale);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Weight quant graph inference failed.";
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
    int prev_prev_val = ord_param - kTwo * step;
    range_end = ord_param;
    range_start = std::max(1, prev_prev_val);
    step = (range_end - range_start) / steps_per_stage;
    if (step < static_cast<int>(sqrt(static_cast<float>(range_end - range_start))) / kTwo) {
      step = static_cast<int>(sqrt(static_cast<float>(range_end - range_start))) / kTwo;
    }
    range_start = ord_param - ((ord_param - range_start) / step) * step;  // align search to meet prev scale
    if ((range_start == search_param.range_start) || (range_start == prev_prev_val) || (range_start == 1)) {
      range_start += step;
    }

    search_param.range_start = range_start;
    search_param.range_end = range_end;
    search_param.step = step;
    std::cout << "=== Fine search " << search_cnt << " in range [" << range_start << "," << range_end << "] ===\n";
    ret = WeightQuantModelInference(func_graph, param, origin_model, origin_model_size, &search_param, &ord_param,
                                    &best_compress_ratio, &found_valid_scale);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Weight quant graph inference failed.";
      return ret;
    }
    if (step <= kFinestGranularity) {
      break;
    }
  }
  std::cout << "best compression is " << best_compress_ratio << " at scale " << ord_param << std::endl;
  *init_scale = 1.0 / (ord_param);
  return RET_OK;
}
}  // namespace mindspore::lite::quant
