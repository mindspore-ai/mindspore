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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANTIZER_UTIL_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANTIZER_UTIL_H

#include <dirent.h>
#include <sys/stat.h>
#include <memory>
#include <string>
#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <limits>
#include <utility>
#include "ops/mat_mul.h"
#include "ops/fusion/full_connection.h"
#include "tools/converter/quantizer/quantizer.h"
#include "include/errorcode.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "include/model.h"
#include "base/base.h"
#include "ir/primitive.h"
#include "abstract/dshape.h"
#include "tools/converter/quantizer/huffman_encode.h"
#include "tools/converter/quantizer/bitpacking.h"
#include "src/lite_session.h"
#include "tools/converter/graphdef_transform.h"
#include "src/common/file_utils.h"

namespace mindspore::lite::quant {
static constexpr size_t UINT8_QUANTIZATION = 8;
static constexpr size_t WEIGHT_INDEX = 1;

const char kMethodMaxMin[] = "MAX_MIN";
const char kMethodKL[] = "KL";
const char kMethodOutlier[] = "RemovalOutlier";

struct PostQuantConfig {
  std::vector<std::string> image_paths;
  uint32_t batch_count{100};
  std::string method_x{kMethodKL};
  uint32_t thread_num{1};
  bool bias_correction{false};
  bool mixed{false};
  float mean_error_threshold{0.04};
  std::vector<std::vector<std::vector<int>>> input_shapes;  // different input
  bool inited{false};
};

struct SessionModel {
  session::LiteSession *session{nullptr};
  Model *model{nullptr};
};

/**
 * 1. when op's weight size > mWeightSize just skip
 * 2. only do conv/deconv/convdepthwise/deconvdepthwise/mul/matmul/batchmatmul quantization
 * 3. when conv/deconv/convdepthwise/deconvdepthwise ops' weight channel size > covWeightQuantChannelThreshold just skip
 * */
class QuantStrategy {
 public:
  explicit QuantStrategy(size_t weightSize, size_t covWeightQuantChannelThreshold = 16);

  ~QuantStrategy() = default;

  bool CanConvOpQuantized(const CNodePtr &node) const;
  bool CanMulOpQuantized(const CNodePtr &node) const;
  bool CanOpPostQuantized(AnfNodePtr &node) const;

  size_t m_weight_size_;
  size_t m_conv_weight_quant_channel_threshold_;

 private:
  static const std::vector<std::string> conv_types_;
  static const std::vector<std::string> mul_types_;
};

constexpr float delta = 0.1;
constexpr float ratio = 10.0;
constexpr int percent = 10;
constexpr int quant_param_size = 32 * 8;

QuantParamHolderPtr GetCNodeQuantHolder(const PrimitivePtr &primitive);

STATUS CalQuantizationParams(schema::QuantParamT *quantParam, double mMin, double mMax, bool narrowRange, int quant_max,
                             int quant_min, int num_bits);

STATUS CalQuantizationParams(schema::QuantParamT *quantParam, double mMin, double mMax, bool narrowRange = false,
                             int numBits = UINT8_QUANTIZATION);

std::pair<float, float> OutlierMethod(std::vector<float> min_datas, std::vector<float> max_datas);

std::vector<int8_t> KMeans(float *data, size_t elem_count, size_t k, size_t epochs, schema::QuantParamT *quantParam);

STATUS UpdateTensorDataAndSize(ParamValueLitePtr weight, void *quant_datas, int new_size);

void GetMaxMinPerchannel(int channels, int one_filter_size, int i, int elem_count, const float *raw_datas,
                         bool channel_at_first, float *desired_max, float *desired_min);

template <typename T>
T QuantizeData(const float originData, const schema::QuantParamT *quantParam) {
  MS_ASSERT(quantParam != nullptr);
  MS_ASSERT(quantParam->inited);
  const auto scale = quantParam->scale;
  const auto zeroPoint = quantParam->zeroPoint;
  const auto numBit = quantParam->numBits;
  const auto narrowRange = quantParam->narrowRange;
  double maxLimitTemp = static_cast<float>((1 << (unsigned int)numBit) - 1);
  const double maxLimit = static_cast<float>(maxLimitTemp - zeroPoint + std::numeric_limits<T>::min()) * scale;
  double minLimit;
  if (narrowRange) {
    minLimit = static_cast<float>(std::numeric_limits<T>::min() + 1 - zeroPoint) * scale;
  } else {
    minLimit = static_cast<float>(std::numeric_limits<T>::min() - zeroPoint) * scale;
  }

  return [maxLimit, minLimit, zeroPoint, scale, narrowRange, originData] {
    double tmp;
    if (originData > maxLimit) {
      tmp = maxLimit;
    } else if (originData < minLimit) {
      tmp = minLimit;
    } else {
      tmp = originData;
    }
    auto quantData = static_cast<T>(std::round(zeroPoint + tmp / scale));
    return quantData;
  }();
}

template <typename T>
T QuantizeData(float originData, const schema::QuantParamT &quantParam, int quant_max, int quant_min) {
  MS_ASSERT(quantParam != nullptr);
  MS_ASSERT(quantParam->inited);
  const auto scale = quantParam.scale;
  const int zeroPoint = quantParam.zeroPoint;
  const auto narrowRange = quantParam.narrowRange;
  const int maxLimit = quant_max;
  const int minLimit = quant_min;

  return [maxLimit, minLimit, zeroPoint, scale, narrowRange, originData] {
    auto quant_data = std::round(originData / scale + zeroPoint);
    if (quant_data > maxLimit) {
      quant_data = maxLimit;
    } else if (quant_data < minLimit) {
      quant_data = minLimit;
    }
    return static_cast<T>(quant_data);
  }();
}

template <typename T>
STATUS DoPerChannelQuant(const ParamValueLitePtr &weight, const QuantType &quant_type,
                         std::vector<schema::QuantParamT> *quant_params, const int &quant_max, const int &quant_min,
                         const size_t &bit_num, const bool &k_means, std::vector<T> *quant_datas,
                         std::vector<float> *dequant_datas, bool channel_at_first = true) {
  auto dims = weight->tensor_shape();
  size_t elem_count = weight->tensor_shape_size();
  auto *raw_datas = static_cast<float *>(weight->tensor_addr());
  auto channels = dims[0];
  if (!channel_at_first) {
    if (dims.size() != 2) {
      MS_LOG(ERROR) << "unexpected dims size: " << dims.size();
      channel_at_first = true;
    } else {
      channels = dims[1];
    }
  }
  if (channels == 0) {
    MS_LOG(ERROR) << "channels is zero";
    return RET_ERROR;
  }
  size_t one_filter_size = elem_count / channels;
  bool do_quant = quant_param_size / (sizeof(float) * 8 - bit_num) < one_filter_size;
  if (!do_quant && quant_type == QuantType_WeightQuant) {
    MS_LOG(INFO) << "too few elements in a filter, no need to quantize. " << one_filter_size;
    return RET_CONTINUE;
  }
  for (int i = 0; i < channels; i++) {
    float min = FLT_MAX;
    float max = -FLT_MAX;
    GetMaxMinPerchannel(channels, one_filter_size, i, elem_count, raw_datas, channel_at_first, &max, &min);
    schema::QuantParamT quant_param;
    STATUS status = CalQuantizationParams(&quant_param, min, max, false, quant_max, quant_min, bit_num);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "CalQuantizationParams failed" << status;
      return status;
    }
    // do quantization
    double average_dequant = 0;
    double average_raw = 0;
    for (uint32_t j = 0; j < one_filter_size; j++) {
      auto index = j + i * one_filter_size;
      if (!channel_at_first) {
        index = j * channels + i;
      }
      MS_ASSERT(index < elem_count);
      float raw_data = raw_datas[index];
      auto quant_data = QuantizeData<T>(raw_data, quant_param, quant_max, quant_min);
      (*quant_datas)[index] = quant_data;

      if (quant_type == QuantType_WeightQuant) {
        float dequant_data = quant_param.scale * (quant_data - quant_param.zeroPoint);
        (*dequant_datas)[index] = dequant_data;
        average_dequant += dequant_data;
        average_raw += raw_data;
      }
    }
    if (quant_type == QuantType_WeightQuant && !k_means) {
      // mean
      average_dequant = average_dequant / one_filter_size;
      average_raw = average_raw / one_filter_size;
      // std
      double variance_dequant = 0;
      double variance_raw = 0;
      for (uint32_t j = 0; j < one_filter_size; j++) {
        auto index = j + i * one_filter_size;
        if (!channel_at_first) {
          index = j * channels + i;
        }
        MS_ASSERT(index < elem_count);
        variance_dequant += std::pow((*dequant_datas)[index] - average_dequant, 2);
        variance_raw += std::pow(raw_datas[index] - average_raw, 2);
      }
      variance_dequant = std::sqrt(variance_dequant / one_filter_size);
      variance_raw = std::sqrt(variance_raw / one_filter_size);
      quant_param.varCorr = 1;
      if (variance_raw != 0 && variance_dequant != 0) {
        auto temp_var_corr = variance_raw / variance_dequant;
        if (temp_var_corr > 0 && temp_var_corr < 10) {
          quant_param.varCorr = temp_var_corr;
        } else {
          MS_LOG(WARNING) << "unexpected var_corr: " << temp_var_corr;
        }
      }
      quant_param.meanCorr = average_raw - average_dequant * quant_param.varCorr;
    }
    quant_params->emplace_back(quant_param);
  }
  auto status = UpdateTensorDataAndSize(weight, quant_datas->data(), quant_datas->size() * sizeof(T));
  if (status != RET_OK) {
    MS_LOG(ERROR) << "UpdateTensorDataAndSize error";
    return RET_ERROR;
  }
  return RET_OK;
}

template <typename T>
STATUS DoPerLayerQuant(const ParamValueLitePtr &weight, const QuantType &quant_type,
                       std::vector<schema::QuantParamT> *quant_params, const int &quant_max, const int &quant_min,
                       const size_t &bit_num, const bool &k_means, std::vector<T> *quant_datas) {
  auto dims = weight->tensor_shape();
  size_t elem_count = weight->tensor_shape_size();
  auto *raw_datas = static_cast<float *>(weight->tensor_addr());
  float min = FLT_MAX;
  float max = -FLT_MIN;
  for (uint32_t i = 0; i < elem_count; i++) {
    // find max min
    min = std::min(min, raw_datas[i]);
    max = std::max(max, raw_datas[i]);
  }

  schema::QuantParamT quant_param;
  if (!k_means) {
    STATUS status = CalQuantizationParams(&quant_param, min, max, false, quant_max, quant_min, bit_num);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "CalQuantizationParams failed" << status;
      return status;
    }
  }
  quant_params->emplace_back(quant_param);
  // update data and datatype
  for (uint32_t i = 0; i < elem_count; i++) {
    float raw_data = raw_datas[i];
    if (!k_means) {
      auto quant_data = QuantizeData<T>(raw_data, quant_param, quant_max, quant_min);
      (*quant_datas)[i] = quant_data;
    }
  }
  auto status = UpdateTensorDataAndSize(weight, quant_datas->data(), quant_datas->size() * sizeof(T));
  if (status != RET_OK) {
    MS_LOG(ERROR) << "UpdateTensorDataAndSize error";
    return RET_ERROR;
  }
  return RET_OK;
}
template <typename T>
STATUS DoBitPack(const ParamValueLitePtr &weight, const size_t &bit_num, const std::vector<T> &quant_datas) {
  if (bit_num != 8 && bit_num != 16) {
    std::vector<T> data{};
    for (size_t i = 0; i < quant_datas.size(); ++i) {
      data.emplace_back((static_cast<T>(quant_datas[i])));
    }
    if (bit_num > 0 && bit_num < 8) {
      std::vector<uint8_t> pack_data{};
      BitPack::BitPacking<T, uint8_t>(bit_num, data, &pack_data);
      auto status = UpdateTensorDataAndSize(weight, pack_data.data(), pack_data.size() * sizeof(uint8_t));
      if (status != RET_OK) {
        MS_LOG(ERROR) << "UpdateTensorDataAndSize error";
        return RET_ERROR;
      }
    } else if (bit_num > 8 && bit_num < 16) {
      std::vector<uint16_t> pack_data{};
      BitPack::BitPacking<T, uint16_t>(bit_num, data, &pack_data);
      auto status = UpdateTensorDataAndSize(weight, pack_data.data(), pack_data.size() * sizeof(uint16_t));
      if (status != RET_OK) {
        MS_LOG(ERROR) << "UpdateTensorDataAndSize error";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

template <typename T>
STATUS QuantFilter(const ParamValueLitePtr &weight, const PrimitivePtr &primitive, QuantType quant_type, int quant_max,
                   int quant_min, size_t bit_num, bool per_channel, int index = 1, bool k_means = false) {
  MS_ASSERT(weight != nullptr);
  MS_ASSERT(primitive != nullptr);
  auto dims = weight->tensor_shape();
  if (per_channel) {
    if (dims.size() <= 1) {
      MS_LOG(WARNING) << "dims is " << dims.size() << " can not per_channel";
      per_channel = false;
    }
  }

  std::vector<schema::QuantParamT> quant_params;
  size_t elem_count = weight->tensor_shape_size();
  auto *raw_data = static_cast<float *>(weight->tensor_addr());
  if (raw_data == nullptr) {
    MS_LOG(ERROR) << "rawDatas is nullptr";
    return RET_ERROR;
  }

  std::vector<T> quant_data(elem_count);
  std::vector<float> dequant_datas(elem_count);
  int ret = RET_OK;
  if (per_channel) {
    bool channel_at_first = true;
    if (primitive->name() == ops::kNameMatMul && weight->tensor_shape().size() == 2) {
      auto matmul_prim = primitive->cast<std::shared_ptr<ops::MatMul>>();
      MS_ASSERT(matmul_prim != nullptr);
      channel_at_first =
        index != 1 || (matmul_prim->GetAttr(ops::kTransposeB) != nullptr && matmul_prim->get_transpose_b());
    }
    // channel at first
    ret = DoPerChannelQuant<T>(weight, quant_type, &quant_params, quant_max, quant_min, bit_num, k_means, &quant_data,
                               &dequant_datas, channel_at_first);
    if (ret == RET_CONTINUE) {
      return ret;
    } else if (ret != RET_OK) {
      MS_LOG(ERROR) << "Do per channel quant failed.";
      return ret;
    }
  } else {
    ret = DoPerLayerQuant<T>(weight, quant_type, &quant_params, quant_max, quant_min, bit_num, k_means, &quant_data);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Do per layer quant failed.";
      return ret;
    }
  }

#ifdef HUFFMAN_ENCODE
  auto huffman_encode = std::make_unique<lite::HuffmanEncode>();
  ret = huffman_encode->DoHuffmanEncode(weight, primitive, quant_datas.data(), bit_num);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Do huffman encode failed.";
    return ret;
  }
#else
  // do bit pack
  ret = DoBitPack(weight, bit_num, quant_data);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Do bit pack failed.";
    return ret;
  }
#endif

  if (quant_params.empty()) {
    MS_LOG(ERROR) << "quant_params empty";
    return RET_ERROR;
  }
  auto quant_param_holder = GetCNodeQuantHolder(primitive);
  if (quant_type == QuantType_PostTraining) {
    quant_param_holder->AddInputQuantParam(quant_params);
  } else {
    quant_param_holder->set_input_quant_param(index, quant_params);
  }
  return ret;
}

// utils

std::string NodePrimitiveType(const CNodePtr &cnode);

STATUS ParseConfigFile(std::string config_file, PostQuantConfig *post_quant_config);

SessionModel CreateSessionByFuncGraph(const FuncGraphPtr &func_graph, const converter::Flags &flags, int thread_num);

STATUS CollectCalibInputs(const std::vector<std::string> &input_dirs, size_t count_limited,
                          std::vector<std::vector<std::string>> *inputs);

STATUS CopyInputDataToTensor(size_t input_index, size_t image_index,
                             const std::vector<std::vector<std::string>> &images, mindspore::tensor::MSTensor *tensor);

FuncGraphPtr CopyFuncGraph(const FuncGraphPtr &);

void GetLiteParameter(const AnfNodePtr &node, ParameterPtr *param_node, ParamValueLitePtr *param_value);
}  // namespace mindspore::lite::quant
#endif
