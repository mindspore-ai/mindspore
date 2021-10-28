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

#include "tools/converter/quantizer/full_quant_quantizer.h"
#include <dirent.h>
#include <future>
#include <map>
#include <set>
#include <memory>
#include <algorithm>
#include <unordered_map>
#include <functional>
#include <numeric>
#include <utility>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include "ops/fusion/full_connection.h"
#include "tools/converter/ops/ops_def.h"
#include "src/tensor.h"
#include "tools/converter/quantizer/quant_cast.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/common/format_utils.h"
#include "src/common/log_adapter.h"
#include "securec/include/securec.h"
#include "tools/common/tensor_util.h"
#include "src/common/quant_utils.h"
#include "src/common/utils.h"
#include "tools/converter/preprocess/image_preprocess.h"
#include "nnacl/op_base.h"

using std::string;
using std::vector;

namespace mindspore::lite::quant {
namespace {
static const std::set<PrimitivePtr> has_bias_operator = {prim::kPrimConv2DFusion, prim::kPrimConv2dTransposeFusion,
                                                         prim::kPrimMatMul, prim::kPrimFullConnection,
                                                         prim::kPrimLayerNormFusion};
constexpr int kMinSize = 0;
constexpr int kMaxSize = 65535;
}  // namespace
namespace {
STATUS ComputeBiasDataAndQuantParam(const std::vector<double> &bias_scales, const std::vector<double> &input_scales,
                                    const float *raw_datas, const QuantParamHolderPtr &quant_param_holder,
                                    std::vector<schema::QuantParamT> *quant_params, std::vector<int32_t> *quant_datas) {
  MS_ASSERT(raw_datas != nullptr && quant_param_holder != nullptr);
  MS_ASSERT(quant_params != nullptr && quant_datas != nullptr);
  double bias_scale_tmp;
  const constexpr double quanted_bias_abs_limit = 0.5 * INT32_MAX;
  MS_CHECK_TRUE_MSG(quant_param_holder->get_input_quant_params().size() > 1, RET_ERROR, "invalid access.");
  auto weight_quant_params = quant_param_holder->get_input_quant_params().at(1);
  auto shape_size = quant_datas->size();
  if (bias_scales.size() == shape_size) {
    for (size_t i = 0; i < shape_size; i++) {
      bias_scale_tmp = bias_scales[i];
      if (fabs(bias_scale_tmp) <= 0.0f) {
        MS_LOG(ERROR) << "divisor 'bias_scale_tmp' cannot be 0.";
        return RET_ERROR;
      }
      if (std::abs(raw_datas[i] / bias_scale_tmp) >= quanted_bias_abs_limit) {
        MS_LOG(DEBUG) << "quanted bias over flow, maybe the scale of weight: " << weight_quant_params[i].scale
                      << " is too small, need to update";
        // update filter scale and zp
        double activate_scale = input_scales[0];
        double filter_scale = std::abs(raw_datas[i]) / (activate_scale * quanted_bias_abs_limit);
        weight_quant_params[i].scale = filter_scale;
        weight_quant_params[i].zeroPoint = 0;
        quant_param_holder->set_input_quant_param(1, weight_quant_params);
        bias_scale_tmp = std::abs(raw_datas[i]) / quanted_bias_abs_limit;
        quant_params->at(i).scale = bias_scale_tmp;
        MS_LOG(DEBUG) << "new filter scale: " << filter_scale;
      }
      auto quant_data = (int32_t)std::round(raw_datas[i] / bias_scale_tmp);
      quant_datas->at(i) = quant_data;
    }
    return RET_OK;
  } else if (bias_scales.size() == 1) {
    // for fc, per tensor quant
    bias_scale_tmp = quant_params->front().scale;
    float max_raw_data = 0.0f;
    for (size_t i = 0; i < shape_size; i++) {
      if (std::abs(raw_datas[i]) > max_raw_data) {
        max_raw_data = std::abs(raw_datas[i]);
      }
    }
    if (fabs(bias_scale_tmp) <= 0.0f) {
      MS_LOG(ERROR) << "divisor 'bias_scale_tmp' cannot be 0.";
      return RET_ERROR;
    }
    if (std::abs(max_raw_data / bias_scale_tmp) >= quanted_bias_abs_limit) {
      MS_LOG(DEBUG) << "quanted bias over flow, maybe the scale of weight: " << weight_quant_params[0].scale
                    << " is too small, need to update";
      double activate_scale = input_scales[0];
      MS_CHECK_TRUE_MSG(activate_scale != 0, RET_ERROR, "activate_scale == 0");
      double filter_scale = std::abs(max_raw_data) / (activate_scale * quanted_bias_abs_limit);
      weight_quant_params[0].scale = filter_scale;
      weight_quant_params[0].zeroPoint = 0;
      quant_param_holder->set_input_quant_param(1, weight_quant_params);
      bias_scale_tmp = max_raw_data / quanted_bias_abs_limit;
      quant_params->front().scale = bias_scale_tmp;
      MS_LOG(DEBUG) << "new filter scale: " << filter_scale;
    }
    for (size_t i = 0; i < shape_size; i++) {
      auto quant_data = (int32_t)std::round(raw_datas[i] / bias_scale_tmp);
      quant_datas->at(i) = quant_data;
    }
    return RET_OK;
  }
  MS_LOG(ERROR) << "unexpected input_scales size: " << input_scales.size()
                << " weight_scales size: " << weight_quant_params.size();
  return RET_ERROR;
}
}  // namespace

STATUS DivergInfo::RecordMaxMinValue(const std::vector<float> &data) {
  for (float val : data) {
    max = std::max(val, max);
    min = std::min(val, min);
  }
  return RET_OK;
}

STATUS DivergInfo::RecordMaxMinValueArray(const std::vector<float> &data) {
  if (data.empty()) {
    return RET_ERROR;
  }
  float max_num = data.at(0);
  float min_num = data.at(0);
  for (float val : data) {
    max_num = std::max(val, max_num);
    min_num = std::min(val, min_num);
  }
  this->max_datas.emplace_back(max_num);
  this->min_datas.emplace_back(min_num);
  return RET_OK;
}

void DivergInfo::UpdateInterval() {
  auto max_value = std::max(fabs(this->max), fabs(this->min));
  MS_ASSERT(bin_num != 0);
  this->interval = max_value / static_cast<float>(bin_num);
}

STATUS DivergInfo::UpdateHistogram(const std::vector<float> &data) {
  for (auto value : data) {
    if (value == 0) {
      continue;
    }
    if (this->interval == 0) {
      MS_LOG(ERROR) << "divisor 'interval' cannot be 0.";
      return RET_ERROR;
    }
    int bin_index = std::min(static_cast<int>(std::fabs(value) / this->interval), bin_num - 1);
    this->histogram[bin_index]++;
  }
  return RET_OK;
}

void DivergInfo::DumpHistogram() {
  MS_LOG(INFO) << "Print node " << cnode->fullname_with_scope() << " histogram";
  for (float item : this->histogram) {
    std::cout << item << " ";
  }
  std::cout << std::endl;
}

void DivergInfo::HandleBinForKL(int quant_bint_nums, int bin_index, std::vector<float> *quantized_histogram,
                                std::vector<float> *expanded_histogram) {
  MS_ASSERT(quantized_histogram != nullptr && expanded_histogram != nullptr);
  MS_ASSERT(quant_bint_nums != 0);
  const float bin_interval = static_cast<float>(bin_index) / static_cast<float>(quant_bint_nums);
  // merge i bins to target bins
  for (int i = 0; i < quant_bint_nums; ++i) {
    const float start = i * bin_interval;
    const float end = start + bin_interval;
    const int left_upper = static_cast<int>(std::ceil(start));
    if (left_upper > start) {
      const double left_scale = left_upper - start;
      quantized_histogram->at(i) += left_scale * this->histogram[left_upper - 1];
    }
    const int right_lower = static_cast<int>(std::floor(end));
    if (right_lower < end) {
      const double right_scale = end - right_lower;
      quantized_histogram->at(i) += right_scale * this->histogram[right_lower];
    }
    std::for_each(this->histogram.begin() + left_upper, this->histogram.begin() + right_lower,
                  [&quantized_histogram, i](float item) { quantized_histogram->at(i) += item; });
  }
  // expand target bins to i bins in order to calculate KL with reference_histogram
  for (int i = 0; i < quant_bint_nums; ++i) {
    const float start = i * bin_interval;
    const float end = start + bin_interval;
    float count = 0;
    const int left_upper = static_cast<int>(std::ceil(start));
    float left_scale = 0.0f;
    if (left_upper > start) {
      left_scale = left_upper - start;
      if (this->histogram[left_upper - 1] != 0) {
        count += left_scale;
      }
    }
    const int right_lower = static_cast<int>(std::floor(end));
    double right_scale = 0.0f;
    if (right_lower < end) {
      right_scale = end - right_lower;
      if (this->histogram[right_lower] != 0) {
        count += right_scale;
      }
    }
    std::for_each(this->histogram.begin() + left_upper, this->histogram.begin() + right_lower, [&count](float item) {
      if (item != 0) {
        count += 1;
      }
    });
    if (count == 0) {
      continue;
    }
    const float average_num = quantized_histogram->at(i) / count;
    if (left_upper > start && this->histogram[left_upper - 1] != 0) {
      expanded_histogram->at(left_upper - 1) += average_num * left_scale;
    }
    if (right_lower < end && this->histogram[right_lower] != 0) {
      expanded_histogram->at(right_lower) += average_num * right_scale;
    }
    for (int k = left_upper; k < right_lower; ++k) {
      if (this->histogram[k] != 0) {
        expanded_histogram->at(k) += average_num;
      }
    }
  }
}

STATUS DivergInfo::ComputeThreshold() {
  if (activation_quant_method == MAX_MIN) {
    this->best_T = std::max(fabs(this->max), fabs(this->min));
    MS_LOG(DEBUG) << "using MAX_MIN, T: " << this->best_T;
    return RET_OK;
  }

  if (activation_quant_method == REMOVAL_OUTLIER && !this->min_datas.empty()) {
    this->percent_result = OutlierMethod(min_datas, max_datas);
    this->best_T = std::max(std::fabs(percent_result.first), std::fabs(percent_result.second));
    return RET_OK;
  }

  int threshold = INT8_MAX + 1;
  float min_kl = FLT_MAX;
  float after_threshold_sum = std::accumulate(this->histogram.begin() + INT8_MAX + 1, this->histogram.end(), 0.0f);

  for (int i = INT8_MAX + 1; i < this->bin_num; ++i) {
    std::vector<float> quantized_histogram(INT8_MAX + 1, 0);
    std::vector<float> reference_histogram(this->histogram.begin(), this->histogram.begin() + i);
    std::vector<float> expanded_histogram(i, 0);
    reference_histogram[i - 1] += after_threshold_sum;
    after_threshold_sum -= this->histogram[i];
    // handle bins for computing KL.
    HandleBinForKL(INT8_MAX + 1, i, &quantized_histogram, &expanded_histogram);
    auto KLDivergence = [](std::vector<float> p, std::vector<float> q) {
      auto sum = 0.0f;
      std::for_each(p.begin(), p.end(), [&sum](float item) { sum += item; });
      std::for_each(p.begin(), p.end(), [sum](float &item) { item /= sum; });
      sum = 0.0f;
      std::for_each(q.begin(), q.end(), [&sum](float item) { sum += item; });
      std::for_each(q.begin(), q.end(), [sum](float &item) { item /= sum; });

      float result = 0.0f;
      const int size = p.size();
      for (int i = 0; i < size; ++i) {
        if (p[i] != 0) {
          if (q[i] == 0) {
            result += 1.0f;
          } else {
            result += (p[i] * std::log((p[i]) / (q[i])));
          }
        }
      }
      return result;
    };
    const float kl = KLDivergence(reference_histogram, expanded_histogram);
    if (kl < min_kl) {
      min_kl = kl;
      threshold = i;
    }
  }
  this->best_T = (static_cast<float>(threshold) + 0.5f) * this->interval;
  MS_LOG(DEBUG) << cnode->fullname_with_scope() << " Best threshold bin index: " << threshold << " T: " << best_T
                << " max: " << std::max(fabs(this->max), fabs(this->min));
  return RET_OK;
}

std::pair<CNodePtr, float> DivergInfo::GetScale() {
  float max_value = this->best_T;
  float min_value = -max_value;

  if (this->activation_quant_method == REMOVAL_OUTLIER) {
    min_value = percent_result.first;
    max_value = percent_result.second;
  }

  MS_CHECK_TRUE_MSG(quant_max - quant_min != 0, {}, "quant_max - quant_min == 0");
  float scale = (max_value - min_value) / (quant_max - quant_min);
  this->scale_tmp = scale;
  MS_ASSERT(fabs(scale) <= 0.0f);
  return std::make_pair(this->cnode, scale);
}

std::pair<CNodePtr, int32_t> DivergInfo::GetZeropoint() {
  int zero_point = 0;
  if (quant_min == 0 && quant_max == UINT8_MAX) {
    zero_point = INT8_MAX + 1;
  } else if (quant_min == INT_LEAST8_MIN + 1 && quant_max == INT8_MAX) {
    zero_point = 0;
  } else {
    MS_LOG(WARNING) << "unexpected quant range, quant_min: " << quant_min << " quant_max: " << quant_max;
  }
  if (this->activation_quant_method == REMOVAL_OUTLIER) {
    MS_CHECK_TRUE_MSG(fabs(scale_tmp) <= 0.0f, {}, "fabs(scale_tmp) > 0.0f");
    zero_point = std::round(quant_max - percent_result.second / scale_tmp);
  }
  return std::make_pair(this->cnode, zero_point);
}

std::unordered_map<std::string, std::vector<std::unique_ptr<DivergInfo>>> *Calibrator::GetInputDivergInfo() {
  return &this->inputs_diverg_info_;
}

std::unordered_map<std::string, std::vector<std::unique_ptr<DivergInfo>>> *Calibrator::GetOutputDivergInfo() {
  return &this->outputs_diverg_info_;
}

STATUS Calibrator::RecordMaxMinValue(const vector<float> &data, const std::unique_ptr<DivergInfo> &diverg_info) {
  auto ret = diverg_info->RecordMaxMinValue(data);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Record max min value failed.";
    return ret;
  }
  ret = diverg_info->RecordMaxMinValueArray(data);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Record max min value array failed.";
    return ret;
  }
  return RET_OK;
}

STATUS Calibrator::ComputeThreshold() {
  for (auto &kv : this->outputs_diverg_info_) {
    auto &outputs_diverg_info = kv.second;
    for (auto &diverg_info : outputs_diverg_info) {
      auto ret = diverg_info->ComputeThreshold();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Compute threshold failed.";
        return ret;
      }
    }
  }
  // node A's input may be node B's output, no need to re-compute the node A's input quant param which is the same as
  for (auto &kv : this->inputs_diverg_info_) {
    auto &input_infos = kv.second;
    for (size_t i = 0; i < input_infos.size(); i++) {
      auto cnode = input_infos[i]->cnode;
      bool already_computed = false;
      auto input = cnode->input(i + 1);
      if (input->isa<mindspore::CNode>()) {
        auto input_cnode = input->cast<CNodePtr>();
        for (const auto &outputs_diverg_info : outputs_diverg_info_) {
          if (already_computed) {
            break;
          }
          for (const auto &output_diverg_info : outputs_diverg_info.second) {
            auto output_diverg_cnode = output_diverg_info->cnode;
            if (output_diverg_cnode == input_cnode) {
              if (NodePrimitiveType(input_cnode) != lite::kNameTupleGetItem) {
                *(input_infos[i]) = *output_diverg_info;
                input_infos[i]->cnode = cnode;
                already_computed = true;
                break;
              }
            }
          }
        }
      }
      if (!already_computed) {
        auto ret = input_infos[i]->ComputeThreshold();
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "ComputeThreshold failed.";
          return ret;
        }
      }
    }
  }
  return RET_OK;
}

STATUS Calibrator::UpdateDivergInterval(
  std::unordered_map<std::string, std::vector<std::unique_ptr<DivergInfo>>> *diverg_info) {
  MS_ASSERT(diverg_info != nullptr);
  for (auto &kv : *diverg_info) {
    for (auto &info : kv.second) {
      info->UpdateInterval();
    }
  }
  return RET_OK;
}

STATUS Calibrator::UpdateDataFrequency(const vector<float> &data, const std::unique_ptr<DivergInfo> &diverg_info) {
  MS_ASSERT(diverg_info != nullptr);
  return diverg_info->UpdateHistogram(data);
}

STATUS Calibrator::AddQuantizedOp(const CNodePtr &cnode) {
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "To be quantized cnode is null";
    return RET_ERROR;
  }
  string node_name = cnode->fullname_with_scope();
  std::unique_ptr<DivergInfo> input_diverg = std::make_unique<DivergInfo>(
    cnode, kDefaultBinNumber, bit_num_, quant_max_, quant_min_, full_quant_param_.activation_quant_method);
  MS_CHECK_TRUE_MSG(input_diverg != nullptr, RET_NULL_PTR, "input_diverg is nullptr.");
  std::unique_ptr<DivergInfo> output_diverg = std::make_unique<DivergInfo>(
    cnode, kDefaultBinNumber, bit_num_, quant_max_, quant_min_, full_quant_param_.activation_quant_method);
  MS_CHECK_TRUE_MSG(output_diverg != nullptr, RET_NULL_PTR, "output_diverg is nullptr.");
  inputs_diverg_info_[node_name].push_back(std::move(input_diverg));
  outputs_diverg_info_[node_name].push_back(std::move(output_diverg));
  return RET_OK;
}

STATUS Calibrator::GenerateInputData(const std::string &input_name, size_t image_index,
                                     mindspore::tensor::MSTensor *tensor) const {
  return preprocess::PreProcess(data_pre_process_param_, input_name, image_index, tensor);
}

FullQuantQuantizer::FullQuantQuantizer(FuncGraphPtr graph, int bit_num, TypeId target_type, bool per_channel)
    : Quantizer(std::move(graph)) {
  MS_ASSERT(graph != nullptr);
  this->per_channel_ = per_channel;
  this->bit_num = bit_num;
  this->target_type_ = target_type;
  if (target_type == kNumberTypeInt8) {
    quant_max = (1 << (this->bit_num - 1)) - 1;  // 127
    quant_min = -quant_max;                      // -127
  } else if (target_type == kNumberTypeUInt8) {
    quant_max = (1 << this->bit_num) - 1;  // 255
    quant_min = 0;
  } else {
    MS_LOG(ERROR) << "unsupported quant value type: " << target_type;
  }
  calibrator_ = std::make_unique<Calibrator>(this->bit_num, quant_max, quant_min);
  if (calibrator_ == nullptr) {
    MS_LOG(ERROR) << "create calibrator failed!";
    return;
  }
}

FullQuantQuantizer::~FullQuantQuantizer() {
  delete fp32_session_;
  delete fp32_model_;
  delete int8_session_;
  delete int8_model_;
}

STATUS FullQuantQuantizer::SetInOutQuantParam(const AnfNodePtr &input_node, const std::unique_ptr<DivergInfo> &info,
                                              const PrimitivePtr &primitive, bool is_input, size_t index) const {
  auto quant_param_holder = GetCNodeQuantHolder(primitive);
  MS_CHECK_TRUE_MSG(quant_param_holder != nullptr, RET_NULL_PTR, "quant_param_holder is nullptr.");
  schema::QuantParamT quant_param;
  TypeId type_id = kTypeUnknown;
  if (opt::GetDataTypeFromAnfNode(input_node, &type_id) != RET_OK) {
    MS_LOG(ERROR) << "Get data type failed.";
    return RET_ERROR;
  }
  if (type_id == kNumberTypeFloat32 && info != nullptr) {
    auto scale = info->GetScale().second;
    if (scale == 0) {
      MS_LOG(WARNING) << "The input or output values are very close to 0, so set the scale to 1.";
      quant_param.scale = 1;
    } else {
      quant_param.scale = scale;
    }
    quant_param.zeroPoint = info->GetZeropoint().second;
    quant_param.max = info->max;
    quant_param.min = info->min;
    quant_param.numBits = bit_num;
    quant_param.narrowRange = false;
    quant_param.inited = true;
    quant_param.roundType = 1;
    quant_param.multiplier = 1;
  } else {
    quant_param.inited = false;
  }
  std::vector<schema::QuantParamT> quant_params = {quant_param};
  if (is_input) {
    quant_param_holder->set_input_quant_param(index, quant_params);
  } else {
    quant_param_holder->set_output_quant_param(index, quant_params);
  }
  return RET_OK;
}

STATUS FullQuantQuantizer::DoWeightQuant(const std::string &op_name, const AnfNodePtr &weight,
                                         const PrimitivePtr &primitive, bool per_channel, int input_index) const {
  MS_ASSERT(weight != nullptr);
  MS_ASSERT(primitive != nullptr);
  // perlayer
  if (!weight->isa<Parameter>()) {
    MS_LOG(ERROR) << "not a parameter";
    return RET_PARAM_INVALID;
  }
  auto parameter = weight->cast<ParameterPtr>();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << weight->fullname_with_scope() << " can not cast to Parameter";
    return RET_NULL_PTR;
  }
  auto tensor_info = parameter->default_param()->cast<tensor::TensorPtr>();
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << weight->fullname_with_scope() << " can not get value";
    return RET_NULL_PTR;
  }
  auto bit_num_t = bit_num;
  auto quant_max_t = quant_max;
  auto quant_min_t = quant_min;
  auto weight_quant_type = per_channel ? WeightQuantType::FIXED_BIT_PER_CHANNEL : WeightQuantType::FIXED_BIT_PER_LAYER;
  auto status = FixedBitQuantFilter<int8_t>(tensor_info, primitive, QuantType_QUANT_ALL, quant_max_t, quant_min_t,
                                            bit_num_t, weight_quant_type, kNumberTypeInt8, input_index - 1);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "QuantFilter failed: " << status;
    return status;
  }
  // set dtype
  auto abstractBase = parameter->abstract();
  if (abstractBase == nullptr) {
    MS_LOG(ERROR) << "Abstract of parameter is nullptr, " << parameter->name();
    return RET_NULL_PTR;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstractBase)) {
    MS_LOG(ERROR) << "Abstract of parameter should be anstract tensor, " << parameter->name();
    return RET_ERROR;
  }
  auto abstractTensor = utils::cast<abstract::AbstractTensorPtr>(abstractBase);
  if (abstractTensor == nullptr || abstractTensor->element() == nullptr) {
    MS_LOG(ERROR) << "abstractTensor is nullptr, " << parameter->name();
    return RET_NULL_PTR;
  }
  abstractTensor->element()->set_type(TypeIdToType(kNumberTypeInt8));
  return RET_OK;
}

STATUS FullQuantQuantizer::DoBiasQuant(const AnfNodePtr &bias, const PrimitivePtr &primitive) {
  if (primitive == nullptr || bias == nullptr) {
    MS_LOG(ERROR) << "null pointer!";
    return RET_NULL_PTR;
  }
  auto bias_parameter_ptr = bias->cast<ParameterPtr>();
  MS_ASSERT(bias_parameter_ptr != nullptr);
  auto bias_default_param = bias_parameter_ptr->default_param();
  auto bias_param = bias_default_param->cast<tensor::TensorPtr>();
  MS_ASSERT(bias_parameter != nullptr);
  auto quant_param_holder = GetCNodeQuantHolder(primitive);
  MS_CHECK_TRUE_MSG(quant_param_holder != nullptr, RET_NULL_PTR, "quant_param_holder is nullptr.");
  auto active_weight_quant_params = quant_param_holder->get_input_quant_params();

  auto active_params = active_weight_quant_params.at(FIRST_INPUT);
  auto weight_params = active_weight_quant_params.at(SECOND_INPUT);

  vector<double> input_scales;
  vector<double> filter_scales;
  vector<double> bias_scales;
  size_t sizeX = active_params.size();
  for (size_t i = 0; i < sizeX; i++) {
    input_scales.emplace_back(active_params[i].scale);
  }
  size_t sizeY = weight_params.size();
  if (sizeX != sizeY) {
    if (sizeX > 1 && sizeY > 1) {
      MS_LOG(ERROR) << "input and filter's scale count cannot match!";
      return RET_ERROR;
    }
  }
  for (size_t i = 0; i < sizeY; i++) {
    filter_scales.emplace_back(weight_params[i].scale);
  }
  size_t size = std::max(sizeX, sizeY);
  for (size_t i = 0; i < size; i++) {
    auto scaleX = sizeX > 1 ? input_scales[i] : input_scales[0];
    auto scaleY = sizeY > 1 ? filter_scales[i] : filter_scales[0];
    bias_scales.push_back(scaleX * scaleY);
  }
  MS_ASSERT(!bias_scales.empty());
  size_t shape_size = bias_param->DataSize();

  // set bias quant param
  std::vector<schema::QuantParamT> quant_params;
  for (double bias_scale : bias_scales) {
    schema::QuantParamT quant_param;
    if (bias_scale == 0) {
      MS_LOG(WARNING) << "bias_scale == 0";
      quant_param.scale = 1;
    } else {
      quant_param.scale = bias_scale;
    }
    quant_param.zeroPoint = 0;
    quant_param.inited = true;
    quant_params.emplace_back(quant_param);
  }
  // quant bias data
  std::vector<int32_t> quant_datas(shape_size);

  auto *raw_datas = static_cast<float *>(bias_param->data_c());
  if (ComputeBiasDataAndQuantParam(bias_scales, input_scales, raw_datas, quant_param_holder, &quant_params,
                                   &quant_datas) != RET_OK) {
    MS_LOG(ERROR) << "compute bias data failed.";
    return RET_ERROR;
  }
  quant_param_holder->set_input_quant_param(THIRD_INPUT, quant_params);
  auto ret = SetTensorData(bias_param, quant_datas.data(), shape_size * sizeof(int32_t));
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "set tensor data failed.";
    return RET_ERROR;
  }
  // set dtype
  auto abstractBase = bias_parameter_ptr->abstract();
  if (abstractBase == nullptr) {
    MS_LOG(ERROR) << "Abstract of parameter is nullptr, " << bias_parameter_ptr->name();
    return RET_ERROR;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstractBase)) {
    MS_LOG(ERROR) << "Abstract of parameter should be anstract tensor, " << bias_parameter_ptr->name();
    return RET_ERROR;
  }
  auto abstractTensor = utils::cast<abstract::AbstractTensorPtr>(abstractBase);
  if (abstractTensor == nullptr || abstractTensor->element() == nullptr) {
    MS_LOG(ERROR) << "abstractTensor is nullptr" << bias_parameter_ptr->name();
    return RET_NULL_PTR;
  }
  abstractTensor->element()->set_type(TypeIdToType(kNumberTypeInt32));
  return RET_OK;
}

STATUS FullQuantQuantizer::DoParameterNodeQuant(const CNodePtr &cnode, const AnfNodePtr &input_node,
                                                size_t input_index) {
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (primitive == nullptr) {
    return RET_ERROR;
  }
  auto op_name = cnode->fullname_with_scope();
  STATUS ret;
  TypeId type_id = kTypeUnknown;
  if (opt::GetDataTypeFromAnfNode(input_node, &type_id) != RET_OK) {
    MS_LOG(ERROR) << "Get data type failed.";
    return RET_ERROR;
  }
  // support for share weight.
  if (type_id == kNumberTypeInt8) {
    return RET_CONTINUE;
  }
  if (type_id != kNumberTypeFloat32) {
    ret = SetInOutQuantParam(input_node, nullptr, primitive, true, input_index - 1);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Set In/Out quant param failed.";
      return ret;
    }
    return RET_QUANT_CONTINUE;
  }
  if (CheckNodeInSet(cnode, has_bias_operator)) {
    if (input_index == FOURTH_INPUT) {
      ret = DoBiasQuant(input_node, primitive);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Do bias quant failed.";
        return ret;
      }
    } else {
      if (opt::CheckPrimitiveType(cnode, prim::kPrimMatMul)) {
        ret = DoWeightQuant(op_name, input_node, primitive, false, input_index);
      } else {
        ret = DoWeightQuant(op_name, input_node, primitive, true, input_index);
      }
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Do bias quant failed.";
        return ret;
      }
    }
  } else {
    ret = DoWeightQuant(op_name, input_node, primitive, false, input_index);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Do bias quant failed.";
      return ret;
    }
  }
  return RET_OK;
}

STATUS FullQuantQuantizer::QuantNodeSimpleOp(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  auto inputs_diverg_info = calibrator_->GetInputDivergInfo();
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (primitive == nullptr) {
    return RET_ERROR;
  }
  auto op_name = cnode->fullname_with_scope();
  auto primitive_quant_holder = GetCNodeQuantHolder(primitive);
  MS_CHECK_TRUE_MSG(primitive_quant_holder != nullptr, RET_NULL_PTR, "primitive_quant_holder is nullptr.");
  size_t activation_input_index = 0;
  STATUS ret;
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    auto input_node = cnode->input(i);
    MS_ASSERT(input_node != nullptr);
    bool is_graph_input = false;
    if (input_node->isa<Parameter>()) {
      if (!input_node->cast<ParameterPtr>()->has_default()) {
        is_graph_input = true;
      }
    }
    if (is_graph_input) {
      // do input quant
      auto &info = (*inputs_diverg_info)[op_name][activation_input_index++];
      ret = SetInOutQuantParam(input_node, info, primitive, true, i - 1);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Set activation quant failed.";
        return ret;
      }
    } else if (input_node->isa<mindspore::CNode>()) {
      auto input_cnode = input_node->cast<mindspore::CNodePtr>();
      auto input_cnode_primitive = GetValueNode<PrimitivePtr>(input_cnode->input(0));
      if (input_cnode_primitive == nullptr) {
        MS_LOG(DEBUG) << "input: " << i << " " << input_cnode->fullname_with_scope() << ": "
                      << " Primitive is null";
        continue;
      }
      auto input_primitive_quant_holder = GetCNodeQuantHolder(input_cnode_primitive);
      MS_CHECK_TRUE_MSG(input_primitive_quant_holder != nullptr, RET_NULL_PTR,
                        "input_primitive_quant_holder is nullptr.");
      if (input_primitive_quant_holder->IsOutputQuantParamsInited()) {
        auto quant_param = input_primitive_quant_holder->get_output_quant_params().front();
        primitive_quant_holder->set_input_quant_param(i - 1, quant_param);
        activation_input_index++;
      } else {
        // do input quant
        auto &info = (*inputs_diverg_info)[op_name][activation_input_index++];
        ret = SetInOutQuantParam(input_node, info, primitive, true, i - 1);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "Set activation quant failed.";
          return ret;
        }
      }
    } else if (input_node->isa<mindspore::Parameter>()) {
      ret = DoParameterNodeQuant(cnode, input_node, i);
      if (ret == RET_QUANT_CONTINUE) {
        continue;
      } else if (ret != RET_OK) {
        MS_LOG(ERROR) << "Do parameter node quant failed.";
        return ret;
      }
    } else {
      MS_LOG(ERROR) << input_node->fullname_with_scope() << ":" << input_node->type_name() << " is not support type";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS FullQuantQuantizer::QuantNode() {
  auto inputs_diverg_info = calibrator_->GetInputDivergInfo();
  auto outputs_diverg_info = calibrator_->GetOutputDivergInfo();

  auto cnodes = funcGraph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    auto op_name = cnode->fullname_with_scope();
    auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (primitive == nullptr) {
      MS_LOG(ERROR) << "primitive is nullptr";
      continue;
    }
    auto primitive_quant_holder = GetCNodeQuantHolder(primitive);
    MS_CHECK_TRUE_MSG(primitive_quant_holder != nullptr, RET_NULL_PTR, "primitive_quant_holder is nullptr.");
    if (inputs_diverg_info->find(op_name) == inputs_diverg_info->end()) {
      MS_LOG(INFO) << op_name << " can not do quant";
      primitive_quant_holder->set_quant_type(schema::QuantType_QUANT_NONE);
      continue;
    }

    auto op_type = primitive->name();
    MS_LOG(DEBUG) << "OpName: " << op_name;
    if (op_type == lite::kNameTupleGetItem) {
      constexpr int tuple_get_item_input_size = 3;
      MS_CHECK_TRUE_MSG(cnode->size() == tuple_get_item_input_size, RET_ERROR, "cnode->size() != 3");
      auto index_node = cnode->input(THIRD_INPUT);
      auto index_value_node = index_node->cast<mindspore::ValueNodePtr>();
      if (index_value_node == nullptr) {
        MS_LOG(WARNING) << "index value node is null";
        continue;
      }
      size_t index = opt::CastToInt(index_value_node->value()).front();
      auto input_node = cnode->input(SECOND_INPUT);
      MS_CHECK_TRUE_MSG(input_node != nullptr, RET_ERROR, "input_node == nullptr");
      auto input_cnode = input_node->cast<mindspore::CNodePtr>();
      MS_CHECK_TRUE_MSG(input_cnode != nullptr, RET_ERROR, "input_cnode == nullptr");
      auto input_cnode_primitive = GetValueNode<PrimitivePtr>(input_cnode->input(0));
      if (input_cnode_primitive == nullptr) {
        MS_LOG(WARNING) << "input_cnode_primitive is null";
        continue;
      }
      auto input_primitive_quant_holder = GetCNodeQuantHolder(input_cnode_primitive);
      MS_CHECK_TRUE_MSG(input_primitive_quant_holder != nullptr, RET_NULL_PTR,
                        "input_primitive_quant_holder is nullptr.");

      if (input_primitive_quant_holder->get_output_quant_params().size() > index) {
        auto quant_param = input_primitive_quant_holder->get_output_quant_params()[index];
        primitive_quant_holder->set_input_quant_param(0, quant_param);
        primitive_quant_holder->set_output_quant_param(0, quant_param);
      } else {
        MS_LOG(WARNING) << "this TupleGetItem node's input node: " << input_cnode->fullname_with_scope()
                        << "'s output quant_params size: "
                        << input_primitive_quant_holder->get_output_quant_params().size() << ", but index: " << index;
      }
      primitive_quant_holder->set_quant_type(schema::QuantType_QUANT_ALL);
      continue;
    } else {  // do simple op quant
      auto status = QuantNodeSimpleOp(cnode);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "simple op quant failed.";
        return status;
      }
    }
    // do output quant, there may multi-output
    auto &infos = (*outputs_diverg_info)[op_name];
    for (size_t index = 0; index < infos.size(); index++) {
      auto &info = infos.at(index);
      auto ret = SetInOutQuantParam(cnode, info, primitive, false, index);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Set In/Out quant param failed.";
        return ret;
      }
      primitive_quant_holder->set_quant_type(schema::QuantType_QUANT_ALL);
    }
  }
  return RET_OK;
}

STATUS FullQuantQuantizer::UpdateDivergeInterval() {
  auto ret = this->calibrator_->UpdateDivergInterval(this->calibrator_->GetInputDivergInfo());
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Update input diverge interval failed.";
    return ret;
  }
  ret = this->calibrator_->UpdateDivergInterval(this->calibrator_->GetOutputDivergInfo());
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Update output diverge interval failed.";
    return ret;
  }
  return RET_OK;
}

/**
 *  Mark quantifiable nodes
 **/
STATUS FullQuantQuantizer::PreProcess() {
  auto cnodes = funcGraph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    AnfNodePtr anf = cnode->cast<AnfNodePtr>();
    if (anf == nullptr) {
      MS_LOG(ERROR) << " cnode is null";
      return RET_NULL_PTR;
    }
    if (mindspore::lite::quant::QuantStrategy::CanOpFullQuantized(anf)) {
      auto ret = calibrator_->AddQuantizedOp(cnode);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Add Quantized Op failed.";
        return ret;
      }
    }
    auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (primitive == nullptr) {
      MS_LOG(ERROR) << cnode->fullname_with_scope() << " primitive is null";
      continue;
    }
    auto quant_param_holder = GetCNodeQuantHolder(primitive);
    MS_CHECK_TRUE_MSG(quant_param_holder != nullptr, RET_NULL_PTR, "quant_param_holder is nullptr.");
    quant_param_holder->ClearInputOutputQuantParam();
  }
  return RET_OK;
}

STATUS FullQuantQuantizer::CheckFp32TensorVec(const std::string &node_name,
                                              const std::vector<mindspore::tensor::MSTensor *> &tensor_vec) {
  MS_ASSERT(tensor_vec != nullptr);
  if (tensor_vec.empty()) {
    MS_LOG(ERROR) << "node: " << node_name << " input tensors is 0";
    return RET_ERROR;
  }
  auto *tensor = tensor_vec[0];
  MS_ASSERT(tensor != nullptr);
  if (tensor->data_type() != kNumberTypeFloat32) {
    MS_LOG(INFO) << "node: " << node_name << " will not quantize"
                 << " tensor data_type: " << tensor->data_type();
    return RET_ERROR;
  }
  return RET_OK;
}

/**
 * 1. create input tensor
 * 2. insert callback to session
 * 3. run session
 **/
STATUS FullQuantQuantizer::DoInference() {
  // get input tensor
  vector<mindspore::tensor::MSTensor *> inputs = fp32_session_->GetInputs();
  if (inputs.size() != calibrator_->GetInputNum()) {
    MS_LOG(ERROR) << "model's input tensor count: " << inputs.size() << " != "
                  << " calibrator count:" << calibrator_->GetInputNum();
    return RET_ERROR;
  }

  for (size_t calib_index = 0; calib_index < calibrator_->GetBatchNum(); calib_index++) {
    // set multi-input data
    for (size_t input_index = 0; input_index < inputs.size(); input_index++) {
      STATUS status =
        calibrator_->GenerateInputData(inputs[input_index]->tensor_name(), calib_index, inputs[input_index]);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "generate input data from images failed!";
        return RET_ERROR;
      }
    }

    KernelCallBack beforeCallBack = [&](const std::vector<mindspore::tensor::MSTensor *> &beforeInputs,
                                        const std::vector<mindspore::tensor::MSTensor *> &beforeOutputs,
                                        const CallBackParam &callParam) -> bool {
      auto diverg_info_map = calibrator_->GetInputDivergInfo();
      if (diverg_info_map->find(callParam.node_name) == diverg_info_map->end()) {
        return true;
      }
      if (FullQuantQuantizer::CheckFp32TensorVec(callParam.node_name, beforeOutputs) != RET_OK) {
        return true;
      }
      bool is_init = beforeInputs.size() > 1 && (*diverg_info_map)[callParam.node_name].size() == 1;
      if (is_init) {
        for (size_t i = 1; i < beforeInputs.size(); i++) {
          if (beforeInputs.at(i)->data_type() != kNumberTypeFloat32 || beforeInputs.at(i)->IsConst()) {
            continue;
          }
          auto input_diverg = std::make_unique<DivergInfo>();
          MS_CHECK_TRUE_MSG(input_diverg != nullptr, false, "input_diverg is nullptr.");
          *input_diverg = *((*diverg_info_map)[callParam.node_name][0]);
          (*diverg_info_map)[callParam.node_name].push_back(std::move(input_diverg));
        }
      }
      for (size_t i = 0; i < (*diverg_info_map)[callParam.node_name].size(); i++) {
        auto tensor = beforeInputs[i];
        MS_ASSERT(tensor != nullptr);
        const auto *tensor_data = static_cast<const float *>(tensor->MutableData());
        MS_CHECK_TRUE_MSG(tensor_data != nullptr, false, "tensor_data is nullptr.");
        size_t elem_count = tensor->ElementsNum();
        vector<float> data(tensor_data, tensor_data + elem_count);
        auto ret = this->calibrator_->RecordMaxMinValue(data, (*diverg_info_map)[callParam.node_name][i]);
        MS_CHECK_TRUE_MSG(ret == RET_OK, false, "Record MaxMinValue failed!");
      }
      return true;
    };
    // func
    KernelCallBack afterCallBack = [&](const std::vector<mindspore::tensor::MSTensor *> &afterInputs,
                                       const std::vector<mindspore::tensor::MSTensor *> &afterOutputs,
                                       const CallBackParam &callParam) -> bool {
      auto diverg_info_map = calibrator_->GetOutputDivergInfo();
      if (diverg_info_map->find(callParam.node_name) == diverg_info_map->end()) {
        return true;
      }
      if (FullQuantQuantizer::CheckFp32TensorVec(callParam.node_name, afterOutputs) != RET_OK) {
        return true;
      }
      bool is_init = afterOutputs.size() > 1 && (*diverg_info_map)[callParam.node_name].size() == 1;
      if (is_init) {
        for (size_t i = 1; i < afterOutputs.size(); i++) {
          auto output_diverg = std::make_unique<DivergInfo>();
          CHECK_NULL_RETURN(output_diverg);
          *output_diverg = *((*diverg_info_map)[callParam.node_name][0]);
          (*diverg_info_map)[callParam.node_name].push_back(std::move(output_diverg));
        }
      }
      size_t output_i = 0;
      for (const auto &tensor : afterOutputs) {
        const auto *tensor_data = static_cast<const float *>(tensor->MutableData());
        CHECK_NULL_RETURN(tensor_data);
        size_t elem_count = tensor->ElementsNum();
        vector<float> data(tensor_data, tensor_data + elem_count);
        auto ret = this->calibrator_->RecordMaxMinValue(data, (*diverg_info_map)[callParam.node_name][output_i]);
        MS_CHECK_TRUE_MSG(ret == RET_OK, false, "Record MaxMinValue failed!");
        output_i++;
      }
      return true;
    };
    auto status = fp32_session_->RunGraph(beforeCallBack, afterCallBack);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "run model failed!";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS FullQuantQuantizer::Int8Inference() {
  // int8 inference
  vector<mindspore::tensor::MSTensor *> inputs = int8_session_->GetInputs();
  for (auto input_tensor : inputs) {
    // get input tensor
    auto elem_count = input_tensor->ElementsNum();
    vector<float> dummy_data(elem_count);
    // set the input data to 0.1
    std::fill(dummy_data.begin(), dummy_data.end(), 0.1);
    auto ret =
      memcpy_s(input_tensor->MutableData(), input_tensor->Size(), dummy_data.data(), sizeof(float) * dummy_data.size());
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy_s error: " << ret;
      return RET_ERROR;
    }
  }

  for (size_t i = 0; i < calibrator_->GetBatchNum(); i++) {
    // before func
    KernelCallBack before_call_back = GetBeforeCallBack(true);
    // after func
    KernelCallBack after_call_back = GetAfterCallBack(true);
    auto ret = int8_session_->RunGraph(before_call_back, after_call_back);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "run model failed!";
      return RET_ERROR;
    }
  }  // end for images
  return RET_OK;
}

STATUS FullQuantQuantizer::BiasCorrection(const FuncGraphPtr &func_graph) {
  std::future<STATUS> int8_inference = std::async(std::launch::async, &FullQuantQuantizer::Int8Inference, this);
  // get input tensor
  vector<mindspore::tensor::MSTensor *> inputs = fp32_session_->GetInputs();
  if (inputs.size() != 1) {
    MS_LOG(ERROR) << "model's input tensor size: " << inputs.size();
    return RET_ERROR;
  }
  // fp32 inference
  for (size_t i = 0; i < calibrator_->GetBatchNum(); i++) {
    for (size_t input_index = 0; input_index < inputs.size(); input_index++) {
      STATUS status = calibrator_->GenerateInputData(inputs[input_index]->tensor_name(), i, inputs[input_index]);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "generate input data from images failed!";
        return RET_ERROR;
      }
    }
    // before func
    KernelCallBack before_call_back = GetBeforeCallBack(false);
    // after func
    KernelCallBack after_call_back = GetAfterCallBack(false);
    auto status = fp32_session_->RunGraph(before_call_back, after_call_back);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "run model failed!";
      return RET_ERROR;
    }
  }  // end for images

  STATUS status = int8_inference.get();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "int8 inference failed!";
    return RET_ERROR;
  }
  if (calibrator_->GetBatchNum() == 0) {
    MS_LOG(ERROR) << "divisor 'calibrate_size' cannot be 0.";
    return RET_ERROR;
  }
  for (auto &key_value : op_bias_diff_map) {
    std::for_each(key_value.second.begin(), key_value.second.end(),
                  [this](float &data) { data = data / calibrator_->GetBatchNum(); });
  }
  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    auto op_name = cnode->fullname_with_scope();
    if (op_bias_diff_map.find(op_name) == op_bias_diff_map.end()) {
      continue;
    }
    status = BiasCorrection(func_graph, cnode);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "do node bias correct failed.";
      break;
    }
  }
  return status;
}

STATUS FullQuantQuantizer::BiasCorrection(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  auto op_name = cnode->fullname_with_scope();
  const auto &bias_diff = op_bias_diff_map[op_name];
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return RET_NULL_PTR;
  }
  auto quant_param_holder = GetCNodeQuantHolder(primitive);
  MS_CHECK_TRUE_MSG(quant_param_holder != nullptr, RET_NULL_PTR, "quant_param_holder is nullptr.");
  auto input_quant_params = quant_param_holder->get_input_quant_params();
  if (input_quant_params.size() == DIMENSION_3D) {
    // compensate the existed
    auto bias_quant_params = input_quant_params.at(THIRD_INPUT);
    auto bias = cnode->input(FOURTH_INPUT);
    auto bias_parameter_ptr = bias->cast<ParameterPtr>();
    auto bias_default_param = bias_parameter_ptr->default_param();
    auto bias_param = bias_default_param->cast<tensor::TensorPtr>();
    int *bias_datas = static_cast<int *>(bias_param->data_c());

    if (static_cast<size_t>(bias_param->DataSize()) != bias_diff.size()) {
      MS_LOG(DEBUG) << "unexpected bias data count: " << bias_param->DataSize()
                    << " not the same as bias_diff: " << bias_diff.size();
      return RET_ERROR;
    }
    if (bias_quant_params.size() != bias_diff.size()) {
      MS_LOG(ERROR) << "unexpected bias quant params size: " << bias_quant_params.size()
                    << " not the same as bias_diff: " << bias_diff.size();
      return RET_ERROR;
    }
    for (int i = 0; i < bias_param->DataSize(); i++) {
      auto scale = bias_quant_params[i].scale;
      if (fabs(scale) <= 0.0f) {
        MS_LOG(ERROR) << "divisor 'scale' cannot be 0.";
        return RET_ERROR;
      }
      double after_correct = std::round(bias_diff[i] / scale) + bias_datas[i];
      const constexpr int32_t corrected_bias_abs_limit = 0.6 * INT32_MAX;
      if (after_correct > corrected_bias_abs_limit) {
        MS_LOG(WARNING) << op_name << " ch: " << i << " bias after_corrected too large: " << after_correct
                        << " origin value: " << bias_datas[i] << " bias_diff: " << bias_diff[i] << " scale: " << scale;
        bias_datas[i] = static_cast<int>(corrected_bias_abs_limit);
      } else if (after_correct < -corrected_bias_abs_limit) {
        MS_LOG(WARNING) << op_name << " ch: " << i << " bias after_corrected too small: " << after_correct
                        << " origin value: " << bias_datas[i] << " bias_diff: " << bias_diff[i] << " scale: " << scale;
        bias_datas[i] = static_cast<int>(-corrected_bias_abs_limit);
      } else {
        auto diff = static_cast<int>(std::round(bias_diff[i] / scale));
        bias_datas[i] += diff;
      }
    }
  } else if (input_quant_params.size() == DIMENSION_2D) {
    MS_LOG(INFO) << op_name << " add bias input";
    // need to add bias input
    auto parameter = func_graph->add_parameter();
    if (parameter == nullptr) {
      MS_LOG(ERROR) << "parameter is nullptr.";
      return RET_NULL_PTR;
    }
    ShapeVector shape;
    shape.push_back(bias_diff.size());

    auto tensor_info = CreateTensorInfo(bias_diff.data(), sizeof(float) * bias_diff.size(), shape, kNumberTypeFloat32);
    if (tensor_info == nullptr) {
      MS_LOG(ERROR) << "create tensor info failed.";
      return RET_ERROR;
    }
    auto status = InitParameterFromTensorInfo(parameter, tensor_info);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "init parameter from tensor info failed";
      return RET_ERROR;
    }
    parameter->set_name("added_" + op_name + "_bias");
    cnode->add_input(parameter);
    status = DoBiasQuant(parameter, primitive);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Do bias quant failed.";
      return RET_ERROR;
    }
  } else {
    MS_LOG(ERROR) << "unexpected get_input_quant_params size: " << input_quant_params.size();
  }
  return RET_OK;
}

STATUS FullQuantQuantizer::CollectDataFrequency() {
  // get input tensor
  vector<mindspore::tensor::MSTensor *> inputs = fp32_session_->GetInputs();
  if (inputs.size() != calibrator_->GetInputNum()) {
    MS_LOG(ERROR) << "model's input tensor cnt: " << inputs.size() << " != " << calibrator_->GetInputNum();
    return RET_ERROR;
  }

  for (size_t i = 0; i < calibrator_->GetBatchNum(); i++) {
    // set multi-input data
    for (size_t input_index = 0; input_index < inputs.size(); input_index++) {
      STATUS status = calibrator_->GenerateInputData(inputs[input_index]->tensor_name(), i, inputs[input_index]);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "generate input data from images failed!";
        return RET_ERROR;
      }
    }

    KernelCallBack before_callback = [&](const std::vector<mindspore::tensor::MSTensor *> &before_inputs,
                                         const std::vector<mindspore::tensor::MSTensor *> &before_outputs,
                                         const CallBackParam &callParam) {
      auto diverg_info_map = calibrator_->GetInputDivergInfo();
      if (diverg_info_map->find(callParam.node_name) == diverg_info_map->end()) {
        return true;
      }
      if (FullQuantQuantizer::CheckFp32TensorVec(callParam.node_name, before_inputs) != RET_OK) {
        return true;
      }
      int input_i = 0;
      for (auto tensor : before_inputs) {
        if (tensor->data_type() != kNumberTypeFloat32 || tensor->IsConst()) {
          continue;
        }
        const auto *tensor_data = static_cast<const float *>(tensor->MutableData());
        MS_ASSERT(tensor_data != nullptr);
        size_t elem_count = tensor->ElementsNum();
        vector<float> data(tensor_data, tensor_data + elem_count);
        auto ret = this->calibrator_->UpdateDataFrequency(data, (*diverg_info_map)[callParam.node_name][input_i++]);
        if (ret != RET_OK) {
          return false;
        }
      }
      return true;
    };

    KernelCallBack after_callBack = [&](const std::vector<mindspore::tensor::MSTensor *> &after_inputs,
                                        const std::vector<mindspore::tensor::MSTensor *> &after_outputs,
                                        const CallBackParam &call_param) {
      auto diverg_info_map = calibrator_->GetOutputDivergInfo();
      if (diverg_info_map->find(call_param.node_name) == diverg_info_map->end()) {
        return true;
      }
      if (FullQuantQuantizer::CheckFp32TensorVec(call_param.node_name, after_outputs) != RET_OK) {
        return true;
      }
      int output_i = 0;
      // all outputs are same dtype.
      for (const auto &tensor : after_outputs) {
        const auto *tensor_data = static_cast<const float *>(tensor->MutableData());
        MS_ASSERT(tensor_data != nullptr);
        size_t elem_count = tensor->ElementsNum();
        vector<float> data(tensor_data, tensor_data + elem_count);
        auto ret = this->calibrator_->UpdateDataFrequency(data, (*diverg_info_map)[call_param.node_name][output_i++]);
        if (ret != RET_OK) {
          return false;
        }
      }
      return true;
    };
    auto status = fp32_session_->RunGraph(before_callback, after_callBack);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "run model failed!";
      return RET_ERROR;
    }
  }

  return RET_OK;
}

STATUS FullQuantQuantizer::ComputeThreshold() { return this->calibrator_->ComputeThreshold(); }

STATUS FullQuantQuantizer::DoQuantize(FuncGraphPtr func_graph) {
  MS_LOG(INFO) << "start to parse config file";
  if (this->calibrator_ == nullptr) {
    MS_LOG(ERROR) << "calibrator is null!";
    return RET_ERROR;
  }
  calibrator_->full_quant_param_ = flags.fullQuantParam;
  calibrator_->data_pre_process_param_ = flags.dataPreProcessParam;
  if (flags.dataPreProcessParam.calibrate_path.empty()) {
    MS_LOG(ERROR) << "calibrate path must pass. The format is input_name_1:input_1_dir,input_name_2:input_2_dir.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (flags.dataPreProcessParam.calibrate_size <= kMinSize || flags.dataPreProcessParam.calibrate_size > kMaxSize) {
    MS_LOG(ERROR) << "calibrate size must pass and the size should in [1, 65535].";
    return RET_INPUT_PARAM_INVALID;
  }
  if (flags.dataPreProcessParam.input_type == preprocess::INPUT_TYPE_MAX) {
    MS_LOG(ERROR) << "input_type must pass IMAGE | BIN.";
    return RET_INPUT_PARAM_INVALID;
  }
  STATUS status = PreProcess();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "do pre process failed!";
    return status;
  }

  // anf -- fb
  flags.commonQuantParam.quant_type = schema::QuantType_QUANT_NONE;
  MS_LOG(INFO) << "start create session";
  auto sm = CreateSessionByFuncGraph(func_graph, flags, calibrator_->GetThreadNum());
  fp32_session_ = sm.session;
  fp32_model_ = sm.model;
  if (fp32_session_ == nullptr || fp32_model_ == nullptr) {
    MS_LOG(ERROR) << "create session failed!";
    return RET_ERROR;
  }
  MS_LOG(INFO) << "start to update divergence's max value";
  status = DoInference();
  if (status != RET_OK) {
    return status;
  }
  MS_LOG(INFO) << "start to update divergence's interval";
  status = UpdateDivergeInterval();
  if (status != RET_OK) {
    return status;
  }
  MS_LOG(INFO) << "start to collect data's distribution";
  status = CollectDataFrequency();
  if (status != RET_OK) {
    return status;
  }
  MS_LOG(INFO) << "compute the best threshold";
  status = ComputeThreshold();
  if (status != RET_OK) {
    return status;
  }
  MS_LOG(INFO) << "start to generate quant param and quantize tensor's data";
  status = QuantNode();
  if (status != RET_OK) {
    return status;
  }

  // add quant_cast
  quant::QuantCast quant_cast;
  status = quant_cast.Run(func_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "add QuantCast error";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return RET_ERROR;
  }

  if (calibrator_->GetBiasCorrection()) {
    // init in8 session
    MS_LOG(INFO) << "create quant session";
    flags.commonQuantParam.quant_type = schema::QuantType_QUANT_ALL;
    auto int8_sm = CreateSessionByFuncGraph(func_graph, flags, calibrator_->GetThreadNum());
    int8_session_ = int8_sm.session;
    int8_model_ = int8_sm.model;
    if (int8_session_ == nullptr || int8_model_ == nullptr) {
      MS_LOG(ERROR) << "create session failed!";
      return RET_ERROR;
    }
    MS_LOG(INFO) << "do bias correction";
    status = BiasCorrection(func_graph);
    if (status != RET_OK) {
      MS_LOG(WARNING) << "BiasCorrection failed.";
    }
  }
  return RET_OK;
}

bool FullQuantQuantizer::OpInputDataHandle(OperationType type, const string &op_name, std::vector<float> *data) {
  MS_ASSERT(data != nullptr);
  std::lock_guard<std::mutex> lg(mutex_op_input);
  if (type == STORE) {
    if (fp32_op_input_map.find(op_name) != fp32_op_input_map.end()) {
      // the data has not been fetched by int8 model
      return false;
    }
    fp32_op_input_map[op_name] = *data;
    return true;
  } else if (type == FETCH) {
    if (fp32_op_input_map.find(op_name) == fp32_op_input_map.end()) {
      // the data not generated by fp32 model yet
      return false;
    }
    *data = fp32_op_input_map[op_name];
    fp32_op_input_map.erase(op_name);
    return true;
  } else {
    MS_LOG(ERROR) << "unexpected type: " << type;
  }
  return false;
}

bool FullQuantQuantizer::OpOutputChMeanDataHandle(OperationType type, const string &op_name, std::vector<float> *data) {
  MS_ASSERT(data != nullptr);
  std::lock_guard<std::mutex> lg(mutex_op_output);
  if (type == STORE) {
    if (fp32_op_output_ch_mean_map.find(op_name) != fp32_op_output_ch_mean_map.end()) {
      // the data has not been fetched by int8 model
      return false;
    }
    fp32_op_output_ch_mean_map[op_name] = *data;
    return true;
  } else if (type == FETCH) {
    if (fp32_op_output_ch_mean_map.find(op_name) == fp32_op_output_ch_mean_map.end()) {
      // the data not generated by fp32 model yet
      return false;
    }
    *data = fp32_op_output_ch_mean_map[op_name];
    fp32_op_output_ch_mean_map.erase(op_name);
    return true;
  } else {
    MS_LOG(ERROR) << "unexpected type: " << type;
  }
  return false;
}

KernelCallBack FullQuantQuantizer::GetBeforeCallBack(bool int8_op) {
  KernelCallBack before_call_back;
  if (!int8_op) {
    before_call_back = [this](const std::vector<mindspore::tensor::MSTensor *> &before_inputs,
                              const std::vector<mindspore::tensor::MSTensor *> &before_outputs,
                              const CallBackParam &callParam) -> bool {
      if (callParam.node_type == kTypeConv2D || callParam.node_type == kTypeDepthwiseConv2D) {
        if (FullQuantQuantizer::CheckFp32TensorVec(callParam.node_name, before_inputs) != RET_OK) {
          return true;
        }
        auto tensor = before_inputs[0];
        MS_ASSERT(tensor != nullptr);
        size_t elem_count = tensor->ElementsNum();
        std::vector<float> fp32_op_input(elem_count);
        auto ret =
          memcpy_s(fp32_op_input.data(), fp32_op_input.size() * sizeof(float), tensor->MutableData(), tensor->Size());
        if (ret != EOK) {
          MS_LOG(ERROR) << "memcpy error: " << ret;
          return false;
        }
        while (!OpInputDataHandle(STORE, callParam.node_name, &fp32_op_input)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(kMillisecondsBase));
        }
      }
      return true;
    };
  } else {
    before_call_back = [this](const std::vector<mindspore::tensor::MSTensor *> &before_inputs,
                              const std::vector<mindspore::tensor::MSTensor *> &before_outputs,
                              const CallBackParam &callParam) -> bool {
      if (callParam.node_type == kTypeConv2D || callParam.node_type == kTypeDepthwiseConv2D) {
        vector<float> fp32_op_input;
        while (!OpInputDataHandle(FETCH, callParam.node_name, &fp32_op_input)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(kMillisecondsBase));
        }
        auto tensor = before_inputs[0];
        MS_ASSERT(tensor != nullptr);
        if (tensor->data_type() != kNumberTypeInt8) {
          MS_LOG(ERROR) << "unexpected tensor type: " << tensor->data_type();
          return false;
        }
        // do quantization: activation is always per layer quantized
        std::vector<int8_t> quant_datas;
        auto quant_params = tensor->quant_params();
        if (quant_params.size() != 1) {
          MS_LOG(ERROR) << "unexpected quant_params size: " << quant_params.size();
          return false;
        }
        schema::QuantParamT quant_param_t;
        quant_param_t.scale = quant_params[0].scale;
        quant_param_t.zeroPoint = quant_params[0].zeroPoint;
        for (auto float_data : fp32_op_input) {
          auto quant_data = QuantizeData<int8_t>(float_data, &quant_param_t, quant_max, quant_min);
          quant_datas.push_back(quant_data);
        }

        if (tensor->Size() != quant_datas.size() * sizeof(int8_t)) {
          MS_LOG(ERROR) << "unexpected tensor size: " << quant_datas.size()
                        << " not the same with: " << quant_datas.size() * sizeof(int8_t);
          return false;
        }

        auto ret =
          memcpy_s(tensor->MutableData(), tensor->Size(), quant_datas.data(), quant_datas.size() * sizeof(int8_t));
        if (ret != EOK) {
          MS_LOG(ERROR) << "memcpy error: " << ret;
          return false;
        }
      }
      return true;
    };
  }
  return before_call_back;
}

KernelCallBack FullQuantQuantizer::GetAfterCallBack(bool int8_op) {
  KernelCallBack after_call_back;
  if (!int8_op) {
    return GetFloatAfterCallBack();
  }
  return GetInt8AfterCallBack();
}

KernelCallBack FullQuantQuantizer::GetInt8AfterCallBack() {
  KernelCallBack after_call_back = [this](const std::vector<mindspore::tensor::MSTensor *> &afterInputs,
                                          const std::vector<mindspore::tensor::MSTensor *> &afterOutputs,
                                          const CallBackParam &callParam) -> bool {
    if (callParam.node_type == kTypeConv2D || callParam.node_type == kTypeDepthwiseConv2D) {
      vector<float> fp32_op_output_ch_mean;
      while (!OpOutputChMeanDataHandle(FETCH, callParam.node_name, &fp32_op_output_ch_mean)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(kMillisecondsBase));
      }
      auto tensor = afterOutputs[0];
      MS_ASSERT(tensor != nullptr);
      if (tensor->data_type() != kNumberTypeInt8) {
        MS_LOG(ERROR) << "unexpected tensor type: " << tensor->data_type();
        return false;
      }
      const int8_t *tensor_data = static_cast<int8_t *>(tensor->MutableData());
      size_t elem_count = tensor->ElementsNum();
      auto shapes = tensor->shape();
      if (shapes.size() != 4) {
        MS_LOG(ERROR) << "unexpected shape size: " << shapes.size();
        return false;
      }
      // suppose the the format is NHWC
      auto channels = shapes[3];
      if (channels == 0) {
        MS_LOG(ERROR) << "unexpected channels: 0";
        return false;
      }
      auto quant_params = tensor->quant_params();
      if (quant_params.size() != 1) {
        MS_LOG(ERROR) << "unexpected activatation quant_params size: " << quant_params.size();
        return false;
      }
      auto scale = quant_params[0].scale;
      auto zp = quant_params[0].zeroPoint;
      std::vector<float> dequant_op_output_ch_mean(channels);
      auto one_filter_size = elem_count / channels;
      for (int i = 0; i < channels; i++) {
        float sum = 0;
        for (size_t j = 0; j < one_filter_size; j++) {
          auto index = j * channels + i;
          if (index >= elem_count) {
            MS_LOG(ERROR) << "over flow!";
            return false;
          }
          // deuqant activation
          auto float_data = scale * (tensor_data[index] - zp);
          sum += float_data;
        }
        if (one_filter_size == 0) {
          MS_LOG(ERROR) << "divisor 'one_filter_size' cannot be 0.";
          return false;
        }
        sum = sum / one_filter_size;
        dequant_op_output_ch_mean[i] = sum;
      }
      std::transform(fp32_op_output_ch_mean.begin(), fp32_op_output_ch_mean.end(), dequant_op_output_ch_mean.begin(),
                     dequant_op_output_ch_mean.begin(), std::minus<>());

      if (op_bias_diff_map.find(callParam.node_name) != op_bias_diff_map.end()) {
        auto &bias_diff = op_bias_diff_map[callParam.node_name];
        std::transform(bias_diff.begin(), bias_diff.end(), dequant_op_output_ch_mean.begin(), bias_diff.begin(),
                       std::plus<>());
      } else {
        op_bias_diff_map[callParam.node_name] = dequant_op_output_ch_mean;
      }
    }
    return true;
  };
  return after_call_back;
}

KernelCallBack FullQuantQuantizer::GetFloatAfterCallBack() {
  KernelCallBack after_call_back = [this](const std::vector<mindspore::tensor::MSTensor *> &afterInputs,
                                          const std::vector<mindspore::tensor::MSTensor *> &afterOutputs,
                                          const CallBackParam &callParam) -> bool {
    if (callParam.node_type == kTypeConv2D || callParam.node_type == kTypeDepthwiseConv2D) {
      if (FullQuantQuantizer::CheckFp32TensorVec(callParam.node_name, afterOutputs) != RET_OK) {
        return true;
      }
      auto tensor = afterOutputs[0];
      MS_ASSERT(tensor != nullptr);
      const auto *tensor_data = static_cast<const float *>(tensor->MutableData());
      size_t elem_count = tensor->ElementsNum();
      auto shapes = tensor->shape();
      if (shapes.size() != 4) {
        MS_LOG(ERROR) << "unexpected shape size: " << shapes.size();
        return false;
      }
      // suppose the activation format: NHWC
      auto channels = shapes[3];
      if (channels == 0) {
        MS_LOG(ERROR) << "unexpected channels: 0";
        return false;
      }
      std::vector<float> fp32_op_output_ch_mean(channels);
      auto one_filter_size = elem_count / channels;
      for (int i = 0; i < channels; i++) {
        float sum = 0;
        for (size_t j = 0; j < one_filter_size; j++) {
          auto index = j * channels + i;
          if (index >= elem_count) {
            MS_LOG(ERROR) << "over flow!";
            return false;
          }
          sum += tensor_data[index];
        }
        if (one_filter_size == 0) {
          MS_LOG(ERROR) << "divisor 'one_filter_size' cannot be 0.";
          return false;
        }
        sum = sum / one_filter_size;
        fp32_op_output_ch_mean[i] = sum;
      }
      while (!OpOutputChMeanDataHandle(STORE, callParam.node_name, &fp32_op_output_ch_mean)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(kMillisecondsBase));
      }
    }
    return true;
  };
  return after_call_back;
}
}  // namespace mindspore::lite::quant
