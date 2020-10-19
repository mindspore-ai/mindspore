/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "tools/converter/quantizer/post_training_quantizer.h"
#include <dirent.h>
#include <sys/stat.h>
#include <future>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <utility>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include "schema/inner/model_generated.h"
#include "src/tensor.h"
#include "tools/anf_exporter/anf_exporter.h"
#include "tools/converter/quantizer/quant_cast.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "src/common/log_adapter.h"
#include "securec/include/securec.h"
#include "tools/common/tensor_util.h"
#include "src/common/file_utils.h"
#include "src/common/utils.h"

using std::string;
using std::vector;

namespace mindspore {
namespace lite {
namespace quant {
STATUS DivergInfo::RecordMaxValue(const std::vector<float> &datas) {
  for (float data : datas) {
    max = std::max(data, max);
    min = std::min(data, min);
  }
  return RET_OK;
}

STATUS DivergInfo::RecordMaxValueArray(const std::vector<float> &datas) {
  if (datas.size() == 0) {
    return RET_ERROR;
  }
  float max_num = datas.at(0);
  float min_num = datas.at(0);
  for (float data : datas) {
    max_num = std::max(data, max_num);
    min_num = std::min(data, min_num);
  }
  this->max_datas.emplace_back(max_num);
  this->min_datas.emplace_back(min_num);
  return RET_OK;
}

void DivergInfo::UpdateInterval() {
  auto max_value = std::max(fabs(this->max), fabs(this->min));
  this->interval = max_value / static_cast<float>(bin_num);
}

STATUS DivergInfo::UpdateHistogram(const std::vector<float> &data) {
  for (auto value : data) {
    if (value == 0) {
      continue;
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

STATUS DivergInfo::ComputeThreshold() {
  if (method_x == kMethodMaxMin) {
    this->best_T = std::max(fabs(this->max), fabs(this->min));
    MS_LOG(DEBUG) << "using MAX_MIN, T: " << this->best_T;
    return RET_OK;
  }

  if (method_x == kMethodOutlier) {
    this->percent_result = PercentMethod(min_datas, max_datas);
    this->best_T = std::max(std::fabs(percent_result.first), std::fabs(percent_result.second));
    return RET_OK;
  }

  constexpr int quant_bint_nums = 128;
  int threshold = quant_bint_nums;
  float min_kl = FLT_MAX;
  float after_threshold_sum = std::accumulate(this->histogram.begin() + quant_bint_nums, this->histogram.end(), 0.0f);

  for (int i = quant_bint_nums; i < this->bin_num; ++i) {
    std::vector<float> quantized_histogram(quant_bint_nums, 0);
    std::vector<float> reference_histogram(this->histogram.begin(), this->histogram.begin() + i);
    std::vector<float> expanded_histogram(i, 0);
    reference_histogram[i - 1] += after_threshold_sum;
    after_threshold_sum -= this->histogram[i];

    const float bin_interval = static_cast<float>(i) / static_cast<float>(quant_bint_nums);

    // merge i bins to target bins
    for (int j = 0; j < quant_bint_nums; ++j) {
      const float start = j * bin_interval;
      const float end = start + bin_interval;
      const int left_upper = static_cast<int>(std::ceil(start));
      if (left_upper > start) {
        const double left_scale = left_upper - start;
        quantized_histogram[j] += left_scale * this->histogram[left_upper - 1];
      }
      const int right_lower = static_cast<int>(std::floor(end));
      if (right_lower < end) {
        const double right_scale = end - right_lower;
        quantized_histogram[j] += right_scale * this->histogram[right_lower];
      }
      std::for_each(this->histogram.begin() + left_upper, this->histogram.begin() + right_lower,
                    [&quantized_histogram, j](float item) { quantized_histogram[j] += item; });
    }
    // expand target bins to i bins in order to calculate KL with reference_histogram
    for (int j = 0; j < quant_bint_nums; ++j) {
      const float start = j * bin_interval;
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
      const float average_num = quantized_histogram[j] / count;
      if (left_upper > start && this->histogram[left_upper - 1] != 0) {
        expanded_histogram[left_upper - 1] += average_num * left_scale;
      }
      if (right_lower < end && this->histogram[right_lower] != 0) {
        expanded_histogram[right_lower] += average_num * right_scale;
      }
      for (int k = left_upper; k < right_lower; ++k) {
        if (this->histogram[k] != 0) {
          expanded_histogram[k] += average_num;
        }
      }
    }
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

  if (this->method_x == kMethodOutlier) {
    min_value = percent_result.first;
    max_value = percent_result.second;
  }

  MS_ASSERT(quant_max - quant_min != 0);
  float scale = (max_value - min_value) / (quant_max - quant_min);
  this->scale_tmp = scale;
  MS_ASSERT(scale != 0);
  return std::make_pair(this->cnode, scale);
}

std::pair<CNodePtr, int32_t> DivergInfo::GetZeropoint() {
  int zero_point = 0;
  if (quant_min == 0 && quant_max == 255) {
    zero_point = 128;
  } else if (quant_min == -127 && quant_max == 127) {
    zero_point = 0;
  } else {
    MS_LOG(WARNING) << "unexpectd quant range, quant_min: " << quant_min << " quant_max: " << quant_max;
  }

  if (this->method_x == kMethodOutlier) {
    zero_point = std::round(quant_max - percent_result.second / scale_tmp);
  }
  return std::make_pair(this->cnode, zero_point);
}

std::unordered_map<CNodePtr, float> Calibrator::GetScale(
  std::unordered_map<std::string, std::unique_ptr<DivergInfo>> *diverg_info) {
  std::unordered_map<CNodePtr, float> result;
  for (auto iter = diverg_info->begin(); iter != diverg_info->end(); iter++) {
    DivergInfo *info = iter->second.get();
    auto item = info->GetScale();
    result.insert(item);
  }
  return result;
}
std::unordered_map<CNodePtr, int32_t> Calibrator::GetZeropoint(
  std::unordered_map<std::string, std::unique_ptr<DivergInfo>> *diverg_info) {
  std::unordered_map<CNodePtr, int32_t> result;
  for (auto iter = diverg_info->begin(); iter != diverg_info->end(); iter++) {
    DivergInfo *info = iter->second.get();
    auto zeropoint = info->GetZeropoint();
    result.insert(zeropoint);
  }
  return result;
}

std::map<CNodePtr, MaxMin> Calibrator::GetMinMax(
  std::unordered_map<std::string, std::unique_ptr<DivergInfo>> *diverg_info) {
  std::map<CNodePtr, MaxMin> result;
  for (auto iter = diverg_info->begin(); iter != diverg_info->end(); iter++) {
    DivergInfo *info = iter->second.get();
    mindspore::lite::quant::MaxMin input_maxmin{};
    input_maxmin.min = info->min;
    input_maxmin.max = info->max;
    result[info->cnode] = input_maxmin;
  }
  return result;
}

void Calibrator::Dump() {
  for (auto iter = this->input_diverg_info_.begin(); iter != this->input_diverg_info_.end(); iter++) {
    DivergInfo *info = iter->second.get();
    info->DumpHistogram();
  }
}

std::unordered_map<std::string, std::unique_ptr<DivergInfo>> *Calibrator::GetInputDivergInfo() {
  return &this->input_diverg_info_;
}

std::unordered_map<std::string, std::unique_ptr<DivergInfo>> *Calibrator::GetOutputDivergInfo() {
  return &this->output_diverg_info_;
}

STATUS Calibrator::RecordMaxValue(const std::string &op_name, const vector<float> &data,
                                  std::unordered_map<std::string, std::unique_ptr<DivergInfo>> *diverg_info) {
  auto got = (*diverg_info).find(op_name);
  if (got != (*diverg_info).end()) {
    ((*got).second)->RecordMaxValue(data);
    ((*got).second)->RecordMaxValueArray(data);
  }
  return RET_OK;
}

STATUS Calibrator::ComputeThreshold() {
  for (auto iter = this->output_diverg_info_.begin(); iter != this->output_diverg_info_.end(); iter++) {
    DivergInfo *info = iter->second.get();
    info->ComputeThreshold();
  }
  // node A's input may be node B's output, no need to re-compute the node A's input quant param which is the same as
  for (auto iter = this->input_diverg_info_.begin(); iter != this->input_diverg_info_.end(); iter++) {
    DivergInfo *info = iter->second.get();
    auto cnode = info->cnode;

    bool already_computed = false;
    auto input = cnode->input(1);
    if (input->isa<mindspore::CNode>()) {
      auto input_cnode = std::dynamic_pointer_cast<mindspore::CNode>(input);
      for (const auto &output_diverg_info : output_diverg_info_) {
        auto output_diverg_cnode = output_diverg_info.second->cnode;
        if (output_diverg_cnode == input_cnode) {
          *info = *(output_diverg_info.second);
          info->cnode = cnode;
          already_computed = true;
          break;
        }
      }
    }
    if (!already_computed) {
      info->ComputeThreshold();
    }
  }
  return RET_OK;
}

STATUS Calibrator::UpdateDivergInverval(std::unordered_map<std::string, std::unique_ptr<DivergInfo>> *diverg_info) {
  for (auto iter = (*diverg_info).begin(); iter != (*diverg_info).end(); iter++) {
    DivergInfo *info = iter->second.get();
    info->UpdateInterval();
  }
  return RET_OK;
}

STATUS Calibrator::UpdateDataFrequency(const std::string &op_name, const vector<float> &data,
                                       std::unordered_map<std::string, std::unique_ptr<DivergInfo>> *diverg_info) {
  auto got = (*diverg_info).find(op_name);
  if (got != (*diverg_info).end()) {
    ((*got).second)->UpdateHistogram(data);
  }
  return RET_OK;
}

STATUS Calibrator::AddQuantizedOp(CNodePtr node) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "To be quantized node is null";
    return RET_ERROR;
  }
  string node_name = node->fullname_with_scope();
  std::unique_ptr<DivergInfo> input_diverg = std::unique_ptr<DivergInfo>(
    new DivergInfo(node, kDefaultBinNumber, bit_num_, quant_max_, quant_min_, config_param_.method_x));
  std::unique_ptr<DivergInfo> output_diverg = std::unique_ptr<DivergInfo>(
    new DivergInfo(node, kDefaultBinNumber, bit_num_, quant_max_, quant_min_, config_param_.method_x));

  input_diverg_info_.insert(std::make_pair(string(node_name), std::move(input_diverg)));
  output_diverg_info_.insert(std::make_pair(string(node_name), std::move(output_diverg)));
  return RET_OK;
}

void Calibrator::AddImage(const string file) {
  auto exist = [](const string file) {
    struct stat buf {};
    return stat(file.c_str(), &buf) == 0;
  };
  if (exist(file)) {
    MS_LOG(INFO) << "load image: " << file;
    this->images_.push_back(file);
  } else {
    MS_LOG(WARNING) << "invalid image file path: " << file;
  }
}

STATUS Calibrator::GenerateInputData(int index, mindspore::tensor::MSTensor *tensor) const {
  string path = images_[index];
  MS_LOG(INFO) << "read image: " << path;
  size_t size;
  char *bin_buf = ReadFile(path.c_str(), &size);
  auto data = tensor->MutableData();
  if (data == nullptr) {
    MS_LOG(ERROR) << "Get tensor MutableData return nullptr";
    return RET_ERROR;
  }
  if (size != tensor->Size()) {
    MS_LOG(ERROR) << "the input data is not consistent with model input, file_size: " << size
                  << " input tensor size: " << tensor->Size();
    return RET_ERROR;
  }
  auto ret = memcpy_s(data, tensor->Size(), bin_buf, size);
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy_s error: " << ret;
    return RET_ERROR;
  }
  delete[] bin_buf;
  return RET_OK;
}

STATUS Calibrator::CollectImages() {
  // check image file path
  DIR *root = opendir(config_param_.image_path.c_str());
  if (root == nullptr) {
    MS_LOG(ERROR) << "invalid image path: " << config_param_.image_path;
    return RET_PARAM_INVALID;
  }
  struct dirent *image_dir = readdir(root);
  size_t count = 0;
  while (image_dir != nullptr) {
    if (image_dir->d_name[0] != '.') {
      const std::string file_name = config_param_.image_path + "/" + image_dir->d_name;
      if (config_param_.batch_count == 0) {
        this->AddImage(file_name);
        count++;
      } else if (count < config_param_.batch_count) {
        this->AddImage(file_name);
        count++;
      } else {
        break;
      }
    }
    image_dir = readdir(root);
  }
  closedir(root);
  return RET_OK;
}

STATUS Calibrator::ReadConfig() {
  if (config_path_.empty() || config_path_.length() > PATH_MAX) {
    MS_LOG(ERROR) << "invalid config path!";
    return RET_PARAM_INVALID;
  }
  // check whether config file path is valid
  char *resolved_path = new (std::nothrow) char[PATH_MAX]{0};
  if (resolved_path == nullptr) {
    MS_LOG(ERROR) << "New an object failed.";
    return RET_ERROR;
  }
#ifdef _WIN32
  if (_fullpath(resolved_path, config_path_.c_str(), 1024) != nullptr) {
    config_path_ = string(resolved_path);
  }
#else
  if (realpath(config_path_.c_str(), resolved_path) != nullptr) {
    config_path_ = string(resolved_path);
  }
#endif
  std::ifstream fs(config_path_.c_str(), std::ifstream::in);
  if (!fs.is_open()) {
    MS_LOG(ERROR) << "config proto file %s open failed: " << config_path_;
    delete[] resolved_path;
    return RET_PARAM_INVALID;
  }
  std::string line;
  while (std::getline(fs, line)) {
    auto index = line.find('=');
    if (index == std::string::npos) {
      MS_LOG(ERROR) << "the config file is invalid, can not find '=', please check";
      delete[] resolved_path;
      return RET_PARAM_INVALID;
    }
    auto key = line.substr(0, index);
    auto value = line.substr(index + 1);
    Trim(&key);
    Trim(&value);
    if (key == "image_path") {
      config_param_.image_path = value;
    } else if (key == "batch_count") {
      config_param_.batch_count = std::stoul(value);
    } else if (key == "thread_num") {
      config_param_.thread_num = std::stoul(value);
    } else if (key == "method_x") {
      if (value != kMethodKL && value != kMethodMaxMin && value != kMethodOutlier) {
        MS_LOG(WARNING) << "unsupported method_x: " << value << ". Use default value.";
      } else {
        config_param_.method_x = value;
      }
    } else if (key == "bias_correction") {
      std::for_each(value.begin(), value.end(), ::tolower);
      if (value == "true") {
        config_param_.bias_correction = true;
      }
    } else {
      MS_LOG(WARNING) << "unsupported parameter";
    }
  }
  MS_LOG(DEBUG) << "image_path: " << config_param_.image_path << "  "
                << "batch_count: " << config_param_.batch_count << "  "
                << "method_x: " << config_param_.method_x << "  "
                << "thread_num: " << config_param_.thread_num << " "
                << "bias_correction: " << config_param_.bias_correction;

  delete[] resolved_path;
  fs.close();
  return RET_OK;
}

Calibrator::Calibrator(string path, size_t bit_num, int quant_max, int quant_min)
    : config_path_(path), bit_num_(bit_num), quant_max_(quant_max), quant_min_(quant_min) {}

PostTrainingQuantizer::PostTrainingQuantizer(FuncGraphPtr graph, string path, int bit_num, TypeId target_type,
                                             bool per_channel)
    : Quantizer(graph) {
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
  calibrator_ = std::unique_ptr<Calibrator>(new Calibrator(path, this->bit_num, quant_max, quant_min));
  if (calibrator_ == nullptr) {
    MS_LOG(ERROR) << "creat calibrator failed!";
    return;
  }
}

STATUS PostTrainingQuantizer::DoQuantInput(double scale, int zeropoint, struct MaxMin *max_min,
                                           std::shared_ptr<PrimitiveC> lite_primitive) {
  if (!lite_primitive->GetInputQuantParams().empty()) {
    return RET_OK;
  }
  schema::QuantParamT quant_param;
  quant_param.scale = scale;
  quant_param.zeroPoint = zeropoint;
  quant_param.max = max_min->max;
  quant_param.min = max_min->min;
  quant_param.numBits = bit_num;
  quant_param.narrowRange = false;
  std::vector<schema::QuantParamT> quant_params = {quant_param};
  lite_primitive->AddInputQuantParam(quant_params);
  return RET_OK;
}

STATUS PostTrainingQuantizer::DoQuantOutput(double scale, int zeropoint, struct MaxMin *max_min,
                                            std::shared_ptr<PrimitiveC> lite_primitive) {
  if (!lite_primitive->GetOutputQuantParams().empty()) {
    return RET_OK;
  }
  schema::QuantParamT quant_param;
  quant_param.scale = scale;
  quant_param.zeroPoint = zeropoint;
  quant_param.max = max_min->max;
  quant_param.min = max_min->min;
  quant_param.numBits = bit_num;
  quant_param.narrowRange = false;
  std::vector<schema::QuantParamT> quant_params = {quant_param};
  lite_primitive->AddOutputQuantParam(quant_params);
  return RET_OK;
}

STATUS PostTrainingQuantizer::DoWeightQuant(AnfNodePtr weight, std::shared_ptr<PrimitiveC> primitive_c,
                                            bool perchanel) {
  // perlayer
  if (!weight->isa<Parameter>()) {
    MS_LOG(ERROR) << "not a parameter";
    return RET_PARAM_INVALID;
  }
  auto parameter = std::dynamic_pointer_cast<Parameter>(weight);
  if (parameter == nullptr) {
    MS_LOG(ERROR) << weight->fullname_with_scope() << " can not cast to Parameter";
    return RET_ERROR;
  }
  ParamValueLitePtr paramValue = std::dynamic_pointer_cast<ParamValueLite>(parameter->default_param());
  if (paramValue == nullptr) {
    MS_LOG(ERROR) << weight->fullname_with_scope() << " can not get value";
    return RET_ERROR;
  }
  auto status =
    QuantFilter<int8_t>(paramValue, primitive_c, QuantType_PostTraining, quant_max, quant_min, bit_num, perchanel);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "QuantFilter failed: " << status;
    return status;
  }
  // set dtype
  auto abstractBase = parameter->abstract();
  if (abstractBase == nullptr) {
    MS_LOG(ERROR) << "Abstract of parameter is nullptr, " << parameter->name();
    return RET_ERROR;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstractBase)) {
    MS_LOG(ERROR) << "Abstract of parameter should be anstract tensor, " << parameter->name();
    return RET_ERROR;
  }
  auto abstractTensor = utils::cast<abstract::AbstractTensorPtr>(abstractBase);
  abstractTensor->element()->set_type(TypeIdToType(kNumberTypeInt8));
  return RET_OK;
}

STATUS PostTrainingQuantizer::DoBiasQuant(AnfNodePtr bias, std::shared_ptr<PrimitiveC> primitive_c) {
  if (primitive_c == nullptr || bias == nullptr) {
    MS_LOG(ERROR) << "null pointer!";
    return RET_NULL_PTR;
  }

  auto bias_parameter_ptr = std::dynamic_pointer_cast<Parameter>(bias);
  auto bias_default_param = bias_parameter_ptr->default_param();
  auto bias_param = std::dynamic_pointer_cast<ParamValueLite>(bias_default_param);

  auto active_weight_quant_params = primitive_c->GetInputQuantParams();
  if (active_weight_quant_params.size() != 2) {
    MS_LOG(ERROR) << "unexpected active_weight_quant_params size: " << active_weight_quant_params.size();
    return RET_ERROR;
  }

  auto active_params = active_weight_quant_params[0];
  auto weight_params = active_weight_quant_params[1];

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
  size_t shape_size = bias_param->tensor_shape_size();

  // set bias quant param
  vector<schema::QuantParamT> quant_params;
  for (size_t i = 0; i < bias_scales.size(); i++) {
    schema::QuantParamT quant_param;
    quant_param.scale = bias_scales[i];
    quant_param.zeroPoint = 0;
    quant_param.inited = true;
    quant_params.emplace_back(quant_param);
  }
  // quant bias data
  int32_t *quant_datas = new (std::nothrow) int32_t[shape_size];
  if (quant_datas == nullptr) {
    MS_LOG(ERROR) << "null pointer dereferencing.";
    return RET_NULL_PTR;
  }
  float *raw_datas = static_cast<float *>(bias_param->tensor_addr());
  double bias_scale_tmp;
  constexpr int32_t quanted_bias_abs_limit = 0.5 * INT32_MAX;
  for (size_t i = 0; i < shape_size; i++) {
    if (bias_scales.size() == 1) {
      bias_scale_tmp = bias_scales[0];
    } else {
      bias_scale_tmp = bias_scales[i];
    }
    if (std::abs(raw_datas[i] / bias_scale_tmp) >= quanted_bias_abs_limit) {
      MS_LOG(DEBUG) << "quanted bias over flow, maybe the scale of weight: " << active_weight_quant_params[1][i].scale
                    << " is too small, need to update";
      // update filter scale and zp
      if (input_scales.size() == 1 && active_weight_quant_params[1].size() == shape_size) {
        double activate_scale = input_scales[0];
        double filter_scale = std::abs(raw_datas[i]) / (activate_scale * quanted_bias_abs_limit);
        active_weight_quant_params[1][i].scale = filter_scale;
        active_weight_quant_params[1][i].zeroPoint = 0;
        primitive_c->SetInputQuantParam(active_weight_quant_params);
        bias_scale_tmp = std::abs(raw_datas[i]) / quanted_bias_abs_limit;
        quant_params[i].scale = bias_scale_tmp;
        MS_LOG(DEBUG) << "new filter scale: " << filter_scale;
      } else {
        MS_LOG(WARNING) << "unexpected input_scales size: " << input_scales.size()
                        << " weight_scales size: " << active_weight_quant_params[1].size();
      }
    }
    auto quant_data = (int32_t)std::round(raw_datas[i] / bias_scale_tmp);
    quant_datas[i] = quant_data;
  }
  primitive_c->AddInputQuantParam(quant_params);
  auto ret = memcpy_s(bias_param->tensor_addr(), bias_param->tensor_size(), quant_datas, shape_size * sizeof(int32_t));
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed.";
    delete[] quant_datas;
    return RET_ERROR;
  }
  delete[] quant_datas;
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
  abstractTensor->element()->set_type(TypeIdToType(kNumberTypeInt32));
  return RET_OK;
}

STATUS PostTrainingQuantizer::QuantNode() {
  auto input_min_max = this->calibrator_->GetMinMax(this->calibrator_->GetInputDivergInfo());
  auto input_scale = this->calibrator_->GetScale(this->calibrator_->GetInputDivergInfo());
  auto input_zero_point = this->calibrator_->GetZeropoint(this->calibrator_->GetInputDivergInfo());

  auto output_min_max = this->calibrator_->GetMinMax(this->calibrator_->GetOutputDivergInfo());
  auto output_scale = this->calibrator_->GetScale(this->calibrator_->GetOutputDivergInfo());
  auto output_zeropoint = this->calibrator_->GetZeropoint(this->calibrator_->GetOutputDivergInfo());

  auto cnodes = funcGraph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    auto op_name = cnode->fullname_with_scope();
    if (this->calibrator_->GetInputDivergInfo()->find(op_name) == this->calibrator_->GetInputDivergInfo()->end()) {
      MS_LOG(INFO) << op_name << " can not do quant";
      continue;
    }
    auto primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(0));
    if (primitive_c == nullptr) {
      MS_LOG(ERROR) << "primitive_c is nullptr";
      continue;
    }
    if (input_scale.find(cnode) == input_scale.end()) {
      primitive_c->SetQuantType(schema::QuantType_QUANT_NONE);
      continue;
    }

    auto op_type = (schema::PrimitiveType)primitive_c->Type();
    MS_LOG(DEBUG) << "OpName: " << op_name;
    if (op_type != PrimitiveType_Conv2D && op_type != PrimitiveType_DepthwiseConv2D &&
        op_type != PrimitiveType_FullConnection) {
      for (size_t i = 1; i < cnode->inputs().size(); i++) {
        auto input_node = cnode->input(i);
        if (!input_node->isa<mindspore::CNode>()) {
          MS_LOG(DEBUG) << "node: " << op_name << " input " << i << " not a cnode";
          // get dtype
          auto abstractBase = input_node->abstract();
          if (abstractBase == nullptr) {
            MS_LOG(ERROR) << "Abstract of parameter is nullptr, " << input_node->fullname_with_scope();
            return RET_ERROR;
          }
          if (!utils::isa<abstract::AbstractTensorPtr>(abstractBase)) {
            MS_LOG(ERROR) << "Abstract of parameter should be anstract tensor, " << input_node->fullname_with_scope();
            return RET_ERROR;
          }
          auto abstractTensor = utils::cast<abstract::AbstractTensorPtr>(abstractBase);
          if (abstractTensor->element()->GetTypeTrack()->type_id() == kNumberTypeFloat32) {
            MS_LOG(DEBUG) << "this parameter do quant";
            DoWeightQuant(input_node, primitive_c, false);
          } else {
            MS_LOG(DEBUG) << "this parameter no need to do quant";
          }
          continue;
        }
        auto input_cnode = std::dynamic_pointer_cast<mindspore::CNode>(input_node);
        auto input_cnode_primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(input_cnode->input(0));
        if (input_cnode_primitive_c == nullptr) {
          MS_LOG(DEBUG) << "input: " << i << " " << input_cnode->fullname_with_scope() << ": "
                        << " PrimitiveC is null";
          continue;
        }
        if (!input_cnode_primitive_c->GetOutputQuantParams().empty()) {
          for (auto &quant_param : input_cnode_primitive_c->GetOutputQuantParams()) {
            primitive_c->AddInputQuantParam(quant_param);
          }
        } else {
          // do input quant
          double scale = input_scale[cnode];
          int32_t zp = input_zero_point[cnode];
          DoQuantInput(scale, zp, &input_min_max[cnode], primitive_c);
        }
      }
    } else {
      // do input quant
      double scale = input_scale[cnode];
      int32_t convInputzeropoint = input_zero_point[cnode];
      DoQuantInput(scale, convInputzeropoint, &input_min_max[cnode], primitive_c);
      // do weight quant
      auto weight = cnode->input(2);
      bool perchannel = per_channel_;
      if (op_type == PrimitiveType_FullConnection) {
        perchannel = false;
      }
      DoWeightQuant(weight, primitive_c, perchannel);
      // do bias quant
      if (cnode->inputs().size() == 4) {
        auto bias = cnode->input(3);
        DoBiasQuant(bias, primitive_c);
      }
    }
    // do output quant
    double OutputScale = output_scale[cnode];
    int32_t OutputZeropoint = output_zeropoint[cnode];
    DoQuantOutput(OutputScale, OutputZeropoint, &output_min_max[cnode], primitive_c);
    primitive_c->SetQuantType(schema::QuantType_PostTraining);
  }
  return RET_OK;
}

STATUS PostTrainingQuantizer::UpdateDivergInverval() {
  this->calibrator_->UpdateDivergInverval(this->calibrator_->GetInputDivergInfo());
  this->calibrator_->UpdateDivergInverval(this->calibrator_->GetOutputDivergInfo());
  return RET_OK;
}

/**
 * Pre Process
 * 1. generate config param
 *   1.1 read config file
 *   1.2 parse txt
 * 2. collect image files
 *   2.1 parse image files to input tensor
 * 3. save quantied node
 **/
STATUS PostTrainingQuantizer::PreProcess() {
  if (this->calibrator_ == nullptr) {
    MS_LOG(ERROR) << "calibrator is null!";
    return RET_ERROR;
  }
  // 1. generate config param
  STATUS status = calibrator_->ReadConfig();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "read proto text failed!";
    return status;
  }
  // 2. collect image files
  status = calibrator_->CollectImages();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "collect images failed!";
    return status;
  }
  // 3. collect to be quantized operators
  // from user input
  QuantStrategy strategy(10);
  auto cnodes = funcGraph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    AnfNodePtr anf = std::dynamic_pointer_cast<AnfNode>(cnode);
    if (strategy.CanOpPostQuantized(anf)) {
      MS_LOG(INFO) << "node: " << cnode->fullname_with_scope() << " will be quantized";
      calibrator_->AddQuantizedOp(cnode);
    }
  }
  return RET_OK;
}

STATUS PostTrainingQuantizer::CheckFp32TensorVec(const std::string &node_name,
                                                 const std::vector<mindspore::tensor::MSTensor *> &tensor_vec) const {
  if (tensor_vec.size() < 1) {
    MS_LOG(ERROR) << "node: " << node_name << " input tensors is 0";
    return RET_ERROR;
  }
  auto *tensor = tensor_vec[0];
  if (tensor->data_type() != kNumberTypeFloat32) {
    MS_LOG(WARNING) << "node: " << node_name << " will not quantize"
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
STATUS PostTrainingQuantizer::DoInference() {
  for (size_t i = 0; i < calibrator_->GetBatchNum(); i++) {
    // get input tensor
    vector<mindspore::tensor::MSTensor *> inputs = fp32_session_->GetInputs();
    if (inputs.size() > 1) {
      MS_LOG(ERROR) << "model's input tensor size: " << inputs.size() << " >1";
      return RET_ERROR;
    }
    STATUS status = calibrator_->GenerateInputData(i, inputs.front());
    if (status != RET_OK) {
      MS_LOG(ERROR) << "generate input data from images failed!";
      return RET_ERROR;
    }
    mindspore::session::KernelCallBack beforeCallBack =
      [&](const std::vector<mindspore::tensor::MSTensor *> &beforeInputs,
          const std::vector<mindspore::tensor::MSTensor *> &beforeOutputs,
          const mindspore::session::CallBackParam &callParam) -> bool {
      if (PostTrainingQuantizer::CheckFp32TensorVec(callParam.node_name, beforeInputs) != RET_OK) {
        return false;
      }
      auto tensor = beforeInputs[0];
      const float *tData = static_cast<const float *>(tensor->MutableData());
      size_t elem_count = tensor->ElementsNum();
      vector<float> data(tData, tData + elem_count);
      this->calibrator_->RecordMaxValue(callParam.node_name, data, this->calibrator_->GetInputDivergInfo());
      return true;
    };
    // func
    mindspore::session::KernelCallBack afterCallBack = [&](
                                                         const std::vector<mindspore::tensor::MSTensor *> &afterInputs,
                                                         const std::vector<mindspore::tensor::MSTensor *> &afterOutputs,
                                                         const mindspore::session::CallBackParam &callParam) -> bool {
      if (PostTrainingQuantizer::CheckFp32TensorVec(callParam.node_name, afterOutputs) != RET_OK) {
        return false;
      }
      auto tensor = afterOutputs[0];
      const float *tensor_data = static_cast<const float *>(tensor->MutableData());
      size_t elem_count = tensor->ElementsNum();
      vector<float> data(tensor_data, tensor_data + elem_count);
      this->calibrator_->RecordMaxValue(callParam.node_name, data, this->calibrator_->GetOutputDivergInfo());
      return true;
    };
    status = fp32_session_->RunGraph(beforeCallBack, afterCallBack);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "run model failed!";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS PostTrainingQuantizer::Int8Inference() {
  // fp32 inference
  vector<mindspore::tensor::MSTensor *> inputs = int8_session_->GetInputs();
  // get input tensor
  if (inputs.size() != 1) {
    MS_LOG(ERROR) << "model's input tensor size: " << inputs.size();
    return RET_ERROR;
  }
  auto elem_count = inputs.front()->ElementsNum();
  vector<float> dummy_data(elem_count);
  std::fill(dummy_data.begin(), dummy_data.end(), 0.1);
  auto ret = memcpy_s(inputs.front()->MutableData(), inputs.front()->Size(), dummy_data.data(),
                      sizeof(float) * dummy_data.size());
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy_s error: " << ret;
    return RET_ERROR;
  }

  for (size_t i = 0; i < calibrator_->GetBatchNum(); i++) {
    mindspore::session::KernelCallBack beforeCallBack =
      [this](const std::vector<mindspore::tensor::MSTensor *> &beforeInputs,
             const std::vector<mindspore::tensor::MSTensor *> &beforeOutputs,
             const mindspore::session::CallBackParam &callParam) -> bool {
      if (callParam.node_type == kTypeConv2D || callParam.node_type == kTypeDepthwiseConv2D) {
        vector<float> fp32_op_input;
        while (!OpInputDataHandle(FETCH, callParam.node_name, &fp32_op_input)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        auto tensor = beforeInputs[0];
        auto lite_tensor = dynamic_cast<mindspore::lite::Tensor *>(tensor);

        if (tensor->data_type() != kNumberTypeInt8) {
          MS_LOG(ERROR) << "unexpected tensor type: " << tensor->data_type();
          return false;
        }

        // do quantization: activation is always per layer quantized
        std::vector<int8_t> quant_datas;
        auto quant_params = lite_tensor->GetQuantParams();
        if (quant_params.size() != 1) {
          MS_LOG(ERROR) << "unexpected quant_params size: " << quant_params.size();
          return false;
        }
        schema::QuantParamT quant_param_t;
        quant_param_t.scale = quant_params[0].scale;
        quant_param_t.zeroPoint = quant_params[0].zeroPoint;
        for (auto float_data : fp32_op_input) {
          auto quant_data = QuantizeData<int8_t>(float_data, quant_param_t, quant_max, quant_min);
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
    // func
    mindspore::session::KernelCallBack afterCallBack = [this](
                                                         const std::vector<mindspore::tensor::MSTensor *> &afterInputs,
                                                         const std::vector<mindspore::tensor::MSTensor *> &afterOutputs,
                                                         const mindspore::session::CallBackParam &callParam) -> bool {
      if (callParam.node_type == kTypeConv2D || callParam.node_type == kTypeDepthwiseConv2D) {
        vector<float> fp32_op_output_ch_mean;
        while (!OpOutputChMeanDataHandle(FETCH, callParam.node_name, &fp32_op_output_ch_mean)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        auto tensor = afterOutputs[0];
        auto lite_tensor = dynamic_cast<mindspore::lite::Tensor *>(tensor);

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
        auto quant_params = lite_tensor->GetQuantParams();
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
              return RET_ERROR;
            }
            // deuqant activation
            auto float_data = scale * (tensor_data[index] - zp);
            sum += float_data;
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
    ret = int8_session_->RunGraph(beforeCallBack, afterCallBack);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "run model failed!";
      return RET_ERROR;
    }
  }  // end for images
  return RET_OK;
}

STATUS PostTrainingQuantizer::BiasCorrection(FuncGraphPtr func_graph) {
  auto ret = RET_OK;
  std::future<STATUS> int8_inference = std::async(std::launch::async, &PostTrainingQuantizer::Int8Inference, this);

  // fp32 inference
  for (size_t i = 0; i < calibrator_->GetBatchNum(); i++) {
    // get input tensor
    vector<mindspore::tensor::MSTensor *> inputs = fp32_session_->GetInputs();
    if (inputs.size() != 1) {
      MS_LOG(ERROR) << "model's input tensor size: " << inputs.size();
      return RET_ERROR;
    }
    STATUS status = calibrator_->GenerateInputData(i, inputs.front());
    if (status != RET_OK) {
      MS_LOG(ERROR) << "generate input data from images failed!";
      return RET_ERROR;
    }
    mindspore::session::KernelCallBack beforeCallBack =
      [this](const std::vector<mindspore::tensor::MSTensor *> &beforeInputs,
             const std::vector<mindspore::tensor::MSTensor *> &beforeOutputs,
             const mindspore::session::CallBackParam &callParam) -> bool {
      if (callParam.node_type == kTypeConv2D || callParam.node_type == kTypeDepthwiseConv2D) {
        if (PostTrainingQuantizer::CheckFp32TensorVec(callParam.node_name, beforeInputs) != RET_OK) {
          return false;
        }
        auto tensor = beforeInputs[0];
        size_t elem_count = tensor->ElementsNum();
        std::vector<float> fp32_op_input(elem_count);
        auto ret =
          memcpy_s(fp32_op_input.data(), fp32_op_input.size() * sizeof(float), tensor->MutableData(), tensor->Size());
        if (ret != EOK) {
          MS_LOG(ERROR) << "memcpy error: " << ret;
          return false;
        }
        while (!OpInputDataHandle(STORE, callParam.node_name, &fp32_op_input)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      }
      return true;
    };
    // func
    mindspore::session::KernelCallBack afterCallBack = [this](
                                                         const std::vector<mindspore::tensor::MSTensor *> &afterInputs,
                                                         const std::vector<mindspore::tensor::MSTensor *> &afterOutputs,
                                                         const mindspore::session::CallBackParam &callParam) -> bool {
      if (callParam.node_type == kTypeConv2D || callParam.node_type == kTypeDepthwiseConv2D) {
        if (PostTrainingQuantizer::CheckFp32TensorVec(callParam.node_name, afterOutputs) != RET_OK) {
          return false;
        }
        auto tensor = afterOutputs[0];
        const float *tensor_data = static_cast<const float *>(tensor->MutableData());
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
              return RET_ERROR;
            }
            sum += tensor_data[index];
          }
          sum = sum / one_filter_size;
          fp32_op_output_ch_mean[i] = sum;
        }
        while (!OpOutputChMeanDataHandle(STORE, callParam.node_name, &fp32_op_output_ch_mean)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      }

      return true;
    };
    status = fp32_session_->RunGraph(beforeCallBack, afterCallBack);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "run model failed!";
      return RET_ERROR;
    }
  }  // end for images

  ret = int8_inference.get();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "int8 inference failed!";
    return RET_ERROR;
  }
  for (auto &key_value : op_bias_diff_map) {
    std::for_each(key_value.second.begin(), key_value.second.end(),
                  [this](float &data) { data = data / calibrator_->GetBatchNum(); });
  }
  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    auto op_name = cnode->fullname_with_scope();
    if (op_bias_diff_map.find(op_name) != op_bias_diff_map.end()) {
      const auto &bias_diff = op_bias_diff_map[op_name];
      auto primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(0));
      if (primitive_c == nullptr) {
        MS_LOG(ERROR) << "primitive_c is nullptr";
        continue;
      }
      auto input_quant_params = primitive_c->GetInputQuantParams();

      if (input_quant_params.size() == 3) {
        // compensate the existed
        auto bias_quant_params = input_quant_params[2];
        auto bias = cnode->input(3);
        auto bias_parameter_ptr = std::dynamic_pointer_cast<Parameter>(bias);
        auto bias_default_param = bias_parameter_ptr->default_param();
        auto bias_param = std::dynamic_pointer_cast<ParamValueLite>(bias_default_param);
        int *bias_datas = static_cast<int *>(bias_param->tensor_addr());

        if (static_cast<size_t>(bias_param->tensor_shape_size()) != bias_diff.size()) {
          MS_LOG(ERROR) << "unexpected bias data count: " << bias_param->tensor_shape_size()
                        << " not the same as bias_diff: " << bias_diff.size();
          continue;
        }
        if (bias_quant_params.size() != bias_diff.size()) {
          MS_LOG(ERROR) << "unexpected bias quant params size: " << bias_quant_params.size()
                        << " not the same as bias_diff: " << bias_diff.size();
        }

        for (int i = 0; i < bias_param->tensor_shape_size(); i++) {
          auto scale = bias_quant_params[i].scale;
          double after_correct = std::round(bias_diff[i] / scale) + bias_datas[i];
          constexpr int32_t corrected_bias_abs_limit = 0.6 * INT32_MAX;
          if (after_correct > corrected_bias_abs_limit) {
            MS_LOG(WARNING) << op_name << " ch: " << i << " bias after_corrected too large: " << after_correct
                            << " origin value: " << bias_datas[i] << " bias_diff: " << bias_diff[i]
                            << " scale: " << scale;
            bias_datas[i] = static_cast<int>(corrected_bias_abs_limit);
          } else if (after_correct < -corrected_bias_abs_limit) {
            MS_LOG(WARNING) << op_name << " ch: " << i << " bias after_corrected too small: " << after_correct
                            << " origin value: " << bias_datas[i] << " bias_diff: " << bias_diff[i]
                            << " scale: " << scale;
            bias_datas[i] = static_cast<int>(-corrected_bias_abs_limit);
          } else {
            auto diff = static_cast<int>(std::round(bias_diff[i] / scale));
            bias_datas[i] += diff;
          }
        }
      } else if (input_quant_params.size() == 2) {
        MS_LOG(INFO) << op_name << " add bias input";
        // need to add bias input
        auto parameter = func_graph->add_parameter();
        ShapeVector shape;
        shape.push_back(bias_diff.size());
        auto type_ptr = TypeIdToType(kNumberTypeFloat32);
        auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape);
        parameter->set_abstract(abstract_tensor);
        parameter->set_name("added_" + op_name + "_bias");

        ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();
        MS_ASSERT(param_value != nullptr);
        param_value->set_tensor_shape(shape);
        param_value->set_tensor_type(kNumberTypeFloat32);
        // param_value->set_format(tensor->format);

        auto size = sizeof(float) * bias_diff.size();
        char *tensor_data = new (std::nothrow) char[size];
        if (tensor_data == nullptr) {
          MS_LOG(ERROR) << "new char[] failed";
          return RET_MEMORY_FAILED;
        }
        std::memcpy(tensor_data, bias_diff.data(), size);
        param_value->set_tensor_addr(tensor_data);
        param_value->set_tensor_size(size);
        parameter->set_default_param(param_value);
        cnode->add_input(parameter);
        DoBiasQuant(parameter, primitive_c);

        auto op_type = (schema::PrimitiveType)primitive_c->Type();
        if (op_type == schema::PrimitiveType_Conv2D) {
          auto conv2d = primitive_c->GetPrimitiveT()->value.AsConv2D();
          if (conv2d == nullptr) {
            MS_LOG(ERROR) << "conv2d is null";
            return RET_ERROR;
          }
          conv2d->hasBias = true;
        } else if (op_type == schema::PrimitiveType_DepthwiseConv2D) {
          auto depthwise_conv2d = primitive_c->GetPrimitiveT()->value.AsDepthwiseConv2D();
          if (depthwise_conv2d == nullptr) {
            MS_LOG(ERROR) << "conv2d is null";
            return RET_ERROR;
          }
          depthwise_conv2d->hasBias = true;
        }
      } else {
        MS_LOG(ERROR) << "unexpected input_quant_params size: " << input_quant_params.size();
        continue;
      }
    }  // end fine op_name
  }

  return ret;
}

STATUS PostTrainingQuantizer::CollectDataFrequency() {
  for (size_t i = 0; i < calibrator_->GetBatchNum(); i++) {
    // get input tensor
    vector<mindspore::tensor::MSTensor *> inputs = fp32_session_->GetInputs();
    if (inputs.size() > 1) {
      MS_LOG(ERROR) << "model's input tensor size: " << inputs.size() << " > 1";
      return RET_ERROR;
    }
    STATUS status = calibrator_->GenerateInputData(i, inputs.front());
    if (status != RET_OK) {
      MS_LOG(ERROR) << "generate input data from images failed!";
      return RET_ERROR;
    }

    mindspore::session::KernelCallBack beforeCallBack =
      [&](const std::vector<mindspore::tensor::MSTensor *> &beforeInputs,
          const std::vector<mindspore::tensor::MSTensor *> &beforeOutputs,
          const mindspore::session::CallBackParam &callParam) {
        if (PostTrainingQuantizer::CheckFp32TensorVec(callParam.node_name, beforeInputs) != RET_OK) {
          return false;
        }
        auto tensor = beforeInputs[0];
        const float *tensor_data = static_cast<const float *>(tensor->MutableData());
        size_t shape_size = tensor->ElementsNum();
        vector<float> data(tensor_data, tensor_data + shape_size);
        this->calibrator_->UpdateDataFrequency(callParam.node_name, data, this->calibrator_->GetInputDivergInfo());
        return true;
      };

    mindspore::session::KernelCallBack afterCallBack =
      [&](const std::vector<mindspore::tensor::MSTensor *> &after_inputs,
          const std::vector<mindspore::tensor::MSTensor *> &after_outputs,
          const mindspore::session::CallBackParam &call_param) {
        if (PostTrainingQuantizer::CheckFp32TensorVec(call_param.node_name, after_outputs) != RET_OK) {
          return false;
        }
        auto tensor = after_outputs[0];
        const float *tenosr_data = static_cast<const float *>(tensor->MutableData());
        size_t shape_size = tensor->ElementsNum();
        vector<float> data(tenosr_data, tenosr_data + shape_size);
        this->calibrator_->UpdateDataFrequency(call_param.node_name, data, this->calibrator_->GetOutputDivergInfo());
        return true;
      };
    status = fp32_session_->RunGraph(beforeCallBack, afterCallBack);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "run model failed!";
      return RET_ERROR;
    }
  }

  return RET_OK;
}

STATUS PostTrainingQuantizer::ComputeThreshold() { return this->calibrator_->ComputeThreshold(); }

STATUS PostTrainingQuantizer::DoQuantize(FuncGraphPtr func_graph) {
  MS_LOG(INFO) << "start to parse config file";
  STATUS status = PreProcess();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "do pre process failed!";
    return status;
  }
  // anf -- fb
  auto meta_graph = Export(func_graph, true, true);
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "Export to meta_graph return nullptr";
    return RET_ERROR;
  }

  // transform
  GraphDefTransform transform;
  transform.SetGraphDef(meta_graph);
  flags.quantType = schema::QuantType_QUANT_NONE;
  status = transform.Transform(flags);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "FBTransform model failed " << status;
    return RET_ERROR;
  }
  MS_LOG(INFO) << "start create session";
  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = schema::MetaGraph::Pack(builder, meta_graph);
  builder.Finish(offset);
  size_t size = builder.GetSize();
  auto *content = reinterpret_cast<const char *>(builder.GetBufferPointer());
  if (content == nullptr) {
    MS_LOG(ERROR) << "GetBufferPointer nullptr";
    return RET_ERROR;
  }
  auto model = lite::Model::Import(content, size);

  Context ctx;
  ctx.thread_num_ = calibrator_->GetThreadNum();

  fp32_session_ = dynamic_cast<mindspore::lite::LiteSession *>(session::LiteSession::CreateSession(&ctx));
  if (fp32_session_ == nullptr) {
    MS_LOG(ERROR) << "create session failed!";
    return RET_ERROR;
  }

  auto ret = fp32_session_->CompileGraph(model);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "compile graph error";
    return RET_ERROR;
  }

  MS_LOG(INFO) << "start to update divergence's max value";
  status = DoInference();
  if (status != RET_OK) {
    return status;
  }
  MS_LOG(INFO) << "start to update divergence's interval";
  status = UpdateDivergInverval();
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
  quant_cast.SetInputDataDType(kNumberTypeFloat32);
  status = quant_cast.Run(func_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "add QuantCast error";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return RET_ERROR;
  }

  if (calibrator_->GetBiasCorrection()) {
    // init in8 session
    // anf -- fb
    auto int8_meta_graph = Export(func_graph, true, true);
    if (int8_meta_graph == nullptr) {
      MS_LOG(ERROR) << "Export to int8_meta_graph return nullptr";
      return RET_ERROR;
    }

    // transform
    GraphDefTransform fb_transform;
    fb_transform.SetGraphDef(int8_meta_graph);
    flags.quantType = schema::QuantType_PostTraining;
    status = fb_transform.Transform(flags);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "FBTransform model failed " << status;
      return RET_ERROR;
    }
    MS_LOG(INFO) << "start create quantized session";
    flatbuffers::FlatBufferBuilder int8_builder(1024);
    auto int8_offset = schema::MetaGraph::Pack(int8_builder, int8_meta_graph);
    int8_builder.Finish(int8_offset);
    size = int8_builder.GetSize();
    auto *int8_content = reinterpret_cast<const char *>(int8_builder.GetBufferPointer());
    if (int8_content == nullptr) {
      MS_LOG(ERROR) << "GetBufferPointer nullptr";
      return RET_ERROR;
    }
    auto int8_model = lite::Model::Import(int8_content, size);

    Context int8_ctx;
    int8_ctx.thread_num_ = calibrator_->GetThreadNum();
    int8_ctx.device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = HIGHER_CPU;

    int8_session_ = dynamic_cast<mindspore::lite::LiteSession *>(session::LiteSession::CreateSession(&int8_ctx));
    if (int8_session_ == nullptr) {
      MS_LOG(ERROR) << "create session failed!";
      return RET_ERROR;
    }
    ret = int8_session_->CompileGraph(int8_model);
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "compile graph error";
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

bool PostTrainingQuantizer::OpInputDataHandle(OperationType type, const string &op_name, std::vector<float> *data) {
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

bool PostTrainingQuantizer::OpOutputChMeanDataHandle(OperationType type, const string &op_name,
                                                     std::vector<float> *data) {
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

}  // namespace quant
}  // namespace lite
}  // namespace mindspore
