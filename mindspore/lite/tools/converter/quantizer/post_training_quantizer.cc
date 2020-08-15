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

#include <dirent.h>
#include <sys/stat.h>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <memory>
#include <numeric>
#include <utility>
#include <string>
#include <vector>
#include <fstream>
#include "schema/inner/model_generated.h"
#include "src/ir/tensor.h"
#include "src/common/anf_exporter/anf_exporter.h"
#include "tools/converter/quantizer/post_training_quantizer.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "src/common/common.h"
#include "utils/log_adapter.h"
#include "securec/include/securec.h"
#include "tools/common/tensor_util.h"
#include "src/common/file_utils.h"

using std::string;
using std::vector;

namespace mindspore {
namespace lite {
namespace quant {

struct DivergInfo {
  std::vector<float> histogram;
  CNodePtr cnode;
  int bin_num;
  float interval = 0;
  float max;
  float min;
  float best_T = 0.0f;
  size_t bit_num;
  int quant_max = 255;
  int quant_min = 0;
  std::string method_x = kMethodKL;

  DivergInfo(CNodePtr cnode, int bins, size_t bits, int quant_max, int quant_min, const std::string &method_x) {
    this->method_x = method_x;
    this->cnode = cnode;
    this->bin_num = bins;
    this->bit_num = bits;
    histogram.resize(bin_num);
    max = -FLT_MAX;
    min = FLT_MAX;
    this->quant_max = quant_max;
    this->quant_min = quant_min;
    std::fill(histogram.begin(), histogram.end(), 1.0e-7);
  }

  STATUS RecordMaxValue(const std::vector<float> &datas) {
    for (float data : datas) {
      max = std::max(data, max);
      min = std::min(data, min);
    }
    return RET_OK;
  }

  void UpdateInterval() {
    auto max_value = std::max(fabs(this->max), fabs(this->min));
    this->interval = max_value / static_cast<float>(bin_num);
  }

  STATUS UpdateHistogram(const std::vector<float> &data, const std::vector<int> &shape) {
    for (auto value : data) {
      if (value == 0) {
        continue;
      }
      int bin_index = std::min(static_cast<int>(std::fabs(value) / this->interval), bin_num - 1);
      this->histogram[bin_index]++;
    }
    return RET_OK;
  }

  void DumpHistogram() {
    MS_LOG(INFO) << "Print node " << cnode->fullname_with_scope() << " histogram";
    for (float item : this->histogram) {
      std::cout << item << " ";
    }
    std::cout << std::endl;
  }

  STATUS ComputeThreshold() {
    if (method_x == kMethodMaxMin) {
      this->best_T = std::max(fabs(this->max), fabs(this->min));
      MS_LOG(DEBUG) << "using MAX_MIN, T: " << this->best_T;
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
        std::for_each(this->histogram.begin() + left_upper, this->histogram.begin() + right_lower,
                      [&count](float item) {
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
    MS_LOG(DEBUG) << cnode->fullname_with_scope() << " Best threshold bin index: " << threshold
                                                  << " T: " << best_T
                                                  << " max: " << std::max(fabs(this->max), fabs(this->min));
    return RET_OK;
  }

  std::pair<CNodePtr, float> GetScale() {
    float max_value = this->best_T;
    float min_value = -max_value;

    MS_ASSERT(quant_max - quant_min != 0);
    float scale = (max_value - min_value) / (quant_max - quant_min);
    MS_ASSERT(scale != 0);
    return std::make_pair(this->cnode, scale);
  }

  std::pair<CNodePtr, int32_t> GetZeropoint() {
    int zero_point = 0;
    if (quant_min == 0 && quant_max == 255) {
      zero_point = 128;
    } else if (quant_min == -128 && quant_max == 127) {
      zero_point = 0;
    } else {
      MS_LOG(ERROR) << "unexpectd quant range, quant_min: " << quant_min << " quant_max: " << quant_max;
    }
    return std::make_pair(this->cnode, zero_point);
  }
};
std::unordered_map<CNodePtr, float> Calibrator::GetResult(
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
  std::unordered_map<std::string, std::unique_ptr<DivergInfo>> *mDivergInfo) {
  std::unordered_map<CNodePtr, int32_t> result;
  for (auto iter = mDivergInfo->begin(); iter != mDivergInfo->end(); iter++) {
    DivergInfo *info = iter->second.get();
    auto zeropoint = info->GetZeropoint();
    result.insert(zeropoint);
  }
  return result;
}

std::map<CNodePtr, MaxMin> Calibrator::GetMinMax(
  std::unordered_map<std::string, std::unique_ptr<DivergInfo>> *mDivergInfo) {
  std::map<CNodePtr, MaxMin> result;
  for (auto iter = mDivergInfo->begin(); iter != mDivergInfo->end(); iter++) {
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

STATUS Calibrator::RecordMaxValue(std::string opName, vector<float> data,
                                  std::unordered_map<std::string, std::unique_ptr<DivergInfo>> *mDivergInfo) {
  auto got = (*mDivergInfo).find(opName);
  if (got != (*mDivergInfo).end()) {
    ((*got).second)->RecordMaxValue(data);
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

STATUS Calibrator::UpdateDataFrequency(std::string op_name, vector<float> data, vector<int> shape,
                                       std::unordered_map<std::string, std::unique_ptr<DivergInfo>> *diverg_info) {
  auto got = (*diverg_info).find(op_name);
  if (got != (*diverg_info).end()) {
    ((*got).second)->UpdateHistogram(data, shape);
  }
  return RET_OK;
}

STATUS Calibrator::AddQuantizedOp(CNodePtr node) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "To be quantized node is null";
    return RET_ERROR;
  }
  string node_name = node->fullname_with_scope();
  std::unique_ptr<DivergInfo> input_diverg =
    std::unique_ptr<DivergInfo>(new DivergInfo(node, 2048, bit_num_, quant_max_, quant_min_, config_param_.method_x));
  std::unique_ptr<DivergInfo> output_diverg =
    std::unique_ptr<DivergInfo>(new DivergInfo(node, 2048, bit_num_, quant_max_, quant_min_, config_param_.method_x));

  input_diverg_info_.insert(std::make_pair(string(node_name), std::move(input_diverg)));
  output_diverg_info_.insert(std::make_pair(string(node_name), std::move(output_diverg)));
  return RET_OK;
}

void Calibrator::AddImage(const string file) {
  auto exist = [](const string file) {
    struct stat buf;
    return stat(file.c_str(), &buf) == 0;
  };
  if (exist(file)) {
    MS_LOG(INFO) << "load image: " << file;
    this->images_.push_back(file);
  } else {
    MS_LOG(WARNING) << "Invaild image file path: " << file;
  }
}

STATUS Calibrator::GenerateInputData(const int index, mindspore::tensor::MSTensor *tensor) const {
  string path = images_[index];
  MS_LOG(INFO) << "read image: " << path;
  size_t size;
  char *binBuf = ReadFile(path.c_str(), &size);
  auto data = tensor->MutableData();
  if (size != tensor->Size()) {
    MS_LOG(ERROR) << "the input data is not consistent with model input, file_size: " << size
                  << " input tensor size: " << tensor->Size();
    return RET_ERROR;
  }
  memcpy(data, binBuf, size);
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
  int count = 0;
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
  if (nullptr != realpath(config_path_.c_str(), resolved_path)) {
    config_path_ = string(resolved_path);
  }
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
    if (key == "image_path") {
      config_param_.image_path = value;
    } else if (key == "batch_count") {
      config_param_.batch_count = std::stoul(value);
    } else if (key == "thread_num") {
      config_param_.thread_num = std::stoul(value);
    } else if (key == "method_x") {
      if (value != kMethodKL && value != kMethodMaxMin) {
        MS_LOG(WARNING) << "unsupported method_x: " << value << ". Use default value.";
      } else {
        config_param_.method_x = value;
      }
    } else {
      MS_LOG(WARNING) << "unsupported parameter";
    }
  }
  MS_LOG(DEBUG) << "image_path: "   << config_param_.image_path    << "  "
                << "batch_count: "  << config_param_.batch_count   << "  "
                << "mothod_x: "     << config_param_.method_x      << "  "
                << "thread_num: "   << config_param_.thread_num;

  delete[] resolved_path;
  fs.close();
  return RET_OK;
}

Calibrator::Calibrator(string path, size_t bitNum, int quantMax, int quantMin)
    : config_path_(path), bit_num_(bitNum), quant_max_(quantMax), quant_min_(quantMin) {}

PostTrainingQuantizer::PostTrainingQuantizer(FuncGraphPtr graph, string path, int bit_num, TypeId target_type,
                                             bool per_channel)
    : Quantizer(graph) {
  this->per_channel_ = per_channel;
  this->bit_num = bit_num;
  this->target_type_ = target_type;
  if (target_type == kNumberTypeInt8) {
    quant_max = (1 << (this->bit_num - 1)) - 1;  // 127
    quant_min = -(1 << (this->bit_num - 1));     // -128
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
                                           std::shared_ptr<PrimitiveTValue> lite_primitive) {
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
  // p->AddAttr("quant_input_dataType", MakeValue((int)DataType_DT_FLOAT));
  return RET_OK;
}

STATUS PostTrainingQuantizer::DoQuantOutput(double scale, int zeropoint, struct MaxMin *max_min,
                                            std::shared_ptr<PrimitiveTValue> lite_primitive) {
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
  // p->AddAttr("quant_output_dataType", MakeValue((int)DataType_DT_FLOAT));
  return RET_OK;
}

STATUS PostTrainingQuantizer::DoWeightQuant(AnfNodePtr node) {
  // const vector<int> dims = filter->dims;
  // perlayer
  if (!node->isa<Parameter>()) {
    MS_LOG(ERROR) << "not a parameter";
    return RET_PARAM_INVALID;
  }
  auto parameter = std::dynamic_pointer_cast<Parameter>(node);
  ParamValueLitePtr paramValue = std::dynamic_pointer_cast<ParamValueLite>(parameter->default_param());
  auto status = QuantFilter(paramValue, QuantType_PostTraining, quant_max, quant_min, bit_num, per_channel_);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "QuantFilter failed: " << status;
    return status;
  }
  return RET_OK;
}

STATUS PostTrainingQuantizer::DoBiasQuant(std::shared_ptr<PrimitiveTValue> input, AnfNodePtr weight, AnfNodePtr bias) {
  if (input == nullptr || weight == nullptr || bias == nullptr) {
    MS_LOG(ERROR) << "null pointer!";
    return RET_NULL_PTR;
  }

  ParameterPtr weightParameterPtr = std::dynamic_pointer_cast<Parameter>(weight);
  auto default_param = weightParameterPtr->default_param();
  auto weight_param = std::dynamic_pointer_cast<ParamValueLite>(default_param);
  // std::vector<std::unique_ptr<mindspore::QuantParamT>> weight_quant_params = weight_param->get_quant_params();

  ParameterPtr biasParameterPtr = std::dynamic_pointer_cast<Parameter>(bias);
  auto bias_default_param = biasParameterPtr->default_param();
  auto bias_param = std::dynamic_pointer_cast<ParamValueLite>(bias_default_param);

  vector<double> input_scales;
  vector<double> filter_scales;
  vector<double> bias_scales;
  auto quant_params = input->GetInputQuantParams();
  size_t sizeX = quant_params.size();
  for (size_t i = 0; i < sizeX; i++) {
    input_scales.emplace_back(quant_params[i].front().scale);
  }
  size_t sizeY = weight_param->quant_param().size();
  if (sizeX != sizeY) {
    if (sizeX > 1 && sizeY > 1) {
      MS_LOG(ERROR) << "input and filter's scale count cannot match!";
      return RET_ERROR;
    }
  }
  for (size_t i = 0; i < sizeY; i++) {
    auto scale = weight_param->quant_param()[i]->scale;
    filter_scales.push_back(scale);
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
  bias_param->quant_param().clear();
  for (size_t i = 0; i < bias_scales.size(); i++) {
    std::unique_ptr<AnfQuantParam> param(new (std::nothrow) AnfQuantParam());
    param->scale = bias_scales[i];
    param->zeroPoint = 0;
    bias_param->quant_param().emplace_back(std::move(param));
  }
  // quant bias data
  int32_t *quant_datas = new (std::nothrow) int32_t[shape_size];
  if (quant_datas == nullptr) {
    MS_LOG(ERROR) << "null pointer dereferencing.";
    return RET_NULL_PTR;
  }
  float *raw_datas = reinterpret_cast<float *>(bias_param->tensor_addr());
  double bias_scale_tmp;
  for (size_t i = 0; i < shape_size; i++) {
    if (bias_scales.size() == 1) {
      bias_scale_tmp = bias_scales[0];
    } else {
      bias_scale_tmp = bias_scales[i];
    }
    auto quant_data = (int32_t)std::round(raw_datas[i] / bias_scale_tmp);
    quant_datas[i] = quant_data;
  }
  auto ret =
    memcpy_s(bias_param->tensor_addr(), bias_param->tensor_size(), quant_datas, shape_size * sizeof(int32_t));
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed.";
    delete[] quant_datas;
    return RET_ERROR;
  }
  delete[] quant_datas;
  bias_param->set_tensor_type(kNumberTypeInt32);
  return RET_OK;
}

// STATUS PostTrainingQuantizer::reformatConvWeight(GraphDefT *graph) {
//   for (auto &subGraph : graphDefT->subgraphs) {
//      for (auto iter = subGraph->nodes.begin(); iter != subGraph->nodes.end(); iter++) {
//          OpDefT *node = (*iter).get();
//          bool isConv = false;
//          kTransFilterType tansType;
//          if ((*node).attr.type == OpT_Conv2D) {
//              tansType = kKCHW2HWCK;
//              isConv = true;
//          }
//          else if ((*node).attr.type == OpT_DepthwiseConv2D) {
//              tansType = kCKHW2HWCK;
//              isConv = true;
//          }
//          if (isConv) {
//              auto status =  TransFilterFormat<uint8_t>(&(*subGraph.get()->allTensors.at(node->inputIndex[1])),
//                                                        tansType);
//              if (status != RET_OK) {
//                  return status;
//              }
//              TensorDefT *weight = subGraph->allTensors.at(node->inputIndex[1]).get();
//              weight->format = Format_HWCK;
//              PostBitPack(weight, bitNum);
//          }
//      }
//  }
//}

STATUS PostTrainingQuantizer::QuantNode() {
  auto input_min_max = this->calibrator_->GetMinMax(this->calibrator_->GetInputDivergInfo());
  auto input_scale = this->calibrator_->GetResult(this->calibrator_->GetInputDivergInfo());
  auto input_zero_point = this->calibrator_->GetZeropoint(this->calibrator_->GetInputDivergInfo());

  auto output_min_max = this->calibrator_->GetMinMax(this->calibrator_->GetOutputDivergInfo());
  auto output_scale = this->calibrator_->GetResult(this->calibrator_->GetOutputDivergInfo());
  auto output_zeropoint = this->calibrator_->GetZeropoint(this->calibrator_->GetOutputDivergInfo());

  auto cnodes = funcGraph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    auto cnode_name = cnode->fullname_with_scope();
    if (this->calibrator_->GetInputDivergInfo()->find(cnode_name) == this->calibrator_->GetInputDivergInfo()->end()) {
      MS_LOG(INFO) << cnode_name << " can not do quant";
      continue;
    }
    auto primitiveT_value = GetValueNode<std::shared_ptr<PrimitiveTValue>>(cnode->input(0));
    if (primitiveT_value == nullptr) {
      MS_LOG(ERROR) << "PrimitiveT_value is nullptr";
      continue;
    }
    if (input_scale.find(cnode) == input_scale.end()) {
      primitiveT_value->SetQuantType(schema::QuantType_QUANT_NONE);
      continue;
    }
    auto input_vec = cnode->inputs();
    auto op_name = cnode->fullname_with_scope();
    auto op_type = primitiveT_value->GetPrimitiveT()->value.type;
    MS_LOG(INFO) << "OpName: " << op_name;
    if (op_type != PrimitiveType_Conv2D && op_type != PrimitiveType_DepthwiseConv2D) {
      for (auto i = 1; i < cnode->inputs().size(); i++) {
        auto input_node = cnode->input(i);
        if (!input_node->isa<mindspore::CNode>()) {
          MS_LOG(WARNING) << "node: " << cnode_name << " input " << i << " not a cnode";
          continue;
        }
        auto input_cnode = std::dynamic_pointer_cast<mindspore::CNode>(input_node);
        auto input_cnode_primitiveT_value = GetValueNode<std::shared_ptr<PrimitiveTValue>>(input_cnode->input(0));
        if (input_cnode_primitiveT_value == nullptr) {
          MS_LOG(DEBUG) << "input: " << i << " " << input_cnode->fullname_with_scope() << ": "
                        << " PrimitiveTValue is null";
          continue;
        }
        for (auto &quant_param : input_cnode_primitiveT_value->GetOutputQuantParams()) {
          primitiveT_value->AddInputQuantParam(quant_param);
        }
      }
    } else {
      // do input quant
      double scale = input_scale[cnode];
      int32_t convInputzeropoint = input_zero_point[cnode];
      DoQuantInput(scale, convInputzeropoint, &input_min_max[cnode], primitiveT_value);
      // do weight quant
      auto weight = cnode->input(2);
      DoWeightQuant(weight);
      // do bias quant
      if (cnode->inputs().size() == 4) {
        auto bias = cnode->input(3);
        DoBiasQuant(primitiveT_value, weight, bias);
      }
    }
    // do output quant
    double OutputScale = output_scale[cnode];
    int32_t OutputZeropoint = output_zeropoint[cnode];
    DoQuantOutput(OutputScale, OutputZeropoint, &output_min_max[cnode], primitiveT_value);
    primitiveT_value->SetQuantType(schema::QuantType_PostTraining);
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
  for (auto cnode : cnodes) {
    AnfNodePtr anf = std::dynamic_pointer_cast<AnfNode>(cnode);
    if (strategy.CanOpPostQuantized(anf)) {
      MS_LOG(INFO) << "node: " << cnode->fullname_with_scope() << " will be quantized";
      calibrator_->AddQuantizedOp(cnode);
    }
  }
  return RET_OK;
}

STATUS PostTrainingQuantizer::CheckTensorVec(const std::string &nodeName,
                                             const std::vector<mindspore::tensor::MSTensor *> &tensorVec) const {
  if (tensorVec.size() < 1) {
    MS_LOG(ERROR) << "node: " << nodeName << " input tensors is 0";
    return RET_ERROR;
  }
  auto *tensor = tensorVec[0];
  if (tensor->data_type() != kNumberTypeFloat32) {
    //&& tensor->RefCount() != MSCONST_WEIGHT_REFCOUNT
    MS_LOG(DEBUG) << "node: " << nodeName << " will not quantize"
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
    // TODO(x) when model has inputs count > 1
    // get input tensor
    vector<mindspore::tensor::MSTensor *> inputs = session_->GetInputs();
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
      if (PostTrainingQuantizer::CheckTensorVec(callParam.name_callback_param, beforeInputs) != RET_OK) {
        return false;
      }
      auto tensor = beforeInputs[0];
      const float *tData = static_cast<const float *>(tensor->MutableData());
      size_t shapeSize = tensor->ElementsNum();
      vector<float> data(tData, tData + shapeSize);
      this->calibrator_->RecordMaxValue(callParam.name_callback_param, data, this->calibrator_->GetInputDivergInfo());
      return true;
    };
    // func
    mindspore::session::KernelCallBack afterCallBack = [&](
                                                         const std::vector<mindspore::tensor::MSTensor *> &afterInputs,
                                                         const std::vector<mindspore::tensor::MSTensor *> &afterOutputs,
                                                         const mindspore::session::CallBackParam &callParam) -> bool {
      if (PostTrainingQuantizer::CheckTensorVec(callParam.name_callback_param, afterOutputs) != RET_OK) {
        return false;
      }
      auto tensor = afterOutputs[0];
      const float *tensor_data = static_cast<const float *>(tensor->MutableData());
      size_t shape_size = tensor->ElementsNum();
      vector<float> data(tensor_data, tensor_data + shape_size);
      this->calibrator_->RecordMaxValue(callParam.name_callback_param, data, this->calibrator_->GetOutputDivergInfo());
      return true;
    };
    status = session_->RunGraph(beforeCallBack, afterCallBack);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "run model failed!";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS PostTrainingQuantizer::CollectDataFrequency() {
  for (size_t i = 0; i < calibrator_->GetBatchNum(); i++) {
    // TODO(x) when model has inputs count > 1
    // get input tensor
    vector<mindspore::tensor::MSTensor *> inputs = session_->GetInputs();
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
        if (PostTrainingQuantizer::CheckTensorVec(callParam.name_callback_param, beforeInputs) != RET_OK) {
          return false;
        }
        auto tensor = beforeInputs[0];
        const float *tensor_data = static_cast<const float *>(tensor->MutableData());
        size_t shape_size = tensor->ElementsNum();
        vector<float> data(tensor_data, tensor_data + shape_size);
        this->calibrator_->UpdateDataFrequency(callParam.name_callback_param, data, tensor->shape(),
                                               this->calibrator_->GetInputDivergInfo());
        return true;
      };

    mindspore::session::KernelCallBack afterCallBack =
      [&](const std::vector<mindspore::tensor::MSTensor *> &after_inputs,
          const std::vector<mindspore::tensor::MSTensor *> &after_outputs,
          const mindspore::session::CallBackParam &call_param) {
        if (PostTrainingQuantizer::CheckTensorVec(call_param.name_callback_param, after_outputs) != RET_OK) {
          return false;
        }
        auto tensor = after_outputs[0];
        const float *tenosr_data = static_cast<const float *>(tensor->MutableData());
        size_t shape_size = tensor->ElementsNum();
        vector<float> data(tenosr_data, tenosr_data + shape_size);
        this->calibrator_->UpdateDataFrequency(call_param.name_callback_param, data, tensor->shape(),
                                               this->calibrator_->GetOutputDivergInfo());
        return true;
      };
    status = session_->RunGraph(beforeCallBack, afterCallBack);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "run model failed!";
      return RET_ERROR;
    }
  }

  return RET_OK;
}

STATUS PostTrainingQuantizer::ComputeThreshold() { return this->calibrator_->ComputeThreshold(); }

STATUS PostTrainingQuantizer::DoQuantize(FuncGraphPtr funcGraph) {
  MS_LOG(INFO) << "start to parse config file";
  STATUS status = PreProcess();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "do pre process failed!";
    return status;
  }

  // anf -- fb
  auto meta_graph = Export(funcGraph);
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
  ctx.device_ctx_.type = DT_CPU;
  ctx.thread_num_ = calibrator_->GetThreadNum();
  ctx.cpu_bind_mode_ = MID_CPU;

  session_ = dynamic_cast<mindspore::lite::LiteSession *>(session::LiteSession::CreateSession(&ctx));
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "create session failed!";
    return RET_ERROR;
  }

  auto ret = session_->CompileGraph(model);
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
  return RET_OK;
}
}  // namespace quant
}  // namespace lite
}  // namespace mindspore
