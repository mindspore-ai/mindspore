/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "tools/converter/quantizer/cle_strategy.h"
#include <memory>
#include <map>
#include <vector>
#include <set>
#include <algorithm>
#include <string>
#include "tools/converter/quantizer/cle_pattern.h"
#include "include/errorcode.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/lite_exporter/fetch_content.h"
#include "ops/fusion/conv2d_fusion.h"
#include "src/common/log_util.h"
#include "tools/common/statistic_utils.h"
#include "tools/converter/quantizer/quantize_util.h"

namespace mindspore::lite::quant {
using lite::RET_ERROR;
using lite::RET_OK;
static const float kDefaultScale = 1;
int CLEStrategy::Run() {
  MS_LOG(INFO) << "CLE start to find pattern.";
  auto ret = FindPattern();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Find pattern failed.";
    return ret;
  }
  MS_LOG(INFO) << "CLE start to adjust weight.";
  ret = WeightEqualization();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Weight equalization failed.";
    return ret;
  }
  return RET_OK;
}

int CLEStrategy::FindPattern() {
  cle_pattern_ = new (std::nothrow) CLEPattern();
  CHECK_NULL_RETURN(cle_pattern_);
  (void)static_cast<opt::Pass *>(cle_pattern_)->Run(func_graph_);
  return RET_OK;
}

int CLEStrategy::CalcDataRange(const float *data, size_t element_cnt, const std::vector<int> &dims, int preferred_dim,
                               std::vector<float> *ranges) {
  CHECK_NULL_RETURN(data);
  CHECK_NULL_RETURN(ranges);
  std::vector<std::vector<size_t>> buckets_data_index;
  auto ret = GetBucketAllIndex(dims, preferred_dim, &buckets_data_index);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get bucket all index failed.";
    return ret;
  }
  auto bucket_count = dims[preferred_dim];
  ranges->resize(bucket_count);
  // ABS MAX
  for (int i = 0; i < bucket_count; i++) {
    auto bucket = buckets_data_index.at(i);
    float range = 0;
    for (size_t j = 0; j < bucket.size(); ++j) {
      auto index = bucket[j];
      MS_CHECK_LT(index, element_cnt, RET_ERROR);
      range = std::max(range, std::abs(data[index]));
    }
    ranges->at(i) = range;
  }
  return RET_OK;
}

int CLEStrategy::CalcRange(const CNodePtr &cnode, std::vector<float> *ranges, int preferred_dim) {
  CHECK_NULL_RETURN(ranges);
  CHECK_NULL_RETURN(cnode);
  auto node_name = cnode->fullname_with_scope();
  size_t weight_index = 2;
  DataInfo data_info;
  auto ret = FetchConstData(cnode, weight_index, converter::kFmkTypeMs, &data_info, false);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << node_name << " fetch data failed";
    return ret;
  }

  auto data = static_cast<float *>(data_info.data_ptr_);
  if (data == nullptr) {
    MS_LOG(ERROR) << node_name << " data is nullptr. ";
    return RET_ERROR;
  }
  int element_cnt = 1;
  ret = GetElementNumFromShape(data_info.shape_, &element_cnt);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << node_name << " get element num from shape failed.";
    return ret;
  }
  ret = CalcDataRange(data, element_cnt, data_info.shape_, preferred_dim, ranges);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << node_name << " Calc data range failed.";
    return ret;
  }
  return RET_OK;
}

CLEStrategy::~CLEStrategy() {
  if (cle_pattern_ != nullptr) {
    delete cle_pattern_;
    cle_pattern_ = nullptr;
  }
}

int CLEStrategy::CalcScaleWithTwoLayer(const CombinationLayer &layer_group, std::vector<double> *scales) {
  CHECK_NULL_RETURN(scales);
  std::vector<float> range1;
  std::vector<float> range2;
  int channel_in = 3;
  int channel_out = 0;
  auto ret = CalcRange(layer_group.layer1, &range1, channel_out);
  if (ret != RET_OK && ret != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "Calc range1 failed.";
    return ret;
  }
  ret = CalcRange(layer_group.layer2, &range2, channel_in);
  if (ret != RET_OK && ret != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "Calc range2 failed.";
    return ret;
  }
  if (range1.size() != range2.size()) {
    MS_LOG(ERROR) << layer_group.layer1->fullname_with_scope() << " range1.size:" << range1.size() << " "
                  << layer_group.layer2->fullname_with_scope() << " range2.size:" << range2.size();
    return RET_ERROR;
  }
  for (size_t i = 0; i < range1.size(); i++) {
    // scale = range1 / sqrt(range1 * range2)
    MS_ASSERT(range1.at(i) * range2.at(i) >= 0);
    auto sqrt_range = sqrt(range1.at(i) * range2.at(i));
    if (sqrt_range <= 0 || isnan(sqrt_range) || isinf(sqrt_range)) {
      MS_LOG(WARNING) << "sqrt_range <= 0, and set scale factor to default." << kDefaultScale;
      scales->push_back(kDefaultScale);
      MS_LOG(INFO) << layer_group.layer1->fullname_with_scope() << " " << 1 << " "
                   << layer_group.layer2->fullname_with_scope() << " " << 1;
      continue;
    }
    auto scale = range1.at(i) / sqrt_range;
    scale = scale <= 0 ? kDefaultScale : scale;
    MS_LOG(INFO) << layer_group.layer1->fullname_with_scope() << " " << 1 / scale << " "
                 << layer_group.layer2->fullname_with_scope() << " " << scale;
    scales->push_back(scale);
  }
  return RET_OK;
}

int CLEStrategy::CalcScaleWithThreeLayer(const CombinationLayer &layer_group, std::vector<double> *scales12,
                                         std::vector<double> *scales23) {
  CHECK_NULL_RETURN(scales12);
  CHECK_NULL_RETURN(scales23);
  std::vector<float> range1;
  std::vector<float> range2;
  std::vector<float> range3;
  int channel_in = 3;
  int channel_out = 0;
  auto ret = CalcRange(layer_group.layer1, &range1, channel_out);
  if (ret != RET_OK && ret != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "Calc range1 failed.";
    return ret;
  }
  ret = CalcRange(layer_group.layer2, &range2, channel_out);
  if (ret != RET_OK && ret != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "Calc range2 failed.";
    return ret;
  }
  ret = CalcRange(layer_group.layer3, &range3, channel_in);
  if (ret != RET_OK && ret != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "Calc range2 failed.";
    return ret;
  }
  if (range1.size() != range2.size() || range1.size() != range3.size()) {
    MS_LOG(ERROR) << layer_group.layer1->fullname_with_scope() << " range1.size:" << range1.size() << " "
                  << layer_group.layer2->fullname_with_scope() << " range2.size:" << range2.size() << " "
                  << layer_group.layer3->fullname_with_scope() << " range3.size:" << range3.size() << " ";
    return RET_ERROR;
  }
  // compute scale1 and scale2 using :
  // scale1 = range1/cubeRoot(range1 * range2 * range3)
  // scale2 = cubeRoot(range1 * range2 * range3)/range3
  for (size_t i = 0; i < range1.size(); i++) {
    auto cube_root = std::pow(range1.at(i) * range2.at(i) * range3.at(i), 1.0f / 3);
    if (cube_root <= 0 || isnan(cube_root) || isinf(cube_root)) {
      MS_LOG(WARNING) << "cube_root <= 0, and set scale factor to default." << kDefaultScale;
      scales12->push_back(kDefaultScale);
      scales23->push_back(kDefaultScale);
      MS_LOG(INFO) << layer_group.layer1->fullname_with_scope() << " " << 1 << " "
                   << layer_group.layer2->fullname_with_scope() << " " << 1 << " "
                   << layer_group.layer3->fullname_with_scope() << " " << 1;
      continue;
    }
    auto scale1 = range1.at(i) / cube_root;
    scale1 = scale1 <= 0 ? kDefaultScale : scale1;
    scales12->push_back(scale1);
    auto scale2 = cube_root / range3.at(i);
    scale2 = scale2 <= 0 ? kDefaultScale : scale2;
    scales23->push_back(scale2);
    MS_LOG(INFO) << layer_group.layer1->fullname_with_scope() << " " << 1 / scale1 << " "
                 << layer_group.layer2->fullname_with_scope() << " " << scale1 / scale2 << " "
                 << layer_group.layer3->fullname_with_scope() << " " << 1 / scale2;
  }
  return RET_OK;
}

int CLEStrategy::WeightEqualization() {
  auto combination_layer_groups = cle_pattern_->GetCombinationLayer();
  for (auto it = combination_layer_groups.rbegin(); it != combination_layer_groups.rend(); ++it) {
    auto group = *it;
    if (it->layer_num == kInputsNum2) {
      std::vector<double> scales;
      auto ret = CalcScaleWithTwoLayer(group, &scales);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Calc scale with two layer failed.";
        return ret;
      }
      ret = EqualizationWithTwoLayer(group, scales);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "weight equalization with two layer failed.";
        return ret;
      }
    } else if (it->layer_num == kInputsNum3) {
      std::vector<double> scales12;
      std::vector<double> scales23;
      auto ret = CalcScaleWithThreeLayer(group, &scales12, &scales23);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Calc scale with two layer failed.";
        return ret;
      }
      ret = EqualizationWithThreeLayer(group, scales12, scales23);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "weight equalization with two layer failed.";
        return ret;
      }
    } else {
      MS_LOG(ERROR) << "Dont support layer_num:" << it->layer_num;
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int CLEStrategy::EqualizationAdjust(const CNodePtr &cnode, const std::vector<double> &scales, size_t input_index,
                                    int preferred_dim, bool multiplication) {
  MS_ASSERT(scales != nullptr);
  auto weight = cnode->input(input_index);
  DataInfo layer1_data_info;
  int status = FetchConstData(cnode, input_index, converter::kFmkTypeMs, &layer1_data_info, false);
  if (status != RET_OK) {
    MS_LOG(ERROR) << weight->fullname_with_scope() << " fetch data failed";
    return status;
  }

  auto raw_datas = static_cast<float *>(layer1_data_info.data_ptr_);
  CHECK_NULL_RETURN(raw_datas);
  auto dims = layer1_data_info.shape_;
  int elem_count = 1;
  auto ret = GetElementNumFromShape(layer1_data_info.shape_, &elem_count);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get element num from shape failed.";
    return ret;
  }

  for (int i = 0; i < elem_count; i++) {
    auto raw_data = raw_datas[i];
    auto bucket_index = GetBucketIndex(dims, preferred_dim, i);
    MS_CHECK_GT(static_cast<int>(scales.size()), bucket_index, RET_ERROR);
    if (multiplication) {
      raw_datas[i] = raw_data * scales.at(bucket_index);
    } else {
      raw_datas[i] = raw_data / scales.at(bucket_index);
    }
  }
  return RET_OK;
}

int CLEStrategy::EqualizationWithTwoLayer(const CombinationLayer &layer_group, const std::vector<double> &scales) {
  auto layer1 = layer_group.layer1;
  auto layer2 = layer_group.layer2;
  auto layer1_name = layer1->fullname_with_scope();
  auto layer2_name = layer2->fullname_with_scope();
  size_t weight_index = 2;
  size_t bias_index = 3;
  int channel_in = 3;
  int channel_out = 0;
  auto ret = EqualizationAdjust(layer1, scales, weight_index, channel_out, false);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << layer1_name << " Equalization weight failed.";
    return ret;
  }
  if (layer1->size() > bias_index) {
    ret = EqualizationAdjust(layer1, scales, bias_index, channel_out, false);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << layer1_name << " Equalization bias failed.";
      return ret;
    }
  }
  ret = EqualizationAdjust(layer2, scales, weight_index, channel_in, true);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << layer2_name << " Equalization bias failed.";
    return ret;
  }
  return RET_OK;
}

int CLEStrategy::EqualizationWithThreeLayer(const CombinationLayer &layer_group, const std::vector<double> &scales12,
                                            const std::vector<double> &scales23) {
  auto layer1 = layer_group.layer1;
  auto layer2 = layer_group.layer2;
  auto layer3 = layer_group.layer3;
  CHECK_NULL_RETURN(layer1);
  CHECK_NULL_RETURN(layer2);
  CHECK_NULL_RETURN(layer3);
  auto layer1_name = layer1->fullname_with_scope();
  auto layer2_name = layer2->fullname_with_scope();
  auto layer3_name = layer3->fullname_with_scope();
  size_t weight_index = 2;
  size_t bias_index = 3;
  int channel_in = 3;
  int channel_out = 0;
  // EqualizationAdjust with scales12
  auto ret = EqualizationAdjust(layer1, scales12, weight_index, channel_out, false);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << layer1_name << " Equalization weight failed.";
    return ret;
  }
  if (layer1->size() > bias_index) {
    ret = EqualizationAdjust(layer1, scales12, bias_index, channel_out, false);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << layer1_name << " Equalization bias failed.";
      return ret;
    }
  }
  ret = EqualizationAdjust(layer2, scales12, weight_index, channel_out, true);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << layer2_name << " Equalization weight failed.";
    return ret;
  }
  // EqualizationAdjust with scales23
  ret = EqualizationAdjust(layer2, scales23, weight_index, channel_out, false);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << layer2_name << " Equalization weight failed.";
    return ret;
  }
  if (layer2->size() > bias_index) {
    ret = EqualizationAdjust(layer2, scales23, bias_index, channel_out, false);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << layer2_name << " Equalization bias failed.";
      return ret;
    }
  }
  ret = EqualizationAdjust(layer3, scales23, weight_index, channel_in, true);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << layer3_name << " Equalization weight failed.";
    return ret;
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
