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
#include "tools/converter/quantizer/cle_pattern.h"
#include "include/errorcode.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/anf_exporter/fetch_content.h"
#include "ops/fusion/conv2d_fusion.h"
#include "src/common/log_util.h"
#include "tools/common/statistic_utils.h"
#include "tools/converter/quantizer/quantize_util.h"

namespace mindspore::lite::quant {
using lite::RET_ERROR;
using lite::RET_OK;
int CLEStrategy::ReplaceGraphRelu6ToRelu() {
  // Optimize with pattern.
  auto cnodes = func_graph_->GetOrderedCnodes();
  for (const auto &cnode : cnodes) {
    auto ret = ReplaceCNodeRelu6ToRelu(cnode);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Replace " << cnode->fullname_with_scope() << " Relu6 to Relu failed.";
      return ret;
    }
  }
  return RET_OK;
}

int CLEStrategy::ReplaceCNodeRelu6ToRelu(const CNodePtr &cnode) {
  // Optimize with pattern.
  if (opt::CheckPrimitiveType(cnode, prim::kPrimConv2DFusion)) {
    auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
    auto name = cnode->fullname_with_scope();
    auto conv2d = api::MakeShared<ops::Conv2DFusion>(primitive);
    if (conv2d->get_activation_type() == RELU6) {
      MS_LOG(INFO) << name << " replace relu6 to relu.";
      conv2d->set_activation_type(RELU);
    }
  }
  return RET_OK;
}

int CLEStrategy::Run() {
  auto ret = FindPattern();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Find pattern failed.";
    return ret;
  }
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
  std::map<int, MinMax> per_channel_min_max;
  GetAllChannelMinMax(data, element_cnt, dims, preferred_dim, &per_channel_min_max);

  for (auto min_max_map : per_channel_min_max) {
    float min = min_max_map.second.min;
    float max = min_max_map.second.max;
    auto range = std::abs(min) > std::abs(max) ? std::abs(min) : std::abs(max);
    ranges->push_back(range);
  }
  return RET_OK;
}

int CLEStrategy::CalcRange(const CNodePtr &cnode, std::vector<float> *ranges, int preferred_dim) {
  CHECK_NULL_RETURN(ranges);
  CHECK_NULL_RETURN(cnode);
  size_t weight_index = 2;
  auto weight = cnode->input(weight_index);
  int status;
  DataInfo data_info;
  if (weight->isa<Parameter>()) {
    status = FetchDataFromParameterNode(cnode, weight_index, converter::kFmkTypeMs, &data_info, true);
  } else if (weight->isa<ValueNode>()) {
    status = FetchDataFromValueNode(cnode, weight_index, converter::kFmkTypeMs, false, &data_info, true);
  } else {
    return RET_NO_CHANGE;
  }

  if (status != RET_OK) {
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " fetch data failed";
    return status;
  }

  auto data = reinterpret_cast<float *>(data_info.data_.data());
  if (data == nullptr) {
    MS_LOG(ERROR) << "data is nullptr. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  size_t element_cnt = data_info.data_.size() / sizeof(float);
  status = CalcDataRange(data, element_cnt, data_info.shape_, preferred_dim, ranges);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Calc data range failed.";
    return status;
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
    MS_CHECK_GT(range1.at(i), 0, RET_ERROR);
    MS_CHECK_GT(range2.at(i), 0, RET_ERROR);
    auto scale = range1.at(i) * (1.0f / sqrt(range1.at(i) * range2.at(i)));
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
    MS_CHECK_GT(range1.at(i), 0, RET_ERROR);
    MS_CHECK_GT(range2.at(i), 0, RET_ERROR);
    MS_CHECK_GT(range3.at(i), 0, RET_ERROR);
    auto cube_root = std::pow(range1.at(i) * range2.at(i) * range3.at(i), 1.0f / 3);
    if (cube_root <= 0) {
      MS_LOG(ERROR) << "cube_root <= 0";
      return RET_ERROR;
    }
    auto scale1 = range1.at(i) / cube_root;
    scales12->push_back(scale1);
    auto scale2 = cube_root / range3.at(i);
    scales23->push_back(scale2);
  }
  return RET_OK;
}

int CLEStrategy::WeightEqualization() {
  auto combination_layer_groups = cle_pattern_->GetCombinationLayer();
  for (auto it = combination_layer_groups.rbegin(); it != combination_layer_groups.rend(); ++it) {
    auto group = *it;
    if (it->layer_num == kInputsNum2) {
      std::vector<double> scales;
      auto ret = ReplaceCNodeRelu6ToRelu(group.layer1);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Replace Relu6 to Relu failed.";
        return ret;
      }
      ret = CalcScaleWithTwoLayer(group, &scales);
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
      auto ret = ReplaceCNodeRelu6ToRelu(group.layer1);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Replace Relu6 to Relu failed.";
        return ret;
      }
      ret = ReplaceCNodeRelu6ToRelu(group.layer2);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Replace Relu6 to Relu failed.";
        return ret;
      }
      ret = CalcScaleWithThreeLayer(group, &scales12, &scales23);
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
  int status;
  if (weight->isa<Parameter>()) {
    status = FetchDataFromParameterNode(cnode, input_index, converter::kFmkTypeMs, &layer1_data_info, false);
  } else if (weight->isa<ValueNode>()) {
    status = FetchDataFromValueNode(cnode, input_index, converter::kFmkTypeMs, false, &layer1_data_info, false);
  } else {
    return RET_NO_CHANGE;
  }
  if (status != RET_OK) {
    MS_LOG(ERROR) << weight->fullname_with_scope() << " fetch data failed";
    return status;
  }

  auto raw_datas = static_cast<float *>(layer1_data_info.data_ptr_);
  auto dims = layer1_data_info.shape_;
  int elem_count = 1;
  for (size_t i = 0; i < dims.size(); ++i) {
    MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW(elem_count, dims.at(i)), RET_ERROR, "Int mul overflow.");
    elem_count *= dims.at(i);
  }

  for (int i = 0; i < elem_count; i++) {
    auto raw_data = raw_datas[i];
    auto bucket_index = GetBucketIndex(dims, preferred_dim, i);
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

int CLEStrategy::AbsorbingHighBias() { return RET_OK; }
}  // namespace mindspore::lite::quant
