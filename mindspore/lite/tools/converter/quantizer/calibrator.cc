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

#include "tools/converter/quantizer/calibrator.h"
#include <utility>
#include "tools/converter/preprocess/image_preprocess.h"
#include "ops/tuple_get_item.h"
#include "ops/core_ops.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"

namespace mindspore::lite::quant {
namespace {
constexpr int kDefaultBinNumber = 2048;
}  // namespace
int Calibrator::RecordMaxMinValue(const std::vector<float> &data,
                                  const std::unique_ptr<DataDistribution> &diverg_info) {
  auto ret = diverg_info->RecordMaxMinValueArray(data);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Record max min value array failed.";
    return ret;
  }
  return RET_OK;
}

int Calibrator::ComputeThreshold() {
  for (auto &kv : this->outputs_diverg_info_) {
    auto &outputs_diverg_info = kv.second;
    for (auto &diverg_info : outputs_diverg_info) {
      MS_CHECK_TRUE_RET(diverg_info.second != nullptr, RET_ERROR);
      auto ret = diverg_info.second->ComputeThreshold();
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
      auto cnode = input_infos[i]->GetCNode();
      MS_CHECK_TRUE_MSG(cnode != nullptr, RET_NULL_PTR, "cnode is nullptr.");
      bool already_computed = false;
      MS_CHECK_GT(cnode->size(), i + 1, RET_ERROR);
      auto input = cnode->input(i + 1);
      if (input->isa<mindspore::CNode>()) {
        auto input_cnode = input->cast<CNodePtr>();
        for (const auto &outputs_diverg_info : outputs_diverg_info_) {
          if (already_computed) {
            break;
          }
          for (const auto &output_diverg_info : outputs_diverg_info.second) {
            MS_CHECK_TRUE_RET(output_diverg_info.second != nullptr, RET_ERROR);
            auto output_diverg_cnode = output_diverg_info.second->GetCNode();
            if (output_diverg_cnode == input_cnode) {
              if (NodePrimitiveType(input_cnode) != ops::kNameTupleGetItem) {
                *(input_infos[i]) = *output_diverg_info.second;
                input_infos[i]->GetCNode() = cnode;
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

int Calibrator::UpdateDivergInterval() {
  for (auto &kv : inputs_diverg_info_) {
    for (auto &info : kv.second) {
      info.second->UpdateInterval();
    }
  }
  for (auto &kv : outputs_diverg_info_) {
    for (auto &info : kv.second) {
      info.second->UpdateInterval();
    }
  }
  return RET_OK;
}

int Calibrator::UpdateDataFrequency(const std::vector<float> &data,
                                    const std::unique_ptr<DataDistribution> &diverg_info) {
  MS_ASSERT(diverg_info != nullptr);
  return diverg_info->UpdateHistogram(data);
}

int Calibrator::AddQuantizedOp(const CNodePtr &cnode) {
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "To be quantized cnode is null";
    return RET_ERROR;
  }
  auto node_name = cnode->fullname_with_scope();
  auto input_size = cnode->inputs().size();
  int index = 0;
  for (size_t i = 1; i < input_size; i++) {
    if (opt::CheckPrimitiveType(cnode->input(i), prim::kPrimMakeTuple)) {
      auto input_cnode = cnode->input(i)->cast<CNodePtr>();
      MS_CHECK_TRUE_MSG(input_cnode != nullptr, RET_ERROR, "input_cnode is nullptr.");
      auto make_tuple_size = input_cnode->size() - 1;
      for (size_t j = 0; j < make_tuple_size; j++) {
        std::unique_ptr<DataDistribution> input_diverg = std::make_unique<DataDistribution>(
          cnode, kDefaultBinNumber, bit_num_, quant_max_, quant_min_, activation_quant_method_, symmetric_);
        MS_CHECK_TRUE_MSG(input_diverg != nullptr, RET_NULL_PTR, "input_diverg is nullptr.");
        inputs_diverg_info_[node_name].insert({index++, std::move(input_diverg)});
      }
    } else {
      std::unique_ptr<DataDistribution> input_diverg = std::make_unique<DataDistribution>(
        cnode, kDefaultBinNumber, bit_num_, quant_max_, quant_min_, activation_quant_method_, symmetric_);
      MS_CHECK_TRUE_MSG(input_diverg != nullptr, RET_NULL_PTR, "input_diverg is nullptr.");
      inputs_diverg_info_[node_name].insert({index++, std::move(input_diverg)});
    }
  }

  if (utils::isa<abstract::AbstractTuple>(cnode->abstract())) {
    auto tuple = std::reinterpret_pointer_cast<abstract::AbstractTuple>(cnode->abstract());
    MS_CHECK_TRUE_MSG(tuple != nullptr, RET_ERROR, "tuple is nullptr");
    auto elements = tuple->elements();
    MS_CHECK_GT(elements.size(), 1, RET_ERROR);
    for (size_t i = 0; i < elements.size(); i++) {
      std::unique_ptr<DataDistribution> output_diverg = std::make_unique<DataDistribution>(
        cnode, kDefaultBinNumber, bit_num_, quant_max_, quant_min_, activation_quant_method_, symmetric_);
      MS_CHECK_TRUE_MSG(output_diverg != nullptr, RET_NULL_PTR, "output_diverg is nullptr.");
      outputs_diverg_info_[node_name].insert({i, std::move(output_diverg)});
    }
  } else {
    std::unique_ptr<DataDistribution> output_diverg = std::make_unique<DataDistribution>(
      cnode, kDefaultBinNumber, bit_num_, quant_max_, quant_min_, activation_quant_method_, symmetric_);
    MS_CHECK_TRUE_MSG(output_diverg != nullptr, RET_NULL_PTR, "output_diverg is nullptr.");
    outputs_diverg_info_[node_name].insert({0, std::move(output_diverg)});
  }
  return RET_OK;
}

int Calibrator::GenerateInputData(const std::string &input_name, size_t image_index,
                                  mindspore::MSTensor *tensor) const {
  return preprocess::PreProcess(data_pre_process_param_, input_name, image_index, tensor);
}

int Calibrator::CollectDataDistribution(
  const std::string &node_name, const std::vector<mindspore::MSTensor> &tensors,
  std::unordered_map<std::string, std::map<int, std::unique_ptr<DataDistribution>>> *diverg_info_map,
  CollectType collect_type) {
  MS_CHECK_TRUE_MSG(diverg_info_map != nullptr, RET_ERROR, "diverg_info_map is nullptr.");
  if (diverg_info_map->find(node_name) == diverg_info_map->end()) {
    return RET_OK;
  }
  for (size_t i = 0; i < tensors.size(); i++) {
    auto tensor = tensors[i];
    if (tensor.IsConst() || tensor.DataType() != DataType::kNumberTypeFloat32) {
      continue;
    }
    const auto *tensor_data = static_cast<const float *>(tensor.Data().get());
    if (tensor_data == nullptr) {
      MS_LOG(ERROR) << tensor.Name() << " tensor_data is nullptr.";
      return RET_ERROR;
    }
    size_t elem_count = static_cast<size_t>(tensor.ElementNum());
    MS_CHECK_GT(elem_count, 0, RET_ERROR);
    std::vector<float> data(tensor_data, tensor_data + elem_count);
    if (collect_type == MIN_MAX) {
      MS_CHECK_LT(i, (*diverg_info_map)[node_name].size(), RET_ERROR);
      auto ret = RecordMaxMinValue(data, (*diverg_info_map)[node_name][i]);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << tensor.Name() << " record max min value failed.";
        return RET_ERROR;
      }
    } else if (collect_type == KL_BIN) {
      MS_CHECK_LT(i, (*diverg_info_map)[node_name].size(), RET_ERROR);
      auto ret = UpdateDataFrequency(data, (*diverg_info_map)[node_name][i]);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << tensor.Name() << " update data frequency failed.";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
