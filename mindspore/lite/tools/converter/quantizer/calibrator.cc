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

#include "tools/converter/quantizer/calibrator.h"
#include <utility>
#include "tools/converter/preprocess/image_preprocess.h"
#include "tools/converter/ops/ops_def.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"

namespace mindspore::lite::quant {
namespace {
constexpr int kDefaultBinNumber = 2048;
}
int Calibrator::RecordMaxMinValue(const std::vector<float> &data, const std::unique_ptr<DivergInfo> &diverg_info) {
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

int Calibrator::ComputeThreshold() {
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
      auto cnode = input_infos[i]->GetCNode();
      bool already_computed = false;
      auto input = cnode->input(i + 1);
      if (input->isa<mindspore::CNode>()) {
        auto input_cnode = input->cast<CNodePtr>();
        for (const auto &outputs_diverg_info : outputs_diverg_info_) {
          if (already_computed) {
            break;
          }
          for (const auto &output_diverg_info : outputs_diverg_info.second) {
            auto output_diverg_cnode = output_diverg_info->GetCNode();
            if (output_diverg_cnode == input_cnode) {
              if (NodePrimitiveType(input_cnode) != lite::kNameTupleGetItem) {
                *(input_infos[i]) = *output_diverg_info;
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

int Calibrator::UpdateDivergInterval(
  std::unordered_map<std::string, std::vector<std::unique_ptr<DivergInfo>>> *diverg_info) {
  MS_ASSERT(diverg_info != nullptr);
  for (auto &kv : *diverg_info) {
    for (auto &info : kv.second) {
      info->UpdateInterval();
    }
  }
  return RET_OK;
}

int Calibrator::UpdateDataFrequency(const std::vector<float> &data, const std::unique_ptr<DivergInfo> &diverg_info) {
  MS_ASSERT(diverg_info != nullptr);
  return diverg_info->UpdateHistogram(data);
}

int Calibrator::AddQuantizedOp(const CNodePtr &cnode) {
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "To be quantized cnode is null";
    return RET_ERROR;
  }
  auto node_name = cnode->fullname_with_scope();
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

int Calibrator::GenerateInputData(const std::string &input_name, size_t image_index,
                                  mindspore::tensor::MSTensor *tensor) const {
  return preprocess::PreProcess(data_pre_process_param_, input_name, image_index, tensor);
}

std::unordered_map<std::string, std::vector<std::unique_ptr<DivergInfo>>> *Calibrator::GetInputDivergInfo() {
  return &this->inputs_diverg_info_;
}

std::unordered_map<std::string, std::vector<std::unique_ptr<DivergInfo>>> *Calibrator::GetOutputDivergInfo() {
  return &this->outputs_diverg_info_;
}
}  // namespace mindspore::lite::quant
