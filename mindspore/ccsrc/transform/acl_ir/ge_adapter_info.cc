/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "transform/acl_ir/ge_adapter_info.h"
#include <limits>
#include <utility>
#include <vector>
#include "include/transform/graph_ir/utils.h"
#include "transform/graph_ir/transform_util.h"
#include "graph/operator_factory.h"
#include "graph/utils/op_desc_utils.h"

namespace mindspore {
namespace transform {
void GeAdapterInfo::InitOpType() { info_.op_type = adapter_->getOpType(); }

void GeAdapterInfo::InitAclInputsAndOutputs() {
  InitParametersMap(adapter_->getInputMap(), adapter_->getDynInputMap(), true);
  InitParametersMap(adapter_->getOutputMap(), adapter_->getDynOutputMap(), false);
}

void GeAdapterInfo::InitRefMap() {
  for (const auto &[output_index, output_param_info] : info_.output_idx_ms2ge) {
    for (const auto &[input_index, input_param_info] : info_.input_idx_ms2ge) {
      if (output_param_info.name == input_param_info.name) {
        (void)info_.ref_map_.emplace(IntToSize(output_index), IntToSize(input_index));
        break;
      }
    }
  }
}

template <typename ParamMap, typename DynParamMap>
void GeAdapterInfo::InitParametersMap(const ParamMap &params, const DynParamMap &dyn_params, bool is_input) {
  auto &mapping_flags = is_input ? info_.input_mapping_flags : info_.output_mapping_flags;
  auto &idx_ms2ge = is_input ? info_.input_idx_ms2ge : info_.output_idx_ms2ge;
  auto &idx_ge2ms = is_input ? info_.input_idx_ge2ms : info_.output_idx_ge2ms;

  if (params.empty() && dyn_params.empty()) {
    mapping_flags |= GeTensorInfo::kEmptyParam;
    return;
  }

  // calculate index of dynamic input/output
  size_t ge_dynmaic_idx = std::numeric_limits<size_t>::max();
  if (!dyn_params.empty()) {
    // NOTE: Now only support one dynamic input or output
    if (dyn_params.size() > 1) {
      MS_LOG(EXCEPTION) << "Now only support op with one dynamic input/output, but op " << adapter_->getOpType()
                        << " has " << dyn_params.size() << " dynamic " << (is_input ? "inputs" : "outputs");
    }
    mapping_flags |= GeTensorInfo::kDynamicParam;
    ge_dynmaic_idx = dyn_params.cbegin()->second.index;
  }

  auto get_ms_idx = [is_input](int index) {
    // for anf cnode, the 1st input is primitive name, so for input the real input index is `index - 1`
    return is_input ? index - 1 : index;
  };

  // process required/optional inputs or required outputs
  for (const auto &[k, v] : params) {
    int ms_idx = get_ms_idx(k);
    uint32_t ge_idx = static_cast<uint32_t>(v.index);
    // MindSpore Index --> GE Info
    if constexpr (std::is_same<std::remove_cv_t<decltype(v)>, InputDesc>::value) {
      idx_ms2ge[ms_idx] = Ms2GeParamInfo{
        ge_idx, v.name, v.type == InputDesc::OPTIONAL ? Ms2GeParamInfo::OPTIONAL : Ms2GeParamInfo::REQUIRED,
        ge_idx > ge_dynmaic_idx};
    } else {
      idx_ms2ge[ms_idx] = Ms2GeParamInfo{ge_idx, v.name, Ms2GeParamInfo::REQUIRED, ge_idx > ge_dynmaic_idx};
    }

    // input/output: GE(GraphEngine) Index --> MindSpore Index
    idx_ge2ms[ge_idx] = ms_idx;
  }

  // process dynamic inputs/outputs
  for (const auto &[k, v] : dyn_params) {
    int ms_idx = get_ms_idx(k);
    uint32_t ge_idx = static_cast<uint32_t>(v.index);
    // MindSpore Index --> GE Info
    idx_ms2ge[ms_idx] = Ms2GeParamInfo{ge_idx, v.name, Ms2GeParamInfo::DYNAMIC, ge_idx > ge_dynmaic_idx};
    // input/output: GE(GraphEngine) Index --> MindSpore Index
    idx_ge2ms[ge_idx] = ms_idx;
  }
}

void GeAdapterInfo::InitInputSupportedDataType() {
  info_.input_supported_dtypes.clear();
  for (const auto &[k, v] : adapter_->getInputMap()) {
    (void)info_.input_supported_dtypes.emplace(k - 1, v.supported_dtypes);
  }
  for (const auto &[k, v] : adapter_->getDynInputMap()) {
    (void)info_.input_supported_dtypes.emplace(k - 1, v.supported_dtypes);
  }
}

void GeAdapterInfo::InitOutputSupportedDataType() {
  info_.output_supported_dtypes.clear();
  for (const auto &[k, v] : adapter_->getOutputMap()) {
    (void)info_.output_supported_dtypes.emplace(k, v.supported_dtypes);
  }
  for (const auto &[k, v] : adapter_->getDynOutputMap()) {
    (void)info_.output_supported_dtypes.emplace(k, v.supported_dtypes);
  }
}

void GeAdapterInfo::GetGeAttrValueByMsAttrValue(const std::string &attr_name, const ValuePtr &ms_value,
                                                ValuePtr *ge_value) {
  MS_EXCEPTION_IF_NULL(ge_value);
  // class Value is a abstract class
  auto iter = get_attr_cache_.find({attr_name, ms_value});
  if (iter != get_attr_cache_.end()) {
    *ge_value = iter->second;
    return;
  }

  int ret = 0;
  if (ms_value != nullptr) {
    ret = adapter_->setAttr(attr_name, ms_value);
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "failed to set attr:" << attr_name << " for primitive " << info_.op_type;
    }
  }

  ret = adapter_->getAttr(attr_name, ge_value);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "failed to get attr:" << attr_name << " for primitive " << info_.op_type;
  }
  get_attr_cache_[{attr_name, ms_value}] = *ge_value;
}

void GeAdapterInfo::GetGeAttrValueByMsInputValue(const uint32_t &input_idx, const ValuePtr &ms_value,
                                                 ValuePtr *ge_value) {
  MS_EXCEPTION_IF_NULL(ge_value);
  // class Value is a abstract class
  auto iter = get_input_attr_cache_.find({input_idx, ms_value});
  if (iter != get_input_attr_cache_.end()) {
    *ge_value = iter->second;
    return;
  }

  int ret = 0;
  ret = adapter_->setAttr(input_idx, ms_value);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "failed to set attr from input[" << input_idx << "] for primitive " << info_.op_type;
  }
  ret = adapter_->getAttr(input_idx, ge_value);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "failed to get attr from input[" << input_idx << "] for primitive " << info_.op_type;
  }
  get_input_attr_cache_[{input_idx, ms_value}] = *ge_value;
}

void GeAdapterInfo::InitAttrMap() {
  info_.attr_map.clear();
  for (const auto &[k, v] : adapter_->getAttrMap()) {
    (void)info_.attr_map.emplace(k, v.name);
  }
}

void GeAdapterInfo::InitInputToAttrMap() {
  info_.input_attr_map.clear();
  for (const auto &[k, v] : adapter_->getInputAttrMap()) {
    (void)info_.input_attr_map.emplace(k - 1, v.name);
  }
}

void GeAdapterInfo::InitAttrToInputMap() {
  auto attr_input_map = adapter_->getAttrInputMap();
  auto input_map = adapter_->getInputMap();
  for (const auto &[ms_attr_name, ge_input_name] : attr_input_map) {
    const auto &ge_input_name_for_cpp17 = ge_input_name;
    auto iter = std::find_if(input_map.begin(), input_map.end(), [&ge_input_name_for_cpp17](const auto &desc) {
      return desc.second.name == ge_input_name_for_cpp17;
    });
    if (iter == input_map.end()) {
      MS_LOG(EXCEPTION) << "Error adapter register of" << ms_attr_name << " and " << ge_input_name
                        << ", type: " << adapter_->getOpType();
    }
    (void)info_.attr_input_map.emplace(IntToSize(iter->first - 1), ms_attr_name);
  }
}

void GeAdapterInfo::InitInfo() {
  InitOpType();

  InitInputSupportedDataType();
  InitOutputSupportedDataType();

  InitAttrMap();
  InitInputToAttrMap();
  InitAttrToInputMap();

  InitAclInputsAndOutputs();
  InitRefMap();
  MS_LOG(DEBUG) << "INIT INFO:" << info_.op_type << " -- " << info_.input_supported_dtypes[0] << " --- "
                << info_.output_supported_dtypes[0];
}

GeAdapterManager &GeAdapterManager::GetInstance() {
  static GeAdapterManager instance;
  return instance;
}

GeAdapterInfoPtr GeAdapterManager::GetInfo(const std::string &prim_name, bool is_training = true) {
  std::lock_guard<std::mutex> guard(lock_);
  auto iter = op_cache_.find(prim_name);
  if (iter != op_cache_.end()) {
    return iter->second;
  }

  OpAdapterPtr adpt = FindAdapter(prim_name, is_training);
  if (adpt == nullptr) {
    MS_LOG(DEBUG) << "The current name '" << prim_name << "' needs to add adapter.";
    return nullptr;
  }
  if (prim_name != adpt->getOpType()) {
    MS_LOG(DEBUG) << "Note: primitive name is difference with adapter: prim name: " << prim_name
                  << ", ge name: " << adpt->getOpType();
  }
  auto info_ptr = std::make_shared<GeAdapterInfo>(adpt);
  info_ptr->InitInfo();
  op_cache_[prim_name] = info_ptr;
  return info_ptr;
}
}  // namespace transform
}  // namespace mindspore
