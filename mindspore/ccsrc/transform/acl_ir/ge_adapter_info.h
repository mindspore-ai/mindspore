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

#ifndef MINDSPORE_CCSRC_TRANSFORM_ACL_IR_GE_ADAPTER_INFO_H_
#define MINDSPORE_CCSRC_TRANSFORM_ACL_IR_GE_ADAPTER_INFO_H_

#include <map>
#include <string>
#include <memory>
#include <utility>
#include <vector>
#include <unordered_map>
#include "ir/anf.h"
#include "ir/tensor.h"
#include "utils/hash_map.h"
#include "graph/op_desc.h"
#include "include/transform/graph_ir/types.h"
#include "transform/graph_ir/op_adapter_base.h"
#include "mindapi/base/shape_vector.h"

namespace mindspore::transform {
using TensorPtr = mindspore::tensor::TensorPtr;

struct ValuePairHasher {
  template <typename T>
  size_t operator()(const std::pair<T, ValuePtr> &p) const {
    auto hash_value = hash_combine(std::hash<T>()(p.first), PointerHash<ValuePtr>{}(p.second));
    return hash_value;
  }
};

struct Ms2GeParamInfo {
  enum ParamType : uint8_t { REQUIRED, OPTIONAL, DYNAMIC };

  uint32_t index;
  std::string name;
  enum ParamType type;
  bool is_after_dynamic = false;
};

struct GeTensorInfo {
  std::string op_type;

  // Attr
  mindspore::HashMap<std::string, std::string> attr_map;
  mindspore::HashMap<uint32_t, std::string> input_attr_map;
  mindspore::HashMap<size_t, std::string> attr_input_map;

  // Input/Output
  enum ParamMappingFlag : unsigned int {
    kDynamicParam = 1 << 0,  // has dynamic input/output
    kEmptyParam = 1 << 1     // empty input/output
  };

  // map input/output indices of operator from MindSpore frontend to GraphEngine backend
  // K: MindSpore operator input index, V: GE operator input index and type info
  mindspore::HashMap<int, Ms2GeParamInfo> input_idx_ms2ge;
  mindspore::HashMap<int, Ms2GeParamInfo> output_idx_ms2ge;
  std::unordered_map<size_t, size_t> ref_map_;
  // fields for recording the mapping flags of input/output
  unsigned int input_mapping_flags = 0;
  unsigned int output_mapping_flags = 0;
  // map input/output indices of operator from GraphEngine backend to MindSpore frontend
  // K: GE operator input index, V: MindSpore operator input index
  mindspore::HashMap<size_t, int> input_idx_ge2ms;
  mindspore::HashMap<size_t, int> output_idx_ge2ms;

  // DataType
  mindspore::HashMap<int, std::vector<enum ::ge::DataType>> input_supported_dtypes;
  mindspore::HashMap<int, std::vector<enum ::ge::DataType>> output_supported_dtypes;
};

class GeAdapterInfo {
 public:
  explicit GeAdapterInfo(OpAdapterPtr adpt) : adapter_(std::move(adpt)) {}
  ~GeAdapterInfo() = default;

  void InitInfo();

  const std::string &op_type() const { return info_.op_type; }
  const mindspore::HashMap<std::string, std::string> &attr_map() const { return info_.attr_map; }
  const mindspore::HashMap<uint32_t, std::string> &input_attr_map() const { return info_.input_attr_map; }
  const mindspore::HashMap<size_t, std::string> &attr_input_map() const { return info_.attr_input_map; }

  // Get number of inputs in mindspore operator prototype, not the real number of inputs
  size_t GetNumInputsOfMsOpProto() const {
    // Note: number of ms operator inputs(not real inputs) is equal to size of info_.input_idx_ms2ge
    return info_.input_idx_ms2ge.size();
  }

  const Ms2GeParamInfo &GetGeInputByMsInputIndex(size_t ms_input_idx) const {
    auto iter = info_.input_idx_ms2ge.find(ms_input_idx);
    if (iter == info_.input_idx_ms2ge.end()) {
      MS_LOG(EXCEPTION) << "Find input info from GE operator " << info_.op_type << " for mindspore input index "
                        << ms_input_idx << " fail.";
    }
    return iter->second;
  }

  const std::optional<Ms2GeParamInfo> GetOptGeInputByMsInputIndex(size_t ms_input_idx) const {
    auto iter = info_.input_idx_ms2ge.find(ms_input_idx);
    if (iter != info_.input_idx_ms2ge.end()) {
      return iter->second;
    }
    return std::nullopt;
  }

  // Get number of outputs in mindspore operator prototype, not the real number of outputs
  size_t GetNumOutputsOfMsOpProto() const {
    // Note: number of ms operator outputs(not real outputs) is equal to size of info_.output_idx_ms2ge
    return info_.output_idx_ms2ge.size();
  }

  const Ms2GeParamInfo GetGeOutputByMsOutputIndex(size_t ms_output_idx) const {
    auto iter = info_.output_idx_ms2ge.find(ms_output_idx);
    if (iter == info_.output_idx_ms2ge.end()) {
      MS_LOG(EXCEPTION) << "Find output info from GE operator " << info_.op_type << " for mindspore output index "
                        << ms_output_idx << " fail.";
    }
    return iter->second;
  }

  const std::optional<Ms2GeParamInfo> GetOptGeOutputByMsOutputIndex(size_t ms_output_idx) const {
    auto iter = info_.output_idx_ms2ge.find(ms_output_idx);
    if (iter != info_.output_idx_ms2ge.end()) {
      return iter->second;
    }
    return std::nullopt;
  }

  unsigned int GetInputMappingFlags() const { return info_.input_mapping_flags; }

  unsigned int GetOutputMappingFlags() const { return info_.output_mapping_flags; }

  const std::unordered_map<size_t, size_t> &GetRefMappingInfo() const { return info_.ref_map_; }

  mindspore::HashMap<int, std::vector<enum ::ge::DataType>> input_supported_dtypes() const {
    return info_.input_supported_dtypes;
  }
  mindspore::HashMap<int, std::vector<enum ::ge::DataType>> output_supported_dtypes() const {
    return info_.output_supported_dtypes;
  }
  void GetGeAttrValueByMsAttrValue(const std::string &attr_name, const ValuePtr &ms_value, ValuePtr *ge_value);
  void GetGeAttrValueByMsInputValue(const uint32_t &input_idx, const ValuePtr &ms_value, ValuePtr *ge_value);

 private:
  void InitOpType();

  void InitAclInputsAndOutputs();
  void InitRefMap();
  template <typename ParamMap, typename DynParamMap>
  void InitParametersMap(const ParamMap &params, const DynParamMap &dyn_params, bool is_input);

  // attr
  void InitAttrMap();
  void InitInputToAttrMap();
  void InitAttrToInputMap();

  void InitInputSupportedDataType();
  void InitOutputSupportedDataType();

  OpAdapterPtr adapter_{nullptr};
  GeTensorInfo info_;
  std::unordered_map<std::pair<std::string, ValuePtr>, ValuePtr, ValuePairHasher> get_attr_cache_;
  std::unordered_map<std::pair<uint32_t, ValuePtr>, ValuePtr, ValuePairHasher> get_input_attr_cache_;
};

using GeAdapterInfoPtr = std::shared_ptr<GeAdapterInfo>;

class GeAdapterManager {
 public:
  static GeAdapterManager &GetInstance();
  GeAdapterInfoPtr GetInfo(const std::string &prim_name, bool is_training);

 private:
  GeAdapterManager() = default;
  ~GeAdapterManager() = default;
  mindspore::HashMap<std::string, GeAdapterInfoPtr> op_cache_;
  std::mutex lock_;
};
}  // namespace mindspore::transform

#endif  // MINDSPORE_CCSRC_TRANSFORM_ACL_IR_GE_ADAPTER_INFO_H_
