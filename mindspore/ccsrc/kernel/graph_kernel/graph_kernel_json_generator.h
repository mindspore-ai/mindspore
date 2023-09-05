/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_GRAPH_KERNEL_GRAPH_KERNEL_JSON_GENERATOR_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_GRAPH_KERNEL_GRAPH_KERNEL_JSON_GENERATOR_H_
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "nlohmann/json.hpp"
#include "kernel/oplib/opinfo.h"
#include "backend/common/graph_kernel/core/graph_kernel_callback.h"
#include "include/common/utils/convert_utils.h"
#include "backend/common/graph_kernel/symbol_engine/symbol_engine.h"

namespace mindspore::graphkernel {
using kernel::OpAttrPtr;
using kernel::OpInfoPtr;

// dump option
struct DumpOption {
  bool is_before_select_kernel = false;
  bool save_ptr_address = false;
  bool extract_opinfo_from_anfnode = false;
  bool get_target_info = false;
  bool gen_kernel_name_only = false;
};

class TargetInfoSetter {
 public:
  static void Set(nlohmann::json *kernel_info) {
    static std::unique_ptr<TargetInfoSetter> instance = nullptr;
    if (instance == nullptr) {
      instance = std::make_unique<TargetInfoSetter>();
      instance->GetTargetInfo();
    }
    instance->SetTargetInfo(kernel_info);
  }

 private:
  void GetTargetInfo();
  void SetTargetInfo(nlohmann::json *kernel_info) const;
  nlohmann::json target_info_;
  bool has_info_{true};
};

class GraphKernelJsonGenerator {
 public:
  GraphKernelJsonGenerator() : cb_(Callback::Instance()) {}
  explicit GraphKernelJsonGenerator(DumpOption dump_option)
      : dump_option_(std::move(dump_option)), cb_(Callback::Instance()) {}
  GraphKernelJsonGenerator(DumpOption dump_option, const CallbackPtr &cb)
      : dump_option_(std::move(dump_option)), cb_(cb) {}
  ~GraphKernelJsonGenerator() = default;

  bool CollectJson(const AnfNodePtr &anf_node, nlohmann::json *kernel_json);
  bool CollectFusedJson(const std::vector<AnfNodePtr> &anf_nodes, const std::vector<AnfNodePtr> &input_list,
                        const std::vector<AnfNodePtr> &output_list, nlohmann::json *kernel_json);
  bool CollectJson(const AnfNodePtr &anf_node);
  bool CollectFusedJson(const std::vector<AnfNodePtr> &anf_nodes, const std::vector<AnfNodePtr> &input_list,
                        const std::vector<AnfNodePtr> &output_list);
  bool CollectFusedJsonWithSingleKernel(const CNodePtr &c_node);

  std::string kernel_name() const { return kernel_name_; }
  nlohmann::json kernel_json() const { return kernel_json_; }
  std::string kernel_json_str() const { return kernel_json_.dump(); }
  const std::vector<size_t> &input_size_list() const { return input_size_list_; }
  const std::vector<size_t> &output_size_list() const { return output_size_list_; }
  std::map<std::string, AnfNodePtr> address_node_map() const { return address_node_map_; }
  const SymbolEnginePtr &symbol_engine() const { return symbol_engine_; }
  void set_symbol_engine(const SymbolEnginePtr &symbol_engine) { symbol_engine_ = symbol_engine; }

 private:
  bool GenerateSingleKernelJson(const AnfNodePtr &anf_node, nlohmann::json *node_json);
  bool CreateInputDescJson(const AnfNodePtr &anf_node, const OpInfoPtr &op_info, nlohmann::json *inputs_json);
  bool CreateOutputDescJson(const AnfNodePtr &anf_node, const OpInfoPtr &op_info, nlohmann::json *outputs_json);
  void GetAttrJson(const AnfNodePtr &anf_node, const std::vector<int64_t> &dyn_input_sizes, const OpAttrPtr &op_attr,
                   nlohmann::json *attr_json, const ValuePtr &attr_value);
  bool CreateAttrDescJson(const AnfNodePtr &anf_node, const OpInfoPtr &op_info, nlohmann::json *attrs_json);
  void GenStitchJson(const std::vector<AnfNodePtr> &anf_nodes, std::map<AnfNodePtr, nlohmann::json> *node_json_map,
                     nlohmann::json *kernel_json) const;
  void GetIOSize(const nlohmann::json &node_json, std::vector<size_t> *input_size,
                 std::vector<size_t> *output_size) const;
  bool GenSingleJsons(const std::vector<AnfNodePtr> &anf_nodes, std::map<AnfNodePtr, nlohmann::json> *node_json_map);
  void UpdateTensorName(const std::vector<AnfNodePtr> &anf_nodes,
                        std::map<AnfNodePtr, nlohmann::json> *node_json_map) const;
  nlohmann::json CreateInputsJson(const std::vector<AnfNodePtr> &anf_nodes, const std::vector<AnfNodePtr> &input_list,
                                  const std::map<AnfNodePtr, nlohmann::json> &node_json_map);
  nlohmann::json CreateOutputsJson(const std::vector<AnfNodePtr> &anf_nodes, const std::vector<AnfNodePtr> &input_list,
                                   const std::vector<AnfNodePtr> &output_list, const nlohmann::json &inputs_json,
                                   const std::map<AnfNodePtr, nlohmann::json> &node_json_map);
  size_t GetInputTensorIdxInc(const AnfNodePtr &anf_node, size_t input_idx);
  size_t GetOutputTensorIdxInc();
  void SetTensorName(const std::string &tag, const std::string &new_name, const std::pair<size_t, size_t> &position,
                     nlohmann::json *node_json) const;
  std::string GetTensorName(const nlohmann::json &node_json, const std::string &tag,
                            const std::pair<size_t, size_t> &position) const;
  void SaveNodeAddress(const AnfNodePtr &anf_node, nlohmann::json *node_json);
  OpInfoPtr ExtractOpInfo(const AnfNodePtr &anf_node) const;
  void GenParallelJson(const std::vector<AnfNodePtr> &anf_nodes, const std::vector<AnfNodePtr> &input_list,
                       const std::vector<AnfNodePtr> &output_list,
                       const std::map<AnfNodePtr, nlohmann::json> &node_json_map, nlohmann::json *kernel_json) const;
  bool GetInputTensorValue(const AnfNodePtr &anf_node, size_t input_idx, ShapeVector *input_shape,
                           nlohmann::json *node_json) const;
  size_t GetTensorSize(const nlohmann::json &node_json) const;
  std::string GetProcessorByTarget() const;
  size_t GenHashId(const std::string &info) const;
  void GenKernelName(const FuncGraphPtr &fg, size_t hash_id, nlohmann::json *kernel_json);
  void SaveSymbolicShape(const AnfNodePtr &node, nlohmann::json *kernel_json);

  DumpOption dump_option_;
  std::string kernel_name_;
  std::string all_ops_name_;
  std::unordered_map<AnfNodePtr, size_t> input_tensor_idx_;
  size_t output_tensor_idx_{0};
  nlohmann::json kernel_json_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::map<std::string, AnfNodePtr> address_node_map_;
  SymbolEnginePtr symbol_engine_;
  std::unordered_map<std::string, std::string> symbol_calc_exprs_;
  bool is_basic_op_{false};
  CallbackPtr cb_;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_GRAPH_KERNEL_GRAPH_KERNEL_JSON_GENERATOR_H_
