/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_AKG_KERNEL_JSON_GENERATOR_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_AKG_KERNEL_JSON_GENERATOR_H_
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "backend/kernel_compiler/oplib/oplib.h"
#include "nlohmann/json.hpp"

namespace mindspore {
namespace kernel {
// json key
constexpr auto kJsonKeyOpDesc = "op_desc";
constexpr auto kJsonKeyAttr = "attr";
constexpr auto kJsonKeyInputDesc = "input_desc";
constexpr auto kJsonKeyFormat = "format";
constexpr auto kJsonKeyInferDataType = "infer_data_type";
constexpr auto kJsonKeyInferShape = "infer_shape";
constexpr auto kJsonKeyShape = "shape";
constexpr auto kJsonKeyDataType = "data_type";
constexpr auto kJsonKeyOutputDesc = "output_desc";
constexpr auto kJsonKeyName = "name";
constexpr auto kJsonKeyTensorName = "tensor_name";
constexpr auto kJsonKeyValue = "value";
constexpr auto kJsonKeyImplPath = "impl_path";
constexpr auto kJsonKeyProcess = "process";
constexpr auto kJsonKeyComposite = "composite";
constexpr auto kJsonKeyId = "id";
constexpr auto kJsonKeyOp = "op";
constexpr auto kJsonKeyPtrAddress = "ptr_address";
constexpr auto kJsonKeyCompositeGraph = "composite_graph";
constexpr auto kJsonKeyPlatform = "platform";
constexpr auto kJsonKeyOpFullName = "op_full_name";
constexpr auto kJsonKeyFusion = "fusion";
constexpr auto kJsonKeyParallelFusion = "parallel_fusion";
constexpr auto kJsonKeyFusionType = "fusion_type";
constexpr auto kJsonKeySubGraph = "sub_graph";
constexpr auto kJsonKeyCoreNum = "core_num";
constexpr auto kJsonKeyTypeInfo = "type_info";
constexpr auto kJsonKeyBufferStitch = "buffer_stitch";
constexpr auto kJsonKeyStitchOp = "stitch_op";
constexpr auto kJsonKeyStitchAtomicOp = "stitch_atomic_op";

constexpr auto kAttrInputNames = "input_names";

// dump option
struct DumpOption {
  bool is_before_select_kernel = false;
  bool save_ptr_address = false;
  bool extract_opinfo_from_anfnode = false;
};

class AkgKernelJsonGenerator {
 public:
  AkgKernelJsonGenerator() { Clear(); }
  explicit AkgKernelJsonGenerator(DumpOption dump_option) : dump_option_(dump_option) { Clear(); }
  ~AkgKernelJsonGenerator() = default;

  bool CollectJson(const AnfNodePtr &anf_node, nlohmann::json *kernel_json);
  bool CollectFusedJson(const std::vector<AnfNodePtr> &anf_nodes, const std::vector<AnfNodePtr> &input_list,
                        const std::vector<AnfNodePtr> &output_list, nlohmann::json *kernel_json);
  bool CollectJson(const AnfNodePtr &anf_node);
  bool CollectFusedJson(const std::vector<AnfNodePtr> &anf_nodes, const std::vector<AnfNodePtr> &input_list,
                        const std::vector<AnfNodePtr> &output_list);
  bool GenerateSingleKernelJson(const AnfNodePtr &anf_node, nlohmann::json *node_json);
  std::string kernel_name() const { return kernel_name_; }
  nlohmann::json kernel_json() const { return kernel_json_; }
  std::string kernel_json_str() const { return kernel_json_.dump(); }
  const std::vector<size_t> &input_size_list() const { return input_size_list_; }
  const std::vector<size_t> &output_size_list() const { return output_size_list_; }
  void Clear() {
    input_tensor_idx_.clear();
    address_node_map_.clear();
    output_tensor_idx_ = 0;
  }
  void set_dump_option(DumpOption dump_option) { dump_option_ = dump_option; }
  std::map<std::string, AnfNodePtr> address_node_map() { return address_node_map_; }

 private:
  bool CreateInputDescJson(const AnfNodePtr &anf_node, const OpInfoPtr &op_info, nlohmann::json *inputs_json);
  bool CreateOutputDescJson(const AnfNodePtr &anf_node, const OpInfoPtr &op_info, nlohmann::json *outputs_json);
  void GetAttrJson(const AnfNodePtr &anf_node, const std::vector<int> &dyn_input_sizes, const OpAttrPtr &op_attr,
                   nlohmann::json *attr_json, const ValuePtr &attr_value);
  bool CreateAttrDescJson(const AnfNodePtr &anf_node, const OpInfoPtr &op_info, nlohmann::json *attrs_json);
  void GenStitchJson(const std::vector<AnfNodePtr> &anf_nodes, std::map<AnfNodePtr, nlohmann::json> *node_json_map,
                     nlohmann::json *kernel_json);
  bool GetIOSize(const nlohmann::json &node_json, std::vector<size_t> *input_size, std::vector<size_t> *output_size);
  bool GenSingleJsons(const std::vector<AnfNodePtr> &anf_nodes, std::map<AnfNodePtr, nlohmann::json> *node_json_map);
  void UpdateTensorName(const std::vector<AnfNodePtr> &anf_nodes, std::map<AnfNodePtr, nlohmann::json> *node_json_map);
  nlohmann::json CreateInputsJson(const std::vector<AnfNodePtr> &anf_nodes, const std::vector<AnfNodePtr> &input_list,
                                  const std::map<AnfNodePtr, nlohmann::json> &node_json_map);
  nlohmann::json CreateOutputsJson(const std::vector<AnfNodePtr> &anf_nodes, const std::vector<AnfNodePtr> &input_list,
                                   const std::vector<AnfNodePtr> &output_list, const nlohmann::json &inputs_json,
                                   const std::map<AnfNodePtr, nlohmann::json> &node_json_map);

  int GetOpCntInc();
  size_t GetInputTensorIdxInc(const AnfNodePtr &anf_node, size_t input_idx);
  size_t GetOutputTensorIdxInc();
  void SetTensorName(const std::string &tag, const std::string &new_name, const std::pair<size_t, size_t> &position,
                     nlohmann::json *node_json);
  std::string GetTensorName(const nlohmann::json &node_json, const std::string &tag,
                            const std::pair<size_t, size_t> &position);
  TypeId GetInputDataType(const AnfNodePtr &anf_node, size_t real_index);
  std::vector<size_t> GetInputShape(const AnfNodePtr &anf_node, size_t real_index);
  std::string GetInputFormat(const AnfNodePtr &anf_node, size_t real_index);
  TypeId GetOutputDataType(const AnfNodePtr &anf_node, size_t index);
  std::vector<size_t> GetOutputShape(const AnfNodePtr &anf_node, size_t index);
  std::string GetOutputFormat(const AnfNodePtr &anf_node, size_t index);
  void SaveNodeAddress(const AnfNodePtr &anf_node, nlohmann::json *node_json);
  OpInfoPtr ExtractOpInfo(const AnfNodePtr &anf_node);
  void CollectParallelDimInfo(const AnfNodePtr &anf_node);
  void GenParallelJson(const std::vector<AnfNodePtr> &anf_nodes, const std::vector<AnfNodePtr> &input_list,
                       const std::vector<AnfNodePtr> &output_list,
                       const std::map<AnfNodePtr, nlohmann::json> &node_json_map, nlohmann::json *kernel_json);

  DumpOption dump_option_;
  static int op_cnt_;
  // lock for variable fusionOpCnt in singleton mode
  static std::mutex op_cnt_mtx_;
  std::string kernel_name_;
  std::unordered_map<AnfNodePtr, size_t> input_tensor_idx_;
  size_t output_tensor_idx_;
  nlohmann::json kernel_json_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::map<std::string, AnfNodePtr> address_node_map_;
  bool is_basic_op_{false};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_AKG_KERNEL_JSON_GENERATOR_H_
