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

#ifndef MINDSPORE_CCSRC_KERNEL_TBE_TBE_KERNEL_BUILD_H_
#define MINDSPORE_CCSRC_KERNEL_TBE_TBE_KERNEL_BUILD_H_

#include <string>
#include <unordered_map>
#include <memory>
#include <map>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>
#include "ir/dtype.h"
#include "kernel/kernel.h"
#include "pybind11/stl.h"
#include "kernel/oplib/oplib.h"
#include "kernel/tbe/tbe_adapter.h"

namespace mindspore {
namespace kernel {
// kernel operate type used for generate json

class TbeKernelBuild {
 public:
  static bool GetIOSize(const nlohmann::json &kernel_json, std::vector<size_t> *input_size_list,
                        std::vector<size_t> *output_size_list);
  // Ub Fuison
  static bool GenFusionScopeJson(const std::vector<AnfNodePtr> &input_nodes,
                                 const std::vector<AnfNodePtr> &compute_nodes, nlohmann::json *fusion_str,
                                 std::string *fusion_kernel);
  static bool GetIOSize(const nlohmann::json &fusion_op_list, const std::vector<AnfNodePtr> &output_nodes,
                        std::vector<size_t> *input_size_list, std::vector<size_t> *output_size_list);

 private:
  TbeKernelBuild() = default;
  ~TbeKernelBuild() = default;
  static bool GenFusionDataInputJson(const shared_ptr<mindspore::AnfNode> &data_input, nlohmann::json *data_str,
                                     size_t *index);
  static bool GenFusionComputeJson(const mindspore::AnfNodePtr &compute_node,
                                   std::vector<std::vector<mindspore::AnfNodePtr>>::iterator *layer_iter,
                                   nlohmann::json *compute_op_str, std::string *fusion_kernel_name, size_t *index);
  static bool GenFusionComputeInputeJson(const mindspore::CNodePtr &cnode,
                                         std::vector<std::vector<mindspore::AnfNodePtr>>::iterator *layer_iter,
                                         std::vector<nlohmann::json> *input_desc_list, size_t *index);
  static void GenDescJson(const shared_ptr<mindspore::AnfNode> &anf_node, size_t out_idx, nlohmann::json *output_desc);
  static void GenReusedOutputDesc(const shared_ptr<mindspore::AnfNode> &anf_node, size_t index, size_t output_index,
                                  nlohmann::json *output_desc);
  static size_t GetIOSizeImpl(const nlohmann::json &desc);
  static bool GetInputLayers(const vector<mindspore::AnfNodePtr> &input_nodes,
                             const vector<mindspore::AnfNodePtr> &compute_nodes,
                             std::vector<std::vector<mindspore::AnfNodePtr>> *input_layers);
  static bool IsDynamicInput(const CNodePtr &cnode);
  static size_t GetOptionalInput(const CNodePtr &cnode, bool is_dynamic_input);
};

class TbeKernelJsonCreator {
 public:
  explicit TbeKernelJsonCreator(kCreaterType creater_type = SINGLE_BUILD) : creater_type_(creater_type) {}
  ~TbeKernelJsonCreator() = default;
  bool GenTbeSingleKernelJson(const std::shared_ptr<AnfNode> &anf_node, nlohmann::json *kernel_json);
  std::string json_name() { return json_name_; }

 private:
  bool GenTbeInputsJson(const std::shared_ptr<AnfNode> &anf_node, const std::shared_ptr<OpInfo> &op_info,
                        nlohmann::json *inputs_json);
  bool GenTbeOutputsJson(const std::shared_ptr<AnfNode> &anf_node, const std::shared_ptr<OpInfo> &op_info,
                         nlohmann::json *outputs_json);
  bool GenTbeAttrJson(const std::shared_ptr<AnfNode> &anf_node, const std::shared_ptr<OpInfo> &op_info,
                      nlohmann::json *attrs_json);
  void ParseAttrValue(const std::string &type, const ValuePtr &value, nlohmann::json *attr_obj);
  bool GenInputDescJson(const shared_ptr<AnfNode> &anf_node, size_t real_input_index, bool value,
                        const shared_ptr<OpIOInfo> &input_ptr, const string &op_input_name, size_t input_i,
                        vector<nlohmann::json> *input_list);
  bool GenOutputDescJson(const shared_ptr<AnfNode> &anf_node, const vector<std::shared_ptr<OpIOInfo>> &outputs_ptr,
                         nlohmann::json *outputs_json);
  bool GenInputList(const shared_ptr<AnfNode> &anf_node, size_t input_tensor_num, const shared_ptr<OpIOInfo> &input_ptr,
                    size_t *real_input_index, string *op_input_name, vector<nlohmann::json> *input_list);
  void GenOutputList(const shared_ptr<AnfNode> &anf_node, const size_t &output_obj_num,
                     const shared_ptr<OpIOInfo> &output_ptr, size_t *output_idx, vector<nlohmann::json> *output_list);
  kCreaterType creater_type_;
  std::string json_name_;
  std::string json_info_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_TBE_TBE_KERNEL_BUILD_H_
