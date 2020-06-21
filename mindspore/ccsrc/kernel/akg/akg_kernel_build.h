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

#ifndef MINDSPORE_CCSRC_KERNEL_AKG_AKGKERNELBUILD_H_
#define MINDSPORE_CCSRC_KERNEL_AKG_AKGKERNELBUILD_H_
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <utility>
#include "kernel/kernel.h"
#include "ir/dtype.h"
#include <nlohmann/json.hpp>
#include "kernel/common_utils.h"
#include "kernel/oplib/oplib.h"

namespace mindspore {
namespace kernel {
class AkgKernelBuild {
 public:
  AkgKernelBuild() {
    input_tensor_idx_ = {};
    output_tensor_idx_ = 0;
  }
  ~AkgKernelBuild() = default;

  KernelPackPtr BuildByJson(const AnfNodePtr &anf_node, std::vector<size_t> *const input_size,
                            std::vector<size_t> *const output_size);
  static std::string GetProcessor(const AnfNodePtr &anf_node);
  static std::string PyObjectToStr(PyObject *const PyObj);

 protected:
  bool CreateInputDescJson(const AnfNodePtr &anf_node, nlohmann::json *const inputs_json);
  bool CreateOutputDescJson(const AnfNodePtr &anf_node, nlohmann::json *const outputs_json);
  bool CreateAttrDescJson(const AnfNodePtr &anf_node, const std::string &op_name,
                          const std::shared_ptr<OpInfo> &op_info, nlohmann::json *const attrs_json);
  KernelPackPtr OpBuild(const std::string &node_json, const AnfNodePtr &anf_node);
  int GetOpCntInc();
  size_t GetInputTensorIdxInc(const AnfNodePtr &anf_node, size_t input_idx);
  size_t GetOutputTensorIdxInc();
  bool GenerateSingleKernelJson(const AnfNodePtr &anf_node, const std::string &op_name,
                                nlohmann::json *const node_json);

  static int op_cnt_;
  // lock for variable fusionOpCnt in singleton mode
  static std::mutex op_cnt_mtx_;
  std::string json_name_;
  std::string json_info_;
  std::unordered_map<AnfNodePtr, size_t> input_tensor_idx_;
  size_t output_tensor_idx_;
};

bool GetIOSize(const nlohmann::json &node_json, std::vector<size_t> *const input_size,
               std::vector<size_t> *const output_size);
void SetTensorName(const std::string &tag, const std::string &new_name, const std::pair<size_t, size_t> &position,
                   nlohmann::json *const node_json);
std::string GetTensorName(const nlohmann::json &node_json, const std::string &tag,
                          const std::pair<size_t, size_t> &position);

}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_AKG_AKGKERNELBUILD_H_
