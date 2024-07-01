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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_IR_BPROP_TENSOR_REPLACE_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_IR_BPROP_TENSOR_REPLACE_H_

#include <string>
#include <map>
#include <vector>
#include <utility>
#include "utils/hash_map.h"
#include "ir/tensor.h"
#include "pipeline/jit/ps/resource.h"

namespace mindspore {
namespace pynative {
using TensorIdWithOpInfo = mindspore::HashMap<std::string, std::pair<std::string, size_t>>;
using OpInfoWithTensorObject = std::map<std::string, std::vector<std::pair<size_t, tensor::BaseTensorPtr>>>;
// key: op info, value: op output value
using OpInfoWithForwardOutput = std::map<std::string, ValuePtr>;

struct TensorReplaceInfo {
  TensorIdWithOpInfo id_with_op_info{};
  OpInfoWithTensorObject op_info_with_tensor_object{};
  OpInfoWithForwardOutput op_info_with_forward_output{};
  size_t need_replace_size{};
  void clear() {
    op_info_with_tensor_object.clear();
    op_info_with_forward_output.clear();
  }
};

void SetIdWithOpInfo(const ValuePtr &v, const std::string &op_info, size_t out_index,
                     TensorIdWithOpInfo *id_with_op_info);
void UpdateForwardOutputTensorInfo(const std::string &op_info, const ValuePtr &v,
                                   const TensorReplaceInfo &replace_info);
void StoreForwardOutputWithOpInfo(const OpInfoWithTensorObject &op_info_with_tensor_object, const std::string &op_info,
                                  const ValuePtr &v, TensorReplaceInfo *replace_info);
void UpdatePipelineTopCellFowardTensor(const TensorReplaceInfo &ir_replace_info,
                                       const TensorReplaceInfo &cur_replace_info);
void SaveForwardOutputTensorInfo(const FuncGraphPtr &func_graph, bool need_save_tensor_info,
                                 TensorReplaceInfo *replace_info);
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_IR_BPROP_TENSOR_REPLACE_H_
