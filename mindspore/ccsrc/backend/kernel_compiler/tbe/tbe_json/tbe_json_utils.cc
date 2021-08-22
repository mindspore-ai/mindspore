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

#include "backend/kernel_compiler/tbe/tbe_json/tbe_json_utils.h"
#include <memory>
#include <string>
#include <vector>
#include "base/core_ops.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/tbe/tbe_convert_utils.h"
#include "backend/kernel_compiler/tbe/tbe_dynaminc_shape_util.h"
#include "runtime/dev.h"
#include "utils/json_operation_utils.h"

namespace mindspore::kernel {

bool TbeJsonUtils::GetInputsRealNum(const AnfNodePtr &anf_node, const std::vector<OpIOInfoPtr> &inputs_ptr,
                                    std::vector<size_t> *inputs_num) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(inputs_num);
  auto primitive = AnfAlgo::GetCNodePrimitive(anf_node);
  // for dynamic input number, dyn_input_sizes has the info of dynamic input num for each input.
  auto dyn_input_sizes_ptr = primitive->GetAttr(kAttrDynInputSizes);
  std::vector<int64_t> dyn_input_sizes = (dyn_input_sizes_ptr != nullptr)
                                           ? GetValue<const std::vector<int64_t>>(dyn_input_sizes_ptr)
                                           : std::vector<int64_t>{};
  size_t dyn_input_index = 0;
  for (const auto &input_ptr : inputs_ptr) {
    if (input_ptr->param_type() == kJParamDynamic) {
      if (dyn_input_index >= dyn_input_sizes.size()) {
        MS_LOG(ERROR) << "Dyn input index" << dyn_input_index << "is over dyn input num" << dyn_input_sizes.size();
        return false;
      } else {
        (*inputs_num).emplace_back(LongToSize(dyn_input_sizes[dyn_input_index]));
        dyn_input_index++;
      }
    } else {
      (*inputs_num).emplace_back(1);
    }
  }
  return true;
}

bool TbeJsonUtils::GetOutputsRealNum(const AnfNodePtr &anf_node, const std::vector<OpIOInfoPtr> &outputs_ptr,
                                     std::vector<size_t> *outputs_num) {
  MS_EXCEPTION_IF_NULL(anf_node);
  size_t real_output_num = AnfAlgo::GetOutputTensorNum(anf_node);
  for (const auto &output_ptr : outputs_ptr) {
    if (output_ptr->param_type() == kJParamDynamic) {
      if (outputs_ptr.size() > 1) {
        MS_LOG(ERROR) << "Dynamic output is unsupported multi output, node [ " << AnfAlgo::GetCNodeName(anf_node)
                      << " ] has " << outputs_ptr.size() << "outputs, however one of the outputs param_type is "
                      << output_ptr->param_type();
        return false;
      }
      outputs_num->emplace_back(real_output_num);
    } else {
      outputs_num->emplace_back(1);
    }
  }
  return true;
}

bool TbeJsonUtils::IsNeedChangeDefaultFormat(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  return anf_node->isa<CNode>() && AnfAlgo::HasNodeAttr(kAttrFormat, anf_node->cast<CNodePtr>()) &&
         AnfAlgo::GetNodeAttr<std::string>(anf_node, kAttrFormat) == kOpFormat_NCDHW;
}

}  // namespace mindspore::kernel
