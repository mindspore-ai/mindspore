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

#include "plugin/device/ascend/kernel/tbe/tbe_json/tbe_json_utils.h"
#include "mindspore/core/ops/core_ops.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/kernel/tbe/tbe_convert_utils.h"
#include "plugin/device/ascend/kernel/tbe/tbe_dynamic_shape_util.h"
#include "runtime/dev.h"
#include "include/common/utils/json_operation_utils.h"

namespace mindspore::kernel {
bool TbeJsonUtils::GetInputsRealNum(const AnfNodePtr &anf_node, const std::vector<OpIOInfoPtr> &inputs_ptr,
                                    std::vector<size_t> *inputs_num) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(inputs_num);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(anf_node);
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
      }
    } else {
      (*inputs_num).emplace_back(1);
    }
    dyn_input_index++;
  }
  return true;
}

bool TbeJsonUtils::GetOutputsRealNum(const AnfNodePtr &anf_node, const std::vector<OpIOInfoPtr> &outputs_ptr,
                                     std::vector<size_t> *outputs_num) {
  MS_EXCEPTION_IF_NULL(anf_node);
  size_t real_output_num = AnfAlgo::GetOutputElementNum(anf_node);
  for (const auto &output_ptr : outputs_ptr) {
    if (output_ptr->param_type() == kJParamDynamic) {
      if (outputs_ptr.size() > 1) {
        MS_LOG(ERROR) << "Dynamic output is unsupported multi output, node [ "
                      << common::AnfAlgo::GetCNodeName(anf_node) << " ] has " << outputs_ptr.size()
                      << "outputs, however one of the outputs param_type is " << output_ptr->param_type();
        return false;
      }
      (void)outputs_num->emplace_back(real_output_num);
    } else {
      (void)outputs_num->emplace_back(1);
    }
  }
  return true;
}

bool TbeJsonUtils::IsNeedChangeDefaultFormat(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  return anf_node->isa<CNode>() && common::AnfAlgo::HasNodeAttr(kAttrFormat, anf_node->cast<CNodePtr>()) &&
         common::AnfAlgo::GetNodeAttr<std::string>(anf_node, kAttrFormat) == kOpFormat_NCDHW;
}

std::vector<int64_t> TbeJsonUtils::GetInputOriShapeForTbeBuild(const AnfNodePtr &anf_node, size_t real_idx) {
  MS_EXCEPTION_IF_NULL(anf_node);
  session::KernelWithIndex kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(anf_node, real_idx);
  return GetOutputOriShapeForTbeBuild(kernel_with_index.first, kernel_with_index.second);
}

std::vector<int64_t> TbeJsonUtils::GetInputDeviceShapeForTbeBuild(const AnfNodePtr &anf_node, size_t real_idx) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::vector<int64_t> shape;
  auto format = AnfAlgo::GetInputFormat(anf_node, real_idx);
  shape = AnfAlgo::GetInputDeviceShapeForTbeBuild(anf_node, real_idx, format);
  if (shape.empty()) {
    shape.emplace_back(1);
  }
  return shape;
}

std::vector<int64_t> TbeJsonUtils::GetOutputOriShapeForTbeBuild(const AnfNodePtr &anf_node, size_t real_idx) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::vector<int64_t> shape;
  auto out_shape = AnfAlgo::GetOutputDetailShape(anf_node, real_idx);
  MS_EXCEPTION_IF_NULL(out_shape);
  if (out_shape->isa<abstract::Shape>()) {
    auto shape_ptr = out_shape->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape_ptr);
    shape = shape_ptr->shape();
  }
  if (shape.empty()) {
    (void)shape.emplace_back(1);
  }
  return shape;
}

std::vector<int64_t> TbeJsonUtils::GetOutputDeviceShapeForTbeBuild(const AnfNodePtr &anf_node, size_t real_idx) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::vector<int64_t> shape;
  auto format = AnfAlgo::GetOutputFormat(anf_node, real_idx);
  shape = AnfAlgo::GetOutputDeviceShapeForTbeBuild(anf_node, real_idx, format);
  if (shape.empty()) {
    (void)shape.emplace_back(1);
  }
  return shape;
}
}  // namespace mindspore::kernel
