/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_property_checker.h"
#include <map>
#include <string>
#include <vector>
#include "include/common/utils/anfalgo.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace kernel {
using Tensor = mindspore::tensor::Tensor;
using TensorPtr = mindspore::tensor::TensorPtr;
using AbstractTensor = mindspore::abstract::AbstractTensor;
using AbstractTensorPtr = mindspore::abstract::AbstractTensorPtr;
using CheckSupportFun = bool (*)(const CNodePtr &cnode);

constexpr char kAttrSorted[] = "sorted";
constexpr char kAttrReduction[] = "reduction";
constexpr char kAttrStrides[] = "strides";
constexpr char kAttrShrinkAxisMask[] = "shrink_axis_mask";

bool CheckValueType(const AnfNodePtr &input_node, size_t inputs_num) {
  MS_EXCEPTION_IF_NULL(input_node);
  auto value_node = input_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<tensor::Tensor>()) {
    MS_EXCEPTION(ValueError) << "The strides of StridedSliceGrad must be a constant. Total inputs of cnode is  "
                             << inputs_num;
  }
  auto tensor = value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor);
  TypePtr data_type = tensor->Dtype();
  MS_EXCEPTION_IF_NULL(data_type);
  TypeId type_id = data_type->type_id();
  auto element_size = tensor->data().size();
  if (type_id == kNumberTypeInt32) {
    auto *data = reinterpret_cast<int *>(tensor->data_c());
    if ((data[element_size - 1]) != 1) {
      return false;
    }
  } else if (type_id == kNumberTypeInt64) {
    auto *data = reinterpret_cast<int64_t *>(tensor->data_c());
    if ((data[element_size - 1]) != 1) {
      return false;
    }
  } else {
    MS_EXCEPTION(TypeError) << "The strides of StridedSliceGrad must be int.";
  }
  return true;
}

static bool CheckStridedSlice(const CNodePtr &cnode) {
  // check stride[-1] != 1
  if (common::AnfAlgo::HasNodeAttr(kAttrStrides, cnode)) {
    auto strides = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(cnode, kAttrStrides);
    bool has_negative = std::any_of(strides.begin(), strides.end(), [](int elem) -> bool { return elem <= 0; });
    if (!strides.empty() && has_negative) {
      return false;
    }
  } else {
    auto inputs = cnode->inputs();
    const size_t kInputNum = 5;
    if (inputs.size() == kInputNum + IntToSize(1)) {
      auto input_node = inputs[kInputNum];
      MS_EXCEPTION_IF_NULL(input_node);
      // Input node can be a cnode, like cast or transdata, which output is a valuenode
      if (input_node->isa<CNode>()) {
        return true;
      }
      return CheckValueType(input_node, inputs.size());
    }
  }
  return true;
}

static bool CheckTopK(const CNodePtr &cnode) {
  if (common::AnfAlgo::HasNodeAttr(kAttrSorted, cnode)) {
    auto sorted = common::AnfAlgo::GetNodeAttr<bool>(cnode, kAttrSorted);
    return sorted;
  }
  MS_LOG(EXCEPTION) << "For 'TopK', it should be have attribute 'sorted'." << trace::DumpSourceLines(cnode);
}

static bool CheckKLDivLoss(const CNodePtr &cnode) {
  if (common::AnfAlgo::HasNodeAttr(kAttrReduction, cnode)) {
    auto reduction = common::AnfAlgo::GetNodeAttr<string>(cnode, kAttrReduction);
    return reduction != "mean";
  }
  MS_LOG(EXCEPTION) << "For 'KLDivLoss', it should be have attribute 'reduction'." << trace::DumpSourceLines(cnode);
}

bool TbePropertyChecker::CheckTbeProperties(const mindspore::CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  // Skip check for ACL op.
  if (common::AnfAlgo::HasNodeAttr(kAttrMutableKernel, cnode)) {
    return true;
  }
  static std::map<std::string, CheckSupportFun> tbe_property_checker = {{kStridedSliceOpName, CheckStridedSlice},
                                                                        {kStridedSliceDOpName, CheckStridedSlice},
                                                                        {kStridedSliceGradOpName, CheckStridedSlice},
                                                                        {kTopKOpName, CheckTopK},
                                                                        {kKLDivOpName, CheckKLDivLoss}};
  auto cnode_type = common::AnfAlgo::GetCNodeName(cnode);
  auto find_iter = tbe_property_checker.find(cnode_type);
  if (find_iter != tbe_property_checker.end()) {
    return find_iter->second(cnode);
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
