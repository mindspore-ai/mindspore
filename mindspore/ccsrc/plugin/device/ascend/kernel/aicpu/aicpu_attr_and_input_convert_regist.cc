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

#include "plugin/device/ascend/kernel/aicpu/aicpu_attr_and_input_convert_regist.h"
#include <algorithm>
#include "ops/nn_ops.h"
#include "ops/math_ops.h"
#include "ops/array_ops.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/kernel_graph.h"
#include "include/common/utils/convert_utils.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace kernel {
namespace {
/*
 * Parameter is input in AICPU, but is attribute in TBE.
 * {
 *   {op_name, {{attr_name, pos_index}, ...},
 *   ...
 * }
 */
std::map<string, std::vector<std::pair<string, size_t>>> AicpuOpAttrToInputMap = {
  {prim::kPrimOneHot->name(), {{"depth", 1}}},
  {prim::kPrimConcat->name(), {{"axis", 0}}},
  {prim::kPrimTranspose->name(), {{"perm", 1}}},
  {prim::kPrimGather->name(), {{"axis", 2}}},
  {prim::kPrimSlice->name(), {{"begin", 1}, {"size", 2}}},
  {prim::kPrimReduceMean->name(), {{"axis", 1}}},
  {prim::kPrimSplit->name(), {{"axis", 0}}},
  {prim::kPrimCumSum->name(), {{"axis", 1}}},
  {prim::kPrimCumProd->name(), {{"axis", 1}}},
  {prim::kPrimScatterNd->name(), {{"shape", 2}}},
  {prim::kPrimReduceProd->name(), {{"axis", 1}}},
  {prim::kPrimReverseV2->name(), {{"axis", 1}}},
  {prim::kPrimBroadcastTo->name(), {{"shape", 1}}},
  {prim::kPrimArgMax->name(), {{"axis", 1}}},
  {prim::kPrimArgmin->name(), {{"axis", 1}}},
  {prim::kPrimReduceSum->name(), {{"axis", 1}}},
  {prim::kPrimTile->name(), {{"multiples", 1}}},
  {prim::kPrimUnsortedSegmentProd->name(), {{"num_segments", 2}}},
  {prim::kPrimUnsortedSegmentSumD->name(), {{"num_segments", 2}}}};

/*
 * Parameter is attr in AICPU, but is input in graph.
 * {
 *   {op_name, {{pos_index， data_type}, ...},
 *   ...
 * }
 */
std::map<std::string, std::map<size_t, std::string>> AicpuOpInputToAttrMap = {
  {kStridedSliceOpName, {{1, "listInt"}, {2, "listInt"}, {3, "listInt"}}}, {kExpandDimsOpName, {{1, "int"}}}};

void ConvertAttrToInput(const CNodePtr &kernel_node, std::vector<std::pair<string, size_t>> *infos) {
  auto graph = kernel_node->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(kernel_node);
  MS_EXCEPTION_IF_NULL(primitive);

  std::ostringstream buf;
  for (auto &info : *infos) {
    buf << " (" << info.first << ", " << info.second << ")";
  }
  MS_LOG(DEBUG) << "Start converting attr to input for aicpu op[" << AnfUtils::GetCNodeName(kernel_node)
                << "] with attr_name and input_index pairs:" << buf.str();

  std::sort(infos->begin(), infos->end(),
            [](const std::pair<string, size_t> &a, const std::pair<string, size_t> &b) { return a.second < b.second; });
  auto orig_inputs = kernel_node->inputs();
  size_t orig_input_num = orig_inputs.size() - 1;
  size_t new_input_num = orig_input_num + infos->size();
  size_t orig_tmp_idx = 0;
  size_t attr_tmp_idx = 0;
  std::vector<AnfNodePtr> new_inputs = {orig_inputs[0]};
  for (size_t idx = 0; idx < new_input_num; ++idx) {
    if (attr_tmp_idx < infos->size() && idx == infos->at(attr_tmp_idx).second) {
      auto attr_name = infos->at(attr_tmp_idx).first;
      auto value = primitive->GetAttr(attr_name);
      if (value == nullptr) {
        MS_LOG(DEBUG) << "Can not get attr[" << attr_name << "].";
        return;
      }
      tensor::TensorPtr tensor_ptr = nullptr;
      if (value->isa<tensor::Tensor>()) {
        tensor_ptr = value->cast<tensor::TensorPtr>();
      } else if (value->isa<Scalar>()) {
        tensor_ptr = ScalarToTensor(value->cast<ScalarPtr>());
      } else if (value->isa<ValueTuple>()) {
        tensor_ptr = opt::CreateTupleTensor(value->cast<ValueTuplePtr>());
      } else {
        MS_LOG(DEBUG) << "The value of attr[" << attr_name << "] should be a tensor or scalar or value tuple.";
        return;
      }
      if (tensor_ptr == nullptr) {
        MS_LOG(DEBUG) << "Convert attr[" << attr_name << "] to tensor value failed.";
        return;
      }
      auto value_node = kernel_graph->NewValueNode(tensor_ptr);
      MS_EXCEPTION_IF_NULL(value_node);
      new_inputs.push_back(value_node);
      ++attr_tmp_idx;
    } else if (orig_tmp_idx < orig_input_num) {
      new_inputs.push_back(orig_inputs[orig_tmp_idx + 1]);
      ++orig_tmp_idx;
    }
  }
  kernel_node->set_inputs(new_inputs);
}

bool ConvertConstInputToAttr(const CNodePtr &cnode, const std::map<size_t, std::string> &input_to_attr_info) {
  AnfNodePtrList new_inputs;
  auto primitive = GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(primitive);
  auto inputs = cnode->inputs();
  new_inputs.push_back(inputs[0]);
  for (size_t i = 0; i < inputs.size() - 1; ++i) {
    auto input_node = inputs[i + 1];
    MS_EXCEPTION_IF_NULL(input_node);
    auto iter = input_to_attr_info.find(i);
    if (iter != input_to_attr_info.end() && input_node->isa<ValueNode>()) {
      auto value_node = input_node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      auto value = value_node->value();
      MS_EXCEPTION_IF_NULL(value);
      if (value->isa<tensor::Tensor>()) {
        auto tensor = value->cast<tensor::TensorPtr>();
        if (tensor->data().const_data() == nullptr && !tensor->has_user_data(kTensorValueIsEmpty)) {
          MS_LOG(DEBUG) << "Const input data ptr is null from op " << cnode->fullname_with_scope() << "'s input " << i;
          return false;
        }
        value = CreateValueFromTensor(tensor);
        value = UpdateValueByAttrDataType(value, iter->second);
        MS_LOG(DEBUG) << "new attr value:" << value_node->ToString() << ", Type:" << value_node->type_name();
      }

      std::string attr_name = common::AnfAlgo::GetInputName(cnode, i);
      if (attr_name.empty()) {
        return false;
      }

      if (cnode->HasAttr(attr_name)) {
        auto origin_primitive = GetCNodePrimitive(cnode);
        MS_EXCEPTION_IF_NULL(origin_primitive);
        MS_LOG(ERROR) << "Origin op already has this attr " << attr_name
                      << ". op attrs:" << origin_primitive->GetAttrsText() << ". DebugString:" << cnode->DebugString();
        return false;
      }

      primitive->set_attr(attr_name, value);
    } else {
      new_inputs.push_back(inputs[i + 1]);
    }
  }
  if (new_inputs.size() != inputs.size()) {
    cnode->set_inputs(new_inputs);
  }
  return true;
}

}  // namespace

bool GetAicpuOpAttrToInputInfo(const CNodePtr &kernel_node, std::vector<std::pair<string, size_t>> *info) {
  std::string op_name = common::AnfAlgo::GetCNodeName(kernel_node);
  if (AicpuOpAttrToInputMap.find(op_name) == AicpuOpAttrToInputMap.end()) {
    return false;
  } else {
    *info = AicpuOpAttrToInputMap[op_name];
    return true;
  }
}

bool GetAicpuOpInputToAttrInfo(const CNodePtr &kernel_node, std::map<size_t, std::string> *input_to_attr_info) {
  std::string op_name = common::AnfAlgo::GetCNodeName(kernel_node);
  if (AicpuOpInputToAttrMap.find(op_name) == AicpuOpInputToAttrMap.end()) {
    return false;
  } else {
    *input_to_attr_info = AicpuOpInputToAttrMap[op_name];
    return true;
  }
}

void ConvertAttrAndInputBeforeAicpuKernelSelect(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<std::pair<string, size_t>> attr_to_input_infos;
  if (kernel::GetAicpuOpAttrToInputInfo(kernel_node, &attr_to_input_infos)) {
    ConvertAttrToInput(kernel_node, &attr_to_input_infos);
  }

  std::map<size_t, std::string> input_to_attr_info;
  if (kernel::GetAicpuOpInputToAttrInfo(kernel_node, &input_to_attr_info) &&
      !common::AnfAlgo::IsDynamicShape(kernel_node)) {
    (void)ConvertConstInputToAttr(kernel_node, input_to_attr_info);
  }
}
}  // namespace kernel
}  // namespace mindspore
