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

#include "coder/shape_info_container.h"
#include <utility>
#include <algorithm>
#include "src/litert/infer_manager.h"
#include "coder/opcoders/op_coder.h"
#include "tools/common/string_util.h"

namespace mindspore::lite::micro {
int ShapeInfoContainer::Init(const std::vector<std::unique_ptr<OperatorCoder>> &nodes_coder,
                             const std::map<Tensor *, std::vector<std::vector<int>>> &graph_inputs) {
  MS_CHECK_TRUE_MSG(!graph_inputs.empty(), RET_ERROR, "Cannot get graph_inputs's shape-info");
  auto scene_num = graph_inputs.begin()->second.size();
  for (const auto &item : graph_inputs) {
    MS_CHECK_TRUE_MSG(item.first, RET_NULL_PTR, "Find a nullptr in graph_inputs");
    MS_CHECK_TRUE_MSG(item.second.size() == scene_num, RET_ERROR, "Graph inputs are invalid.");
  }
  var_tensor_shapes_.insert(graph_inputs.begin(), graph_inputs.end());
  for (size_t i = 0; i < scene_num; ++i) {
    for (const auto &item : graph_inputs) {
      item.first->set_shape(item.second[i]);
    }
    for (const auto &node_coder : nodes_coder) {
      auto in_tensors = node_coder->input_tensors();
      auto out_tensors = node_coder->output_tensors();
      auto op_param = node_coder->get_parameter();
      MS_CHECK_TRUE_MSG(op_param, RET_NULL_PTR, "NodeCoder's op_param is a nullptr.");
      auto node = node_coder->node();
      MS_CHECK_TRUE_MSG(node, RET_NULL_PTR, "NodeCoder's node is a nullptr.");
      auto prim = node->primitive_;
      auto ret = DoInferShape(in_tensors, &out_tensors, op_param, prim);
      MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "ShapeInfoContainer Init failed.");
    }
  }
  auto ret = DetermineShapeVarInfos();
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "DetermineShapeVarInfos failed.");
  return RET_OK;
}

int ShapeInfoContainer::DoInferShape(const std::vector<Tensor *> &in_tensors, std::vector<Tensor *> *out_tensors,
                                     OpParameter *op_param, const void *primitive) {
  auto ret = KernelInferShape(in_tensors, *out_tensors, primitive, {}, lite::SCHEMA_CUR);
  if (ret == lite::RET_NOT_SUPPORT) {
    ret = KernelInferShape(in_tensors, *out_tensors, op_param);
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Infer shape failed.";
    return ret;
  }
  for (const auto out_tensor : *out_tensors) {
    var_tensor_shapes_[out_tensor].push_back(out_tensor->shape());
  }
  return RET_OK;
}

int ShapeInfoContainer::DetermineShapeVarInfos() {
  MS_CHECK_TRUE_MSG(kShapePrefixName, RET_NULL_PTR, "kShapePrefixName is a nullptr.");
  int index = 0;
  for (const auto &item : var_tensor_shapes_) {
    auto &tensor = item.first;
    auto &shapes = item.second;
    MS_CHECK_TRUE_MSG(!shapes.empty(), RET_ERROR, "Cannot get some tensor's shape.");
    auto shape = shapes.front();
    auto dims = shape.size();
    auto is_same_dim =
      std::all_of(shapes.begin(), shapes.end(), [dims](const std::vector<int> &item) { return item.size() == dims; });
    MS_CHECK_TRUE_MSG(is_same_dim, RET_ERROR, "Tensor's shape-dims-num are not same.");
    std::vector<std::string> shape_symbols;
    for (size_t i = 0; i < dims; ++i) {
      int dim = shape[i];
      std::vector<int> real_nums;
      auto is_same_pos =
        std::all_of(shapes.begin(), shapes.end(), [dim, i](const std::vector<int> &item) { return item[i] == dim; });
      if (is_same_pos) {
        shape_symbols.push_back(std::to_string(dim));
        continue;
      }
      (void)std::transform(shapes.begin(), shapes.end(), std::back_inserter(real_nums),
                           [i](const std::vector<int> &item) { return item[i]; });
      std::string shape_symbol;
      auto ret =
        std::find_if(shape_to_nums_.begin(), shape_to_nums_.end(),
                     [&real_nums](const std::pair<std::string, std::vector<int>> &a) { return real_nums == a.second; });
      if (ret != shape_to_nums_.end()) {
        shape_symbol = ret->first;
      }
      if (shape_symbol.empty()) {
        for (size_t scene_index = 0; scene_index < real_nums.size(); ++scene_index) {
          shapes_whole_scenes_[scene_index].push_back(real_nums[scene_index]);
        }
        shape_symbol = std::string(kShapePrefixName) + "[" + std::to_string(index++) + "]";
        shape_to_nums_[shape_symbol] = real_nums;
      }
      shape_symbols.push_back(shape_symbol);
    }
    shape_templates_[tensor] = shape_symbols;
  }
  return RET_OK;
}

std::vector<std::string> ShapeInfoContainer::GetTemplateShape(const Tensor *tensor) const {
  if (shape_templates_.find(tensor) == shape_templates_.end()) {
    return {};
  }
  return shape_templates_.at(tensor);
}

std::vector<int> ShapeInfoContainer::GetRealNums(const std::string &shape_var) const {
  if (IsNumber(shape_var)) {
    return {std::stoi(shape_var.c_str())};
  }
  if (shape_to_nums_.find(shape_var) == shape_to_nums_.end()) {
    return {};
  }
  return shape_to_nums_.at(shape_var);
}
}  // namespace mindspore::lite::micro
