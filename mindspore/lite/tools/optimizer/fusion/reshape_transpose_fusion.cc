/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/reshape_transpose_fusion.h"
#include <numeric>
#include <vector>
#include <unordered_map>
#include "ops/op_utils.h"
#include "ops/transpose.h"
#include "tools/lite_exporter/fetch_content.h"
#include "tools/optimizer/common/format_utils.h"
#include "nnacl/op_base.h"

namespace mindspore::opt {
namespace {
const auto &p1 = std::placeholders::_1;
}  // namespace

VectorRef ReshapeTransposeFusion::DefineReshapeTransposePattern() const {
  auto input = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input != nullptr, {});
  auto is_reshape = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape));
  MS_CHECK_TRUE_RET(is_reshape != nullptr, {});
  auto is_const = std::make_shared<CondVar>(IsParamOrValueNodeWithData);
  MS_CHECK_TRUE_RET(is_const != nullptr, {});
  auto reshape = VectorRef({is_reshape, input, is_const});
  auto is_transpose = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose));
  MS_CHECK_TRUE_RET(is_transpose != nullptr, {});
  auto is_const_perm = std::make_shared<CondVar>(IsParamOrValueNodeWithData);
  MS_CHECK_TRUE_RET(is_const_perm != nullptr, {});
  return VectorRef({is_transpose, reshape, is_const_perm});
}

VectorRef ReshapeTransposeFusion::DefineTransposeReshapePattern() const {
  auto input = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input != nullptr, {});
  auto is_transpose = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose));
  MS_CHECK_TRUE_RET(is_transpose != nullptr, {});
  auto is_const = std::make_shared<CondVar>(IsParamOrValueNodeWithData);
  MS_CHECK_TRUE_RET(is_const != nullptr, {});
  auto transpose = VectorRef({is_transpose, input, is_const});
  auto is_reshape = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape));
  MS_CHECK_TRUE_RET(is_reshape != nullptr, {});
  auto is_const_shape = std::make_shared<CondVar>(IsParamOrValueNodeWithData);
  MS_CHECK_TRUE_RET(is_const_shape != nullptr, {});
  return VectorRef({is_reshape, transpose, is_const_shape});
}

std::unordered_map<std::string, VectorRef> ReshapeTransposeFusion::DefinePatterns() const {
  std::unordered_map<std::string, VectorRef> patterns;
  patterns["ReshapeTranspose"] = DefineReshapeTransposePattern();
  patterns["TransposeReshape"] = DefineTransposeReshapePattern();
  return patterns;
}

bool CheckTransposeCanFused(const FuncGraphPtr &func_graph, const CNodePtr &transpose) {
  MS_ASSERT(func_graph != nullptr && transpose != nullptr);
  MS_CHECK_TRUE_RET(transpose->size() == kInputSizeThree, false);
  auto input_abstract = GetCNodeInputAbstract(transpose, 1);
  MS_CHECK_TRUE_RET(input_abstract != nullptr, false);
  ShapeVector input_shape;
  if (FetchShapeFromAbstract(input_abstract, &input_shape) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get shape from abstract failed.";
    return false;
  }
  auto output_abstract = transpose->abstract();
  MS_CHECK_TRUE_RET(output_abstract != nullptr, false);
  ShapeVector output_shape;
  if (FetchShapeFromAbstract(output_abstract, &output_shape) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get shape from abstract failed.";
    return false;
  }
  if (input_shape.empty() || std::find(input_shape.begin(), input_shape.end(), -1) != input_shape.end() ||
      output_shape.empty() || std::find(output_shape.begin(), output_shape.end(), -1) != output_shape.end()) {
    MS_LOG(WARNING) << "The input shape or output shape of transpose is invalid.";
    return false;
  }
  input_shape.erase(std::remove_if(input_shape.begin(), input_shape.end(), [](int64_t x) { return x == 1; }),
                    input_shape.end());
  output_shape.erase(std::remove_if(output_shape.begin(), output_shape.end(), [](int64_t x) { return x == 1; }),
                     output_shape.end());
  if (input_shape != output_shape) {
    return false;
  }
  return true;
}

std::vector<int> GetShapeOfReshape(const CNodePtr &reshape_cnode, bool *changed = nullptr) {
  MS_ASSERT(reshape_cnode != nullptr);
  lite::DataInfo data_info;
  if (lite::FetchConstData(reshape_cnode, kInputIndexTwo, converter::kFmkTypeMs, &data_info, true) != lite::RET_OK) {
    return {};
  }
  MS_CHECK_TRUE_RET(data_info.data_type_ == kNumberTypeInt || data_info.data_type_ == kNumberTypeInt32, {});
  std::vector<int> shape(data_info.data_.size() / C4NUM);
  if (memcpy_s(shape.data(), shape.size() * sizeof(int), data_info.data_.data(), data_info.data_.size()) != EOK) {
    return {};
  }
  auto abstract = GetCNodeInputAbstract(reshape_cnode, 1);
  MS_CHECK_TRUE_RET(abstract != nullptr, {});
  ShapeVector input_shape;
  if (FetchShapeFromAbstract(abstract, &input_shape) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get shape from abstract failed.";
    return {};
  }
  for (size_t i = 0; i < input_shape.size() && i < shape.size(); i++) {
    if (changed != nullptr && !(*changed) && shape[i] == 0) {
      *changed = true;
    }
    shape[i] = shape[i] == 0 ? input_shape[i] : shape[i];
  }
  return shape;
}

AnfNodePtr ReshapeTransposeFusion::ReshapeTransFusion(const FuncGraphPtr &func_graph, const CNodePtr &transpose) const {
  MS_ASSERT(func_graph != nullptr && transpose != nullptr);
  auto reshape = transpose->input(1);
  MS_CHECK_TRUE_RET(reshape != nullptr, nullptr);
  auto reshape_cnode = reshape->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_cnode != nullptr && reshape_cnode->size() == kInputSizeThree, nullptr);
  if (CheckPrimitiveType(reshape_cnode->input(1), prim::kPrimTranspose)) {
    return TransReshapeTransFusion(func_graph, transpose);
  }

  if (IsMultiOutputTensors(func_graph, reshape)) {
    return nullptr;
  }
  if (!CheckTransposeCanFused(func_graph, transpose)) {
    return nullptr;
  }
  std::vector<int> perm;
  if (GetTransposePerm(transpose, &perm) != RET_OK) {
    MS_LOG(ERROR) << "fetch transpose's perm failed.";
    return nullptr;
  }
  auto shape = GetShapeOfReshape(reshape_cnode);
  MS_CHECK_TRUE_RET(shape.size() == perm.size(), nullptr);
  std::vector<int> new_shape(shape.size());
  for (size_t i = 0; i < perm.size(); i++) {
    MS_CHECK_TRUE_RET(perm.at(i) >= 0 && static_cast<size_t>(perm.at(i)) < shape.size(), nullptr);
    new_shape.at(i) = shape.at(perm.at(i));
  }
  auto new_shape_param = BuildIntVecParameterNode(func_graph, new_shape, reshape->fullname_with_scope() + "_transpose");
  MS_CHECK_TRUE_RET(new_shape_param != nullptr, nullptr);
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, nullptr);
  if (transpose->abstract() != nullptr) {
    reshape->set_abstract(transpose->abstract()->Clone());
  }
  manager->SetEdge(reshape, kInputIndexTwo, new_shape_param);
  return reshape;
}

AnfNodePtr ReshapeTransposeFusion::TransReshapeFusion(const FuncGraphPtr &func_graph,
                                                      const CNodePtr &reshape_cnode) const {
  MS_ASSERT(func_graph != nullptr && reshape_cnode != nullptr);
  MS_CHECK_TRUE_RET(reshape_cnode->size() == kInputSizeThree, nullptr);
  auto transpose = reshape_cnode->input(1);
  MS_CHECK_TRUE_RET(transpose != nullptr, nullptr);
  auto transpose_cnode = transpose->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(transpose_cnode != nullptr, nullptr);
  if (!CheckTransposeCanFused(func_graph, transpose_cnode)) {
    return nullptr;
  }
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, nullptr);
  bool changed = false;
  auto shape = GetShapeOfReshape(reshape_cnode, &changed);
  if (changed) {
    MS_CHECK_TRUE_RET(!shape.empty(), nullptr);
    auto new_shape_param =
      BuildIntVecParameterNode(func_graph, shape, reshape_cnode->fullname_with_scope() + "_new_shape");
    MS_CHECK_TRUE_RET(new_shape_param != nullptr, nullptr);
    manager->SetEdge(reshape_cnode, kInputIndexTwo, new_shape_param);
  }

  MS_CHECK_TRUE_RET(transpose_cnode->size() == kInputSizeThree, nullptr);
  manager->SetEdge(reshape_cnode, 1, transpose_cnode->input(1));

  return reshape_cnode;
}

int FindFixedPositionOfReshape(const ShapeVector &input_shape, const ShapeVector &shape,
                               const std::vector<int> &pre_perm, const std::vector<int> &post_perm,
                               std::vector<size_t> *in_pos, std::vector<size_t> *out_pos) {
  size_t i = 0;
  size_t j = 0;
  std::vector<size_t> tmp_in_pos;
  std::vector<size_t> tmp_out_pos;
  while (i < input_shape.size() && j < shape.size()) {
    if (input_shape.at(i) == shape.at(j)) {
      tmp_in_pos.push_back(i++);
      tmp_out_pos.push_back(j++);
    } else {
      size_t in_num = input_shape.at(i++);
      size_t out_num = shape.at(j++);
      while (in_num != out_num) {
        if (in_num < out_num) {
          MS_CHECK_TRUE_RET(i < input_shape.size(), lite::RET_ERROR);
          in_num = in_num * input_shape.at(i++);
        } else {
          MS_CHECK_TRUE_RET(j < shape.size(), lite::RET_ERROR);
          out_num = out_num * shape.at(j++);
        }
      }
    }
  }
  for (auto ele : tmp_in_pos) {
    MS_CHECK_TRUE_RET(ele < pre_perm.size(), lite::RET_ERROR);
    in_pos->push_back(pre_perm.at(ele));
  }
  for (auto ele : tmp_out_pos) {
    auto itr = std::find(post_perm.begin(), post_perm.end(), ele);
    MS_CHECK_TRUE_RET(itr != post_perm.end(), lite::RET_ERROR);
    out_pos->push_back(itr - post_perm.begin());
  }
  return lite::RET_OK;
}

bool CheckPermAndShape(const std::vector<int> &input_shape, const std::vector<int> &output_shape,
                       const std::vector<int> &pre_perm, const std::vector<int> &post_perm,
                       const std::vector<size_t> &in_fixed_pos, const std::vector<size_t> &out_fixed_pos) {
  if (in_fixed_pos.empty() || out_fixed_pos.empty()) {
    return false;
  }
  for (size_t i = 0; i < in_fixed_pos.size() || i < out_fixed_pos.size(); i++) {
    size_t pre_num = 1;
    auto in_begin = i < in_fixed_pos.size() ? in_fixed_pos.at(i) : input_shape.size() - 1;
    auto in_end = i < in_fixed_pos.size() - 1 ? in_fixed_pos.at(i + 1) : input_shape.size();
    auto itr = std::find(pre_perm.begin(), pre_perm.end(), in_begin + 1) - 1;
    for (auto j = in_begin + 1; j < in_end && j < input_shape.size(); j++) {
      auto tmp_itr = std::find(pre_perm.begin(), pre_perm.end(), j);
      if (tmp_itr - itr != 1) {
        return false;
      }
      itr = tmp_itr;

      MS_CHECK_INT_MUL_NOT_OVERFLOW(static_cast<int>(pre_num), static_cast<int>(input_shape.at(j)), false);
      pre_num *= input_shape.at(j);
    }

    size_t post_num = 1;
    auto out_begin = i < out_fixed_pos.size() ? out_fixed_pos.at(i) : output_shape.size() - 1;
    auto out_end = i < out_fixed_pos.size() - 1 ? out_fixed_pos.at(i + 1) : output_shape.size();
    auto pos = out_begin + 1 < post_perm.size() ? post_perm.at(out_begin + 1) - 1 : 0;
    for (auto j = out_begin + 1; j < out_end && j < output_shape.size(); j++) {
      auto tmp_pos = post_perm.at(j);
      if (tmp_pos - pos != 1) {
        return false;
      }
      pos = tmp_pos;

      MS_CHECK_INT_MUL_NOT_OVERFLOW(static_cast<int>(post_num), static_cast<int>(output_shape.at(j)), false);
      post_num *= output_shape.at(j);
    }
    if (pre_num != post_num) {
      return false;
    }
  }
  return true;
}

bool CheckTransReshapeTransCanFused(const ShapeVector &input_shape, const ShapeVector &output_shape,
                                    const std::vector<int> &pre_perm, const std::vector<int> &post_perm) {
  if (input_shape.size() != pre_perm.size() || output_shape.size() != post_perm.size()) {
    return false;
  }
  std::vector<size_t> in_fixed_pos;
  std::vector<size_t> out_fixed_pos;
  if (FindFixedPositionOfReshape(input_shape, output_shape, pre_perm, post_perm, &in_fixed_pos, &out_fixed_pos) !=
      lite::RET_OK) {
    MS_LOG(ERROR) << "Find fixed position of reshape failed.";
    return false;
  }

  std::vector<int> pre_trans_in_shape;
  for (int i = 0; i < static_cast<int>(pre_perm.size()); i++) {
    auto itr = std::find(pre_perm.begin(), pre_perm.end(), i);
    MS_CHECK_TRUE_RET(itr != pre_perm.end(), false);
    pre_trans_in_shape.push_back(input_shape.at(itr - pre_perm.begin()));
  }
  std::vector<int> trans_out_shape;
  for (auto dim : post_perm) {
    MS_CHECK_TRUE_RET(static_cast<size_t>(dim) < output_shape.size(), false);
    trans_out_shape.push_back(output_shape.at(static_cast<size_t>(dim)));
  }

  return CheckPermAndShape(pre_trans_in_shape, trans_out_shape, pre_perm, post_perm, in_fixed_pos, out_fixed_pos);
}

STATUS DealReshapeWithMultiOutputs(const FuncGraphPtr &func_graph, const CNodePtr &reshape, const CNodePtr &transpose,
                                   const std::vector<int> &post_perm) {
  MS_ASSERT(func_graph != nullptr && reshape != nullptr && transpose != nullptr);
  std::vector<int> new_perm;
  std::vector<int> tmp_perm(post_perm.size());
  std::iota(tmp_perm.begin(), tmp_perm.end(), 0);
  for (auto ele : tmp_perm) {
    auto itr = std::find(post_perm.begin(), post_perm.end(), ele);
    MS_CHECK_TRUE_RET(itr != post_perm.end(), lite::RET_ERROR);
    new_perm.push_back(itr - post_perm.begin());
  }
  auto insert_trans_perm_param = BuildIntVecParameterNode(func_graph, new_perm, "transpose_perm");
  MS_CHECK_TRUE_RET(insert_trans_perm_param != nullptr, lite::RET_ERROR);
  auto transpose_prim = std::make_shared<ops::Transpose>();
  if (transpose_prim == nullptr) {
    MS_LOG(ERROR) << "Build reshape primitive failed.";
    return lite::RET_ERROR;
  }
  auto transpose_prim_c = transpose_prim->GetPrim();
  MS_CHECK_TRUE_RET(transpose_prim_c != nullptr, lite::RET_ERROR);
  auto value_node = NewValueNode(transpose_prim_c);
  MS_CHECK_TRUE_RET(value_node != nullptr, lite::RET_ERROR);
  auto new_trans = func_graph->NewCNode({value_node, transpose, insert_trans_perm_param});
  MS_CHECK_TRUE_RET(new_trans != nullptr, lite::RET_ERROR);
  auto output_node_list = GetRealNodeUsedList(func_graph, reshape);

  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, lite::RET_ERROR);
  for (auto output_node_pair : *output_node_list) {
    if (output_node_pair.first != transpose) {
      manager->SetEdge(output_node_pair.first, output_node_pair.second, new_trans);
    }
  }
  return lite::RET_OK;
}

AnfNodePtr ReshapeTransposeFusion::TransReshapeTransFusion(const FuncGraphPtr &func_graph,
                                                           const CNodePtr &trans_cnode) const {
  MS_ASSERT(func_graph != nullptr && trans_cnode != nullptr);
  MS_CHECK_TRUE_RET(trans_cnode->size() == kInputSizeThree, nullptr);
  auto reshape = trans_cnode->input(1);
  MS_CHECK_TRUE_RET(reshape != nullptr, nullptr);

  auto reshape_cnode = reshape->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_cnode != nullptr && reshape_cnode->size() == kInputSizeThree, nullptr);
  auto pre_trans = reshape_cnode->input(1);
  MS_CHECK_TRUE_RET(pre_trans != nullptr, nullptr);
  if (IsMultiOutputTensors(func_graph, pre_trans)) {
    return nullptr;
  }

  auto abstract = pre_trans->abstract();
  MS_CHECK_TRUE_RET(abstract != nullptr, nullptr);
  ShapeVector input_shape;
  if (FetchShapeFromAbstract(abstract, &input_shape) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get shape from abstract failed.";
    return nullptr;
  }
  auto reshape_abstract = reshape->abstract();
  MS_CHECK_TRUE_RET(reshape_abstract != nullptr, nullptr);
  ShapeVector output_shape;
  if (FetchShapeFromAbstract(reshape_abstract, &output_shape) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get shape from abstract failed.";
    return nullptr;
  }
  if (input_shape.empty() || std::find(input_shape.begin(), input_shape.end(), -1) != input_shape.end() ||
      output_shape.empty() || std::find(output_shape.begin(), output_shape.end(), -1) != output_shape.end()) {
    MS_LOG(WARNING) << "The input shape or output shape of reshape is invalid.";
    return nullptr;
  }

  std::vector<int> pre_perm;
  auto pre_trans_cnode = pre_trans->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(pre_trans_cnode != nullptr, nullptr);
  if (GetTransposePerm(pre_trans_cnode, &pre_perm) != RET_OK) {
    MS_LOG(ERROR) << "fetch transpose's perm failed.";
    return nullptr;
  }
  std::vector<int> post_perm;
  if (GetTransposePerm(trans_cnode, &post_perm) != RET_OK) {
    MS_LOG(ERROR) << "fetch transpose's perm failed.";
    return nullptr;
  }

  if (!CheckTransReshapeTransCanFused(input_shape, output_shape, pre_perm, post_perm)) {
    return nullptr;
  }

  if (IsMultiOutputTensors(func_graph, reshape) &&
      DealReshapeWithMultiOutputs(func_graph, reshape_cnode, trans_cnode, post_perm) != lite::RET_OK) {
    MS_LOG(ERROR) << "deal with multi-output reshape failed.";
    return nullptr;
  }
  std::vector<int> new_shape;
  for (auto ele : post_perm) {
    MS_CHECK_TRUE_RET(static_cast<size_t>(ele) < output_shape.size(), nullptr);
    new_shape.push_back(output_shape.at(ele));
  }
  auto new_shape_param =
    BuildIntVecParameterNode(func_graph, new_shape, reshape_cnode->fullname_with_scope() + "new_shape");
  MS_CHECK_TRUE_RET(new_shape_param != nullptr, nullptr);
  if (trans_cnode->abstract() != nullptr) {
    reshape->set_abstract(trans_cnode->abstract()->Clone());
  }
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, nullptr);
  manager->SetEdge(reshape, 1, pre_trans_cnode->input(1));
  manager->SetEdge(reshape, kInputIndexTwo, new_shape_param);
  return reshape;
}

AnfNodePtr ReshapeTransposeFusion::Process(const std::string &pattern_name, const mindspore::FuncGraphPtr &func_graph,
                                           const mindspore::AnfNodePtr &node, const mindspore::EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(cnode != nullptr, nullptr);
  if (IsMarkedTrainOp(cnode)) {
    return nullptr;
  }
  if (pattern_name == "ReshapeTranspose") {
    return ReshapeTransFusion(func_graph, cnode);
  } else if (pattern_name == "TransposeReshape") {
    return TransReshapeFusion(func_graph, cnode);
  }

  return nullptr;
}
}  // namespace mindspore::opt
