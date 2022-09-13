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
#include "tools/optimizer/fusion/mul_reduce_fusion.h"
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "tools/optimizer/common/format_utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/lite_exporter/fetch_content.h"
#include "ops/fusion/mat_mul_fusion.h"
#include "ops/squeeze.h"
#include "ops/op_name.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
namespace {
constexpr int kReciprocalFirstIndex = -1;
constexpr int kReciprocalSecondIndex = -2;
int CommonInferShape(const CNodePtr &cnode, const std::vector<ShapeVector> &in_shapes,
                     std::vector<ShapeVector> *out_shapes) {
  out_shapes->clear();
  (void)out_shapes->insert(out_shapes->begin(), in_shapes.begin(), in_shapes.end());
  return lite::RET_OK;
}

int ExpandDimsInferShape(const CNodePtr &cnode, const std::vector<ShapeVector> &in_shapes,
                         std::vector<ShapeVector> *out_shapes) {
  MS_ASSERT(cnode != nullptr);
  if (cnode->size() < kInputSizeThree || in_shapes.size() < kInputSizeTwo) {
    MS_LOG(ERROR) << "Expanddims should have two inputs.";
    return lite::RET_ERROR;
  }
  auto second_input = cnode->input(kInputIndexTwo);
  MS_CHECK_TRUE_MSG(second_input != nullptr, lite::RET_ERROR, "Expanddims's second input is a nullptr.");
  if (second_input->isa<CNode>()) {
    return lite::RET_NOT_SUPPORT;
  }
  lite::DataInfo data_info;
  auto ret = lite::FetchConstData(cnode, kInputIndexTwo, converter::kFmkTypeMs, &data_info, false);
  MS_CHECK_TRUE_MSG(ret == lite::RET_OK, lite::RET_ERROR, "Expanddims fetch second-input's data failed.");
  MS_CHECK_TRUE_MSG(data_info.data_ptr_ != nullptr, lite::RET_ERROR,
                    "Expanddims's second-input's data shouldn't a nullptr.");
  MS_CHECK_TRUE_MSG(data_info.data_type_ == kNumberTypeInt || data_info.data_type_ == kNumberTypeInt32, lite::RET_ERROR,
                    "Expanddims's second-input's data-type should be int.");
  auto element_num = std::accumulate(data_info.shape_.begin(), data_info.shape_.end(), 1L, std::multiplies<int64_t>());
  MS_CHECK_TRUE_MSG(element_num == 1, lite::RET_ERROR, "Expanddims's second-input should be a scalar.");
  auto axis = *static_cast<int *>(data_info.data_ptr_);
  auto first_shape = in_shapes.front();
  auto first_shape_size = static_cast<int>(first_shape.size());
  if (axis < 0) {
    axis = first_shape_size + axis + 1;
  }
  MS_CHECK_TRUE_MSG(axis >= 0 && axis <= first_shape_size, lite::RET_ERROR, "Expanddims's second-input is invalid.");
  out_shapes->clear();
  (void)first_shape.insert(first_shape.begin() + axis, 1);
  out_shapes->push_back(first_shape);
  return lite::RET_OK;
}

int GatherInferShape(const CNodePtr &cnode, const std::vector<ShapeVector> &in_shapes,
                     std::vector<ShapeVector> *out_shapes) {
  MS_ASSERT(cnode != nullptr);
  if (cnode->size() < kInputSizeFour || in_shapes.size() < kInputSizeThree) {
    MS_LOG(ERROR) << "Gther should have three inputs.";
    return lite::RET_ERROR;
  }
  auto third_input = cnode->input(kInputIndexThree);
  MS_CHECK_TRUE_MSG(third_input != nullptr, lite::RET_ERROR, "Gather's third input is a nullptr.");
  if (third_input->isa<CNode>()) {
    return lite::RET_NOT_SUPPORT;
  }
  lite::DataInfo data_info;
  auto ret = lite::FetchConstData(cnode, kInputIndexThree, converter::kFmkTypeMs, &data_info, false);
  MS_CHECK_TRUE_MSG(ret == lite::RET_OK, lite::RET_ERROR, "Gather fetch second-input's data failed.");
  auto element_num = std::accumulate(data_info.shape_.begin(), data_info.shape_.end(), 1L, std::multiplies<int64_t>());
  MS_CHECK_TRUE_MSG(element_num <= 1, lite::RET_ERROR, "Gather's second-input should be a scalar.");
  int axis{0};
  if (element_num == 1) {
    MS_CHECK_TRUE_MSG(data_info.data_ptr_ != nullptr, lite::RET_ERROR,
                      "Gather's second-input's data shouldn't a nullptr.");
    if (data_info.data_type_ == kNumberTypeInt || data_info.data_type_ == kNumberTypeInt32) {
      axis = *static_cast<int *>(data_info.data_ptr_);
    } else if (data_info.data_type_ == kNumberTypeInt64) {
      axis = *static_cast<int64_t *>(data_info.data_ptr_);
    } else {
      MS_LOG(ERROR) << "Gather's axis is invalid, which should be int or int64.";
      return lite::RET_ERROR;
    }
  }
  auto first_shape = in_shapes.front();
  auto first_shape_size = static_cast<int>(first_shape.size());
  if (axis < 0) {
    axis = first_shape_size + axis;
  }
  MS_CHECK_TRUE_MSG(axis >= 0 && axis < first_shape_size, lite::RET_ERROR, "Gather's axis out of range.");
  auto second_shape = in_shapes[1];
  ShapeVector out_shape;
  for (int i = 0; i < axis; ++i) {
    out_shape.push_back(first_shape[i]);
  }
  (void)out_shape.insert(out_shape.end(), second_shape.begin(), second_shape.end());
  for (int i = axis + 1; i < first_shape_size; ++i) {
    out_shape.push_back(first_shape[i]);
  }
  out_shapes->clear();
  out_shapes->push_back(out_shape);
  return lite::RET_OK;
}

int MulInferShape(const CNodePtr &cnode, const std::vector<ShapeVector> &in_shapes,
                  std::vector<ShapeVector> *out_shapes) {
  MS_ASSERT(cnode != nullptr);
  if (cnode->size() < kInputSizeThree || in_shapes.size() < kInputSizeTwo) {
    MS_LOG(ERROR) << "Mul should have two inputs.";
    return lite::RET_ERROR;
  }
  const auto &first_shape = in_shapes.front();
  const auto &second_shape = in_shapes[1];
  size_t out_shape_size = first_shape.size() >= second_shape.size() ? first_shape.size() : second_shape.size();
  ShapeVector first_shape_expand;
  for (size_t i = 0; i < (out_shape_size - first_shape.size()); ++i) {
    first_shape_expand.push_back(1);
  }
  (void)first_shape_expand.insert(first_shape_expand.end(), first_shape.begin(), first_shape.end());
  ShapeVector second_shape_expand;
  for (size_t i = 0; i < (out_shape_size - second_shape.size()); ++i) {
    second_shape_expand.push_back(1);
  }
  (void)second_shape_expand.insert(second_shape_expand.end(), second_shape.begin(), second_shape.end());
  ShapeVector out_shape;
  for (size_t i = 0; i < out_shape_size; ++i) {
    if (first_shape_expand[i] == second_shape_expand[i]) {
      out_shape.push_back(first_shape_expand[i]);
      continue;
    }
    if (first_shape_expand[i] == 1) {
      out_shape.push_back(second_shape_expand[i]);
      continue;
    }
    if (second_shape_expand[i] == 1) {
      out_shape.push_back(first_shape_expand[i]);
      continue;
    }
    MS_LOG(INFO) << "Mul cannot determine out-shape.";
    return lite::RET_NOT_SUPPORT;
  }
  out_shapes->clear();
  out_shapes->push_back(out_shape);
  return lite::RET_OK;
}

int ReshapeInferShape(const CNodePtr &cnode, const std::vector<ShapeVector> &in_shapes,
                      std::vector<ShapeVector> *out_shapes) {
  MS_ASSERT(cnode != nullptr);
  out_shapes->clear();
  if (cnode->size() < kInputSizeTwo) {
    (void)out_shapes->emplace_back();
    return lite::RET_OK;
  }
  if (in_shapes.size() < kInputSizeTwo) {
    MS_LOG(ERROR) << "Reshape should have two inputs.";
    return lite::RET_ERROR;
  }
  auto second_input = cnode->input(kInputIndexTwo);
  MS_CHECK_TRUE_MSG(second_input != nullptr, lite::RET_ERROR, "Reshape's second input is a nullptr.");
  if (second_input->isa<CNode>()) {
    return lite::RET_NOT_SUPPORT;
  }
  lite::DataInfo data_info;
  auto ret = lite::FetchConstData(cnode, kInputIndexTwo, converter::kFmkTypeMs, &data_info, false);
  MS_CHECK_TRUE_MSG(ret == lite::RET_OK, lite::RET_ERROR, "Reshape fetch second-input's data failed.");
  MS_CHECK_TRUE_MSG(data_info.shape_.size() <= 1, lite::RET_ERROR, "Reshape second-input should be <= 1D.");
  if (data_info.data_ptr_ == nullptr || (data_info.shape_.size() == 1 && data_info.shape_.front() == 0)) {
    (void)out_shapes->emplace_back();
  }
  auto element_num = std::accumulate(data_info.shape_.begin(), data_info.shape_.end(), 1L, std::multiplies<int64_t>());
  ShapeVector out_shape;
  if (data_info.data_type_ == kNumberTypeInt || data_info.data_type_ == kNumberTypeInt32) {
    for (int i = 0; i < element_num; ++i) {
      out_shape.push_back(*(static_cast<int *>(data_info.data_ptr_) + i));
    }
  } else if (data_info.data_type_ == kNumberTypeInt64) {
    for (int i = 0; i < element_num; ++i) {
      out_shape.push_back(*(static_cast<int64_t *>(data_info.data_ptr_) + i));
    }
  } else {
    return lite::RET_NOT_SUPPORT;
  }
  out_shapes->push_back(out_shape);
  return lite::RET_OK;
}

int SplitInferShape(const CNodePtr &cnode, const std::vector<ShapeVector> &in_shapes,
                    std::vector<ShapeVector> *out_shapes) {
  MS_ASSERT(cnode != nullptr);
  out_shapes->clear();
  if (cnode->size() < kInputSizeTwo || in_shapes.empty()) {
    MS_LOG(ERROR) << "Split should have one inputs.";
    return lite::RET_ERROR;
  }
  auto prim = GetCNodePrimitive(cnode);
  auto out_num = prim->GetAttr(ops::kOutputNum) == nullptr ? 0 : GetValue<int64_t>(prim->GetAttr(ops::kOutputNum));
  auto size_splits = prim->GetAttr(ops::kSizeSplits) == nullptr
                       ? std::vector<int64_t>{}
                       : GetValue<std::vector<int64_t>>(prim->GetAttr(ops::kSizeSplits));
  out_num = (out_num == 0 ? static_cast<int64_t>(size_splits.size()) : out_num);
  if (out_num <= 0) {
    return lite::RET_NOT_SUPPORT;
  }
  auto axis = prim->GetAttr(ops::kAxis) == nullptr ? 0 : GetValue<int64_t>(prim->GetAttr(ops::kAxis));
  auto &in_shape = in_shapes.front();
  axis = axis < 0 ? static_cast<int64_t>(in_shape.size()) + axis : axis;
  MS_CHECK_TRUE_MSG(axis >= 0 && axis < static_cast<int64_t>(in_shape.size()), lite::RET_ERROR,
                    "Split's axis is out of range.");
  ShapeVector out_shape = in_shape;
  if (size_splits.empty()) {
    MS_CHECK_TRUE_MSG(in_shape[axis] > 0 && in_shape[axis] % out_num == 0, lite::RET_ERROR,
                      "Split's dim doesn't match split-axis.");
    out_shape[axis] = in_shape[axis] / out_num;
    (void)out_shapes->insert(out_shapes->end(), out_num, out_shape);
  } else {
    for (auto v : size_splits) {
      out_shape[axis] = v;
      out_shapes->push_back(out_shape);
    }
  }
  return lite::RET_OK;
}
}  // namespace

int PreprocessorOfFusion::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  reduce_ops_.clear();
  op_shape_infos_.clear();
  auto is_dynamic = CheckIsDynamicModel(func_graph);
  if (!is_dynamic) {
    return lite::RET_NOT_SUPPORT;
  }
  auto ret = ProcessOps(func_graph);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Preprocess for mul-reduce-fusion failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

bool PreprocessorOfFusion::CheckIsDynamicModel(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(graph_input_shape != nullptr);
  auto graph_inputs = func_graph->get_inputs();
  lite::DataInfo data_info;
  bool is_dynamic{false};
  for (auto &input : graph_inputs) {
    if (!utils::isa<Parameter>(input)) {
      continue;
    }
    auto ret = lite::FetchFromDefaultParam(input->cast<ParameterPtr>(), converter::kFmkTypeMs, &data_info, false);
    if (ret != lite::RET_OK) {
      return false;
    }
    ShapeVector shape(data_info.shape_.begin(), data_info.shape_.end());
    is_dynamic = is_dynamic || std::any_of(shape.begin(), shape.end(), [](int64_t v) { return v == -1; });
    op_shape_infos_[input] = std::make_pair(std::vector<ShapeVector>{}, std::vector<ShapeVector>{shape});
  }
  return is_dynamic;
}

int PreprocessorOfFusion::ProcessOps(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(ops_can_infer != nullptr);
  std::set<std::string> support_ops = {prim::kPrimCast->name(),     prim::kPrimExpandDims->name(),
                                       prim::kPrimGather->name(),   prim::kPrimMulFusion->name(),
                                       prim::kPrimNotEqual->name(), prim::kPrimReduceFusion->name(),
                                       prim::kPrimReshape->name(),  prim::kPrimSplit->name()};
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto prim = GetCNodePrimitive(cnode);
    if (prim == nullptr) {
      continue;
    }
    auto op_type = prim->name();
    if (support_ops.find(op_type) == support_ops.end()) {
      continue;
    }
    auto origin_inputs = cnode->inputs();
    if (lite::RemoveIfDepend(cnode) != RET_OK) {
      cnode->set_inputs(origin_inputs);
      continue;
    }
    if (lite::RemoveIfMakeTuple(cnode)) {
      cnode->set_inputs(origin_inputs);
      continue;
    }
    RemoveIfMonad(cnode);
    auto current_inputs = cnode->inputs();
    bool can_infer = std::any_of(current_inputs.begin(), current_inputs.end(), [this](AnfNodePtr &anf_node) {
      return op_shape_infos_.find(anf_node) != op_shape_infos_.end() || !utils::isa<CNode>(anf_node);
    });
    if (!can_infer) {
      cnode->set_inputs(origin_inputs);
      continue;
    }
    if (op_type == prim::kPrimReduceFusion->name()) {
      cnode->set_inputs(origin_inputs);
      reduce_ops_.push_back(cnode);
      continue;
    }
    auto ret = DoInfer(cnode, op_type);
    cnode->set_inputs(origin_inputs);
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "error occurred when infer " << op_type;
      return ret;
    }
  }
  return lite::RET_OK;
}

int PreprocessorOfFusion::DoInfer(const CNodePtr &cnode, std::string op_type) {
  MS_ASSERT(cnode != nullptr);
  std::map<std::string, std::function<int(const CNodePtr &cnode, const std::vector<ShapeVector> &in_shapes,
                                          std::vector<ShapeVector> *out_shapes)>>
    infer_func = {
      {prim::kPrimCast->name(), CommonInferShape},     {prim::kPrimExpandDims->name(), ExpandDimsInferShape},
      {prim::kPrimGather->name(), GatherInferShape},   {prim::kPrimMulFusion->name(), MulInferShape},
      {prim::kPrimNotEqual->name(), CommonInferShape}, {prim::kPrimReshape->name(), ReshapeInferShape},
      {prim::kPrimSplit->name(), SplitInferShape}};
  if (infer_func.find(op_type) == infer_func.end()) {
    MS_LOG(ERROR) << "Current op: " << op_type << " doesn't support infer.";
    return lite::RET_ERROR;
  }
  std::vector<ShapeVector> in_shapes;
  lite::DataInfo data_info;
  for (size_t i = 1; i < cnode->size(); ++i) {
    auto input = cnode->input(i);
    if (input == nullptr) {
      continue;
    }
    if (utils::isa<CNode>(input)) {
      auto real_input_info = GetRealCertainVarInput(cnode, i);
      MS_CHECK_TRUE_MSG(real_input_info.first != nullptr, lite::RET_ERROR, "Current op is invalid.");
      if (op_shape_infos_.find(real_input_info.first) == op_shape_infos_.end()) {
        return lite::RET_OK;
      }
      auto &upper_node_out = op_shape_infos_[real_input_info.first].second;
      auto index = real_input_info.second;
      MS_CHECK_TRUE_MSG(index >= 0 && index < static_cast<int>(upper_node_out.size()), lite::RET_ERROR,
                        "Current op is invalid.");
      in_shapes.push_back(upper_node_out[index]);
    } else {
      auto ret = lite::FetchConstData(cnode, i, converter::kFmkTypeMs, &data_info, false);
      if (ret != lite::RET_OK) {
        MS_LOG(ERROR) << "Fetch constant info failed, " << cnode->fullname_with_scope();
        return lite::RET_ERROR;
      }
      ShapeVector in_shape(data_info.shape_.begin(), data_info.shape_.end());
      in_shapes.push_back(in_shape);
    }
  }
  auto func = infer_func[op_type];
  MS_ASSERT(func != nullptr);
  std::vector<ShapeVector> out_shapes;
  auto ret = func(cnode, in_shapes, &out_shapes);
  if (ret == lite::RET_NOT_SUPPORT) {
    return lite::RET_OK;
  }
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "current op is invalid, " << op_type;
    return lite::RET_ERROR;
  }
  op_shape_infos_[cnode] = std::make_pair(in_shapes, out_shapes);
  return lite::RET_OK;
}

bool MulReduceFusion::Run(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    return false;
  }
  auto ret = preprocessor_.Run(func_graph);
  if (ret == lite::RET_NOT_SUPPORT) {
    return true;
  }
  if (preprocessor_.Run(func_graph) != lite::RET_OK) {
    return false;
  }
  auto &reduce_ops = preprocessor_.GetReduceOps();
  for (auto reduce_op : reduce_ops) {
    ret = ProcessOp(func_graph, reduce_op);
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "mul-reduce fusion process failed.";
      return false;
    }
  }
  ret = PostProcess(func_graph);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "mul-reduce fusion post-process failed.";
    return false;
  }
  return true;
}

int MulReduceFusion::ProcessOp(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  auto is_meet_cond = CheckBasicCond(func_graph, cnode);
  if (!is_meet_cond) {
    return lite::RET_OK;
  }
  if (reduce_mode_ == ReduceMode::Reduce_Mean) {
    auto ret = ProcessGather();
    if (ret == lite::RET_NOT_SUPPORT) {
      return lite::RET_OK;
    }
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "Process Gather op failed.";
      return lite::RET_ERROR;
    }
  }
  if (!keep_dim_) {
    auto ret = GenerateSqueeze(func_graph, cnode);
    if (ret != lite::RET_OK) {
      return lite::RET_ERROR;
    }
  }
  auto ret = GenerateMatmul(func_graph, cnode);
  if (ret != lite::RET_OK) {
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

int MulReduceFusion::ProcessGather() {
  MS_ASSERT(gather_.size() > C1NUM);
  auto gather_table = gather_->input(1);
  if (gather_table == nullptr || utils::isa<CNode>(gather_table)) {
    return lite::RET_NOT_SUPPORT;
  }
  lite::DataInfo data_info;
  auto ret = lite::FetchConstData(gather_, 1, converter::kFmkTypeMs, &data_info, false);
  MS_CHECK_TRUE_MSG(ret == lite::RET_OK, lite::RET_ERROR, "Fetch const data of gather failed.");
  if (data_info.data_type_ != kNumberTypeFloat && data_info.data_type_ != kNumberTypeFloat32) {
    return lite::RET_NOT_SUPPORT;
  }
  if (data_info.data_ptr_ == nullptr) {
    return lite::RET_NOT_SUPPORT;
  }
  auto *float_data = static_cast<float *>(data_info.data_ptr_);
  auto element_num = std::accumulate(data_info.shape_.begin(), data_info.shape_.end(), 1L, std::multiplies<int64_t>());
  for (int64_t i = 0; i < element_num; ++i) {
    float_data[i] *= coeff_;
  }
  return lite::RET_OK;
}

int MulReduceFusion::PostProcess(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  if (squeeze_infos_.empty()) {
    return lite::RET_OK;
  }
  std::set<CNodePtr> concat_ops;
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  auto &node_users = manager->node_users();
  for (auto &squeeze : squeeze_infos_) {
    auto &node_user = node_users[squeeze.first];
    for (auto &user : node_user) {
      auto node = user.first;
      if (!utils::isa<CNode>(node)) {
        continue;
      }
      auto cnode = node->cast<CNodePtr>();
      if (CheckPrimitiveType(cnode, prim::kPrimConcat)) {
        (void)concat_ops.insert(cnode);
      }
    }
  }
  for (auto &concat : concat_ops) {
    auto ret = PostProcessSqueezeWithConcat(func_graph, concat);
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "mul-reduce-fusion's PostProcess failed.";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

int MulReduceFusion::PostProcessSqueezeWithConcat(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(cnode != nullptr);
  if (!CheckConcatOp(func_graph, cnode)) {
    return lite::RET_OK;
  }
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  for (int i = 1; i < static_cast<int>(cnode->size()); ++i) {
    manager->SetEdge(cnode, i, cnode->input(i)->cast<CNodePtr>()->input(1));
  }
  auto concat_prim = GetCNodePrimitive(cnode);
  MS_ASSERT(concat_prim != nullptr);
  (void)concat_prim->AddAttr(ops::kAxis, MakeValue<int64_t>(concat_axis_));
  auto &node_users = manager->node_users();
  auto &concat_users = node_users[cnode];
  CNodePtr post_squeeze{nullptr};
  for (auto &user : concat_users) {
    if (CheckPrimitiveType(user.first, prim::kPrimReshape)) {
      continue;
    }
    if (post_squeeze == nullptr) {
      auto squeeze = std::make_shared<ops::Squeeze>();
      MS_CHECK_TRUE_MSG(post_squeeze != nullptr, lite::RET_ERROR, "Squeeze create failed.");
      squeeze->set_axis(std::vector<int64_t>{axis_});
      auto squeeze_prim = squeeze->GetPrim();
      MS_CHECK_TRUE_MSG(squeeze_prim != nullptr, lite::RET_ERROR, "Squeeze create failed.");
      post_squeeze = func_graph->NewCNode(squeeze_prim, {cnode});
      MS_CHECK_TRUE_MSG(post_squeeze != nullptr, lite::RET_ERROR, "Squeeze-cnode create failed.");
      post_squeeze->set_fullname_with_scope(cnode->fullname_with_scope() + "/Squeeze");
    }
    manager->SetEdge(user.first, user.second, post_squeeze);
  }
  return lite::RET_OK;
}

int MulReduceFusion::GenerateMatmul(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(cnode != nullptr);
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, lite::RET_ERROR, "Manager is a nullptr.");
  auto mul_op = cnode->input(1)->cast<CNodePtr>();  // which has been checked before.
  if (exchange_) {
    manager->SetEdge(cnode, 1, mul_op->input(kInputIndexTwo));
    manager->SetEdge(cnode, kInputIndexTwo, mul_op->input(1));
  } else {
    manager->SetEdge(cnode, 1, mul_op->input(1));
    manager->SetEdge(cnode, kInputIndexTwo, mul_op->input(kInputIndexTwo));
  }
  auto matmul_prim = std::make_shared<ops::MatMulFusion>();
  MS_CHECK_TRUE_MSG(matmul_prim != nullptr, lite::RET_ERROR, "Matmul create failed.");
  auto matmul_prim_c = matmul_prim->GetPrim();
  MS_CHECK_TRUE_MSG(matmul_prim_c != nullptr, lite::RET_ERROR, "Matmul create failed.");
  matmul_prim->set_transpose_a(transpose_a_);
  matmul_prim->set_transpose_b(transpose_b_);
  MS_ASSERT(cnode->input(0) != nullptr);
  auto reduce_prim_carrier = cnode->input(0)->cast<ValueNodePtr>();
  MS_ASSERT(reduce_prim_carrier != nullptr);
  reduce_prim_carrier->set_value(matmul_prim_c);
  return lite::RET_OK;
}

int MulReduceFusion::GenerateSqueeze(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(cnode != nullptr);
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, lite::RET_ERROR, "Manager is a nullptr.");
  auto squeeze = std::make_shared<ops::Squeeze>();
  MS_CHECK_TRUE_MSG(squeeze != nullptr, lite::RET_ERROR, "Squeeze create failed.");
  squeeze->set_axis(std::vector<int64_t>{axis_});
  auto squeeze_prim = squeeze->GetPrim();
  MS_CHECK_TRUE_MSG(squeeze_prim != nullptr, lite::RET_ERROR, "Squeeze create failed.");
  auto squeeze_cnode = func_graph->NewCNode(squeeze_prim, {cnode});
  MS_CHECK_TRUE_MSG(squeeze_cnode != nullptr, lite::RET_ERROR, "Squeeze-cnode create failed.");
  auto mul_op = cnode->input(1);
  MS_ASSERT(mul_op != nullptr);
  squeeze_cnode->set_fullname_with_scope(mul_op->fullname_with_scope() + "/Squeeze");
  auto success = manager->Replace(cnode, squeeze_cnode);
  MS_CHECK_TRUE_MSG(success, lite::RET_ERROR, "Replace old node failed.");
  auto &shape_infos = preprocessor_.GetShapeContainer();
  MS_ASSERT(shape_infos.find(mul_op) != shape_infos.end());
  auto &out_shape_infos = shape_infos.at(mul_op).second;
  MS_ASSERT(!out_shape_infos.empty());
  squeeze_infos_[squeeze_cnode] = std::make_pair(axis_, out_shape_infos.front().size() - 1);
  return lite::RET_OK;
}

bool MulReduceFusion::CheckBasicCond(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  if (cnode->size() < kInputSizeThree) {
    return false;
  }
  if (IsMarkedTrainOp(cnode)) {
    return false;
  }
  auto prim = GetCNodePrimitive(cnode);
  MS_ASSERT(prim != nullptr);
  bool is_to_end = prim->GetAttr(ops::kReduceToEnd) != nullptr && GetValue<bool>(prim->GetAttr(ops::kReduceToEnd));
  if (is_to_end) {
    return false;
  }
  keep_dim_ = prim->GetAttr(ops::kKeepDims) != nullptr && GetValue<bool>(prim->GetAttr(ops::kKeepDims));
  auto mode_attr = prim->GetAttr(ops::kMode);
  if (mode_attr == nullptr) {
    return false;
  }
  reduce_mode_ = static_cast<int>(GetValue<int64_t>(mode_attr));
  if (reduce_mode_ != ReduceMode::Reduce_Sum && reduce_mode_ != ReduceMode::Reduce_Mean) {
    return false;
  }
  auto first_input = cnode->input(1);
  if (!utils::isa<CNode>(first_input)) {
    return false;
  }
  if (!CheckPrimitiveType(first_input, prim::kPrimMulFusion)) {
    return false;
  }
  if (IsMarkedTrainOp(first_input->cast<CNodePtr>())) {
    return false;
  }
  auto mul_prim = GetCNodePrimitive(first_input);
  MS_ASSERT(mul_prim != nullptr);
  auto act_type = mul_prim->GetAttr(ops::kActivationType) == nullptr
                    ? ActivationType::NO_ACTIVATION
                    : GetValue<int64_t>(mul_prim->GetAttr(ops::kActivationType));
  if (act_type != ActivationType::NO_ACTIVATION) {
    return false;
  }
  if (IsMultiOutputTensors(func_graph, first_input)) {
    return false;
  }
  bool is_axis_meet = CheckAxisCond(cnode);
  if (!is_axis_meet) {
    return false;
  }
  bool is_shape_meet = CheckShapeCond(cnode);
  if (!is_shape_meet) {
    return false;
  }
  return CheckGatherOp(func_graph, cnode);
}

bool MulReduceFusion::CheckAxisCond(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  auto &shape_container = preprocessor_.GetShapeContainer();
  auto first_input = cnode->input(1);
  if (shape_container.find(first_input) == shape_container.end()) {
    return false;
  }
  if (shape_container.at(first_input).second.empty()) {
    return false;
  }
  auto in_shape = shape_container.at(first_input).second.front();
  auto second_input = cnode->input(kInputIndexTwo);
  if (second_input == nullptr || utils::isa<CNode>(second_input)) {
    return false;
  }
  lite::DataInfo data_info;
  auto ret = lite::FetchConstData(cnode, kInputIndexTwo, converter::kFmkTypeMs, &data_info, false);
  MS_CHECK_TRUE_MSG(ret == lite::RET_OK, false, "Fetch reduceOp's axis failed.");
  auto element_num = std::accumulate(data_info.shape_.begin(), data_info.shape_.end(), 1L, std::multiplies<int64_t>());
  if (data_info.data_ptr_ == nullptr || element_num != 1) {
    return false;
  }
  if (data_info.data_type_ == kNumberTypeInt || data_info.data_type_ == kNumberTypeInt32) {
    axis_ = *(static_cast<int *>(data_info.data_ptr_));
  } else if (data_info.data_type_ == kNumberTypeInt64) {
    axis_ = static_cast<int>(*(static_cast<int64_t *>(data_info.data_ptr_)));
  } else {
    return false;
  }
  if (axis_ > 0) {
    axis_ -= static_cast<int>(in_shape.size());
  }
  if (axis_ != kReciprocalFirstIndex && axis_ != kReciprocalSecondIndex) {
    return false;
  }
  return true;
}

bool MulReduceFusion::CheckShapeCond(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  auto &shape_container = preprocessor_.GetShapeContainer();
  auto first_input = cnode->input(1);
  if (shape_container.find(first_input) == shape_container.end()) {
    return false;
  }
  if (shape_container.at(first_input).first.size() != kInputSizeTwo) {
    return false;
  }
  auto mul_in0_shape = shape_container.at(first_input).first.front();
  auto mul_in1_shape = shape_container.at(first_input).first.back();
  if (mul_in0_shape.size() < kInputSizeTwo || mul_in1_shape.size() < kInputSizeTwo) {
    return false;
  }
  if (mul_in0_shape.back() <= 0 || mul_in0_shape[mul_in0_shape.size() - C2NUM] <= 0 || mul_in1_shape.back() <= 0 ||
      mul_in1_shape[mul_in1_shape.size() - C2NUM] <= 0) {
    return false;
  }
  if (axis_ == kReciprocalFirstIndex) {
    if (mul_in0_shape.back() != mul_in1_shape.back() ||
        (mul_in0_shape[mul_in0_shape.size() - C2NUM] != 1 && mul_in1_shape[mul_in1_shape.size() - C2NUM] != 1)) {
      return false;
    }
    exchange_ = mul_in1_shape[mul_in1_shape.size() - C2NUM] != 1;
    transpose_a_ = false;
    transpose_b_ = true;
    MS_ASSERT(mul_in0_shape.back() != 0);
    coeff_ = 1.0f / mul_in0_shape.back();
    return true;
  }
  if (axis_ == kReciprocalSecondIndex) {
    if (mul_in0_shape[mul_in0_shape.size() - C2NUM] != mul_in1_shape[mul_in1_shape.size() - C2NUM] ||
        (mul_in0_shape.back() != 1 && mul_in1_shape.back() != 1)) {
      return false;
    }
    exchange_ = mul_in0_shape.back() != 1;
    transpose_a_ = true;
    transpose_b_ = false;
    MS_ASSERT(mul_in0_shape[mul_in0_shape.size() - C2NUM] != 0);
    coeff_ = 1.0f / mul_in0_shape[mul_in0_shape.size() - C2NUM];
    return true;
  }
  return false;
}

bool MulReduceFusion::CheckGatherOp(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  if (reduce_mode_ == ReduceMode::Reduce_Sum) {
    return true;
  }
  if (reduce_mode_ != ReduceMode::Reduce_Mean) {
    return false;
  }
  auto mul_op = cnode->input(1);
  if (!utils::isa<CNode>(mul_op)) {
    return false;
  }
  auto mul_op_cnode = mul_op->cast<CNodePtr>();
  for (size_t i = 1; i < mul_op_cnode->size(); ++i) {
    if (!utils::isa<CNode>(mul_op_cnode->input(i))) {
      continue;
    }
    if (CheckPrimitiveType(mul_op_cnode->input(i), prim::kPrimGather)) {
      gather_ = mul_op_cnode->input(i)->cast<CNodePtr>();
      break;
    }
  }
  if (gather_ == nullptr) {
    return false;
  }
  if (IsMarkedTrainOp(gather_)) {
    return false;
  }
  if (IsMultiOutputTensors(func_graph, gather_)) {
    return false;
  }
  return true;
}

bool MulReduceFusion::CheckConcatOp(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  int axis{0};
  int out_dims{0};
  for (size_t i = 1; i < cnode->size(); ++i) {
    auto in_node = cnode->input(i);
    if (!utils::isa<CNode>(in_node)) {
      return false;
    }
    auto in_cnode = in_node->cast<CNodePtr>();
    if (squeeze_infos_.find(in_cnode) == squeeze_infos_.end()) {
      return false;
    }
    if (IsMultiOutputTensors(func_graph, in_node)) {
      return false;
    }
    if (i == 1) {
      axis = squeeze_infos_[in_cnode].first;
      out_dims = squeeze_infos_[in_cnode].second;
    } else {
      if (squeeze_infos_[in_cnode].first != axis || squeeze_infos_[in_cnode].second != out_dims) {
        return false;
      }
    }
  }
  auto concat_prim = GetCNodePrimitive(cnode);
  MS_CHECK_TRUE_RET(concat_prim != nullptr, false);
  concat_axis_ = concat_prim->GetAttr(ops::kAxis) == nullptr
                   ? 0
                   : static_cast<int>(GetValue<int64_t>(concat_prim->GetAttr(ops::kAxis)));
  axis = axis < 0 ? axis + out_dims + 1 : axis;
  MS_CHECK_TRUE_RET(axis >= 0 && axis <= out_dims, false);
  concat_axis_ = concat_axis_ < 0 ? concat_axis_ + out_dims : concat_axis_;
  MS_CHECK_TRUE_RET(concat_axis_ >= 0 && concat_axis_ < out_dims, false);
  if (concat_axis_ >= axis) {
    ++concat_axis_;
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
