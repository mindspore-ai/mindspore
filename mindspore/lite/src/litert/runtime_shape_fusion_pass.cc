/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef RUNTIME_PASS_CLIP
#include "src/litert/runtime_shape_fusion_pass.h"
#include <set>
#include <queue>
#include <algorithm>
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "nnacl/op_base.h"

namespace mindspore::lite {
namespace {
constexpr size_t kInitialSize = 1024;
}  // namespace
int ShapeFusionPass::ConvertToShapeFusion(LiteGraph::Node *node) {
  MS_ASSERT(node != nullptr);
  auto input_tensor = src_tensors_->at(node->input_indices_.front());
  MS_CHECK_TRUE_RET(input_tensor != nullptr, RET_ERROR);
  auto shape = input_tensor->shape();
  if (shape.empty() || std::find(shape.begin(), shape.end(), -1) != shape.end()) {
    MS_LOG(INFO) << "The input shape is invalid.";
    return RET_ERROR;
  }

  flatbuffers::FlatBufferBuilder fbb(kInitialSize);
  auto val_offset = schema::CreateCustomDirect(fbb, "ShapeFusion");
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(PrimType::PrimType_Custom), val_offset.o);
  fbb.Finish(prim_offset);
  void *prim = malloc(fbb.GetSize());
  if (prim == nullptr) {
    MS_LOG(ERROR) << "malloc primitive failed.";
    return RET_ERROR;
  }
  memcpy(prim, fbb.GetBufferPointer(), fbb.GetSize());
  lite_model_->node_bufs_.push_back(prim);
  fbb.Clear();

  auto shape_fusion_prim = flatbuffers::GetRoot<schema::Primitive>(prim);
  MS_CHECK_TRUE_RET(shape_fusion_prim != nullptr, RET_ERROR);
  ShapeFusionMatrix shape_fusion_matrix(shape.size());
  MS_CHECK_TRUE_RET(!node->output_indices_.empty(), RET_ERROR);
  shape_fusion_matrices_[node->output_indices_.front()] = shape_fusion_matrix;
  auto shape_fusion_matrix_tensor = BuildTensorFromShapeFusionMatrix(shape_fusion_matrix);
  MS_CHECK_TRUE_RET(shape_fusion_matrix_tensor != nullptr, RET_ERROR);

  node->name_ += "_fusion";
  node->primitive_ = shape_fusion_prim;
  node->node_type_ = PrimType::PrimType_Inner_ShapeFusion;
  node->input_indices_.push_back(src_tensors_->size());
  src_tensors_->push_back(shape_fusion_matrix_tensor);
  return RET_OK;
}

Tensor *ShapeFusionPass::BuildTensorFromShapeFusionMatrix(const ShapeFusionMatrix &shape_fusion_matrix) {
  MS_CHECK_TRUE_RET(!shape_fusion_matrix.shape_matrix.empty(), nullptr);
  std::vector<int> matrix_shape;
  if (shape_fusion_matrix.shape_matrix.size() != 1 || !shape_fusion_matrix.scalar) {
    matrix_shape.push_back(static_cast<int>(shape_fusion_matrix.shape_matrix.size()));
  }
  matrix_shape.push_back(static_cast<int>(shape_fusion_matrix.shape_matrix.front().size()));
  auto tensor = new (std::nothrow) Tensor(kNumberTypeFloat32, matrix_shape, NUM_OF_FORMAT, Category::CONST_TENSOR);
  MS_CHECK_TRUE_RET(tensor != nullptr, nullptr);
  auto matrix_data = tensor->MutableData();
  if (matrix_data == nullptr) {
    MS_LOG(ERROR) << "Mutable data failed for tensor: " << tensor->tensor_name();
    delete tensor;
    return nullptr;
  }
  for (size_t row = 0; row < shape_fusion_matrix.shape_matrix.size(); row++) {
    auto dst_data = reinterpret_cast<float *>(matrix_data) + row * shape_fusion_matrix.shape_matrix.front().size();
    memcpy(dst_data, shape_fusion_matrix.shape_matrix.at(row).data(),
           shape_fusion_matrix.shape_matrix.front().size() * sizeof(float));
  }
  return tensor;
}

int ShapeFusionPass::FusePostNodes(LiteGraph::Node *node, size_t subgraph_index) {
  // fuse arithmetic/concat/gather/squeeze/unsqueeze/shape/cast
  MS_ASSERT(node != nullptr);
  std::queue<LiteGraph::Node *> candidate_nodes;
  auto output_index = node->output_indices_.front();
  MS_CHECK_TRUE_RET(used_nodes_.find(output_index) != used_nodes_.end(), RET_ERROR);
  std::vector<uint32_t> visited_outputs;
  for (auto out_node : used_nodes_[output_index]) {
    if (CheckCanFused(node, out_node, output_index, subgraph_index)) {
      candidate_nodes.push(out_node);
    }
    visited_outputs.push_back(output_index);
  }

  while (!candidate_nodes.empty()) {
    auto output_node = candidate_nodes.front();
    candidate_nodes.pop();
    std::vector<uint32_t> used_outputs;
    if (DoFuse(node, output_node, &used_outputs, subgraph_index) != RET_OK) {
      MS_LOG(WARNING) << "Fused to shape fusion failed: " << output_node->name_;
      continue;
    }
    // remove unused input and output
    for (auto original_output : used_outputs) {
      MS_CHECK_TRUE_RET(used_nodes_.find(original_output) != used_nodes_.end(), RET_ERROR);
      if (used_nodes_[original_output].empty()) {
        auto remove_itr = std::find(node->output_indices_.begin(), node->output_indices_.end(), original_output);
        if (remove_itr == node->output_indices_.end()) {
          MS_LOG(ERROR) << "can not find original output";
          return RET_ERROR;
        }
        node->output_indices_.erase(remove_itr);
        node->input_indices_.erase(node->input_indices_.begin() + (remove_itr - node->output_indices_.begin()) + 1);
      }
    }

    for (auto idx : node->output_indices_) {
      MS_CHECK_TRUE_RET(used_nodes_.find(idx) != used_nodes_.end(), RET_ERROR);
      if (std::find(visited_outputs.begin(), visited_outputs.end(), idx) != visited_outputs.end()) {
        continue;
      }
      visited_outputs.push_back(idx);
      for (auto out_node : used_nodes_[idx]) {
        if (CheckCanFused(node, out_node, idx, subgraph_index)) {
          candidate_nodes.push(out_node);
        }
      }
    }
  }
  return RET_OK;
}

bool ShapeFusionPass::CheckArithmetic(const LiteGraph::Node *shape_fusion, const LiteGraph::Node *post_node,
                                      uint32_t input_idx) {
  MS_ASSERT(shape_fusion != nullptr && post_node != nullptr);
  auto type = post_node->node_type_;
  if (is_div_ && type != schema::PrimitiveType_DivFusion) {
    // couldn't fuse add/sub/mul+div or add/sub/div+mul, because it maybe change indivisible to divisible.
    return false;
  }
  MS_CHECK_TRUE_RET(post_node->input_indices_.size() == kInputSize1, false);
  auto input1_index =
    post_node->input_indices_.at(0) == input_idx ? post_node->input_indices_.at(1) : post_node->input_indices_.at(0);
  auto tensor = src_tensors_->at(input1_index);
  MS_CHECK_TRUE_RET(tensor != nullptr, false);
  if (tensor->IsConst()) {
    return true;
  }
  auto shape_fusion_outputs = shape_fusion->output_indices_;
  auto fused_output =
    std::find(shape_fusion_outputs.begin(), shape_fusion_outputs.end(), input1_index) != shape_fusion_outputs.end();
  return fused_output && (type == schema::PrimitiveType_AddFusion || type == schema::PrimitiveType_SubFusion);
}

bool ShapeFusionPass::CheckCanFused(const LiteGraph::Node *shape_fusion, const LiteGraph::Node *post_node,
                                    uint32_t input_idx, size_t subgraph_index) {
  MS_ASSERT(shape_fusion != nullptr && post_node != nullptr);
  MS_CHECK_TRUE_RET(subgraph_index < lite_model_->graph_.sub_graphs_.size(), false);
  auto subgraph = lite_model_->graph_.sub_graphs_.at(subgraph_index);
  MS_CHECK_TRUE_RET(subgraph != nullptr, false);
  auto &subgraph_node_indices = subgraph->node_indices_;
  bool belong_to_current_subgraph = std::any_of(subgraph_node_indices.begin(), subgraph_node_indices.end(),
                                                [&](uint32_t idx) { return all_nodes_->at(idx) == post_node; });
  if (!belong_to_current_subgraph) {
    return false;
  }
  auto shape_fusion_outputs = shape_fusion->output_indices_;
  switch (post_node->node_type_) {
    case schema::PrimitiveType_Cast: {
      MS_CHECK_TRUE_RET(post_node->input_indices_.size() == kInputSize1, false);
      auto dst_type_tensor = src_tensors_->at(post_node->input_indices_.at(1));
      MS_CHECK_TRUE_RET(dst_type_tensor != nullptr && dst_type_tensor->data() != nullptr, false);
      auto data_type = reinterpret_cast<int *>(dst_type_tensor->data())[0];
      return data_type == kNumberTypeInt || data_type == kNumberTypeInt32;
    }
    case schema::PrimitiveType_AddFusion:
    case schema::PrimitiveType_SubFusion:
    case schema::PrimitiveType_MulFusion:
    case schema::PrimitiveType_DivFusion:
      return CheckArithmetic(shape_fusion, post_node, input_idx);
    case schema::PrimitiveType_Concat: {
      bool is_supported =
        std::all_of(post_node->input_indices_.begin(), post_node->input_indices_.end(), [&](uint32_t idx) {
          auto tensor = src_tensors_->at(idx);
          return tensor->IsConst() ||
                 std::find(shape_fusion_outputs.begin(), shape_fusion_outputs.end(), idx) != shape_fusion_outputs.end();
        });
      return is_supported;
    }
    case schema::PrimitiveType_Gather:
    case schema::PrimitiveType_Squeeze:
    case schema::PrimitiveType_Unsqueeze:
    case schema::PrimitiveType_Shape:
      return true;
    default:
      break;
  }
  return false;
}

int ShapeFusionPass::DoFuse(LiteGraph::Node *shape_fusion, const LiteGraph::Node *post_node,
                            std::vector<uint32_t> *input_indices, size_t subgraph_index) {
  MS_ASSERT(shape_fusion != nullptr && post_node != nullptr && input_indices != nullptr);
  ShapeFusionMatrix shape_fusion_matrix;
  auto type = post_node->node_type_;
  if (type == schema::PrimitiveType_AddFusion || type == schema::PrimitiveType_SubFusion ||
      type == schema::PrimitiveType_MulFusion || type == schema::PrimitiveType_DivFusion ||
      type == schema::PrimitiveType_Concat) {
    if (GenerateFusedShapeFusionMatrix(shape_fusion, post_node, input_indices, &shape_fusion_matrix) != RET_OK) {
      MS_LOG(WARNING) << "GenerateFusedShapeMatrix failed while fuse op: " << post_node->name_;
      return RET_ERROR;
    }
  } else {
    auto input_index = post_node->input_indices_.front();
    MS_CHECK_TRUE_RET(shape_fusion_matrices_.find(input_index) != shape_fusion_matrices_.end(), RET_ERROR);
    shape_fusion_matrix = shape_fusion_matrices_[input_index];
    input_indices->push_back(input_index);
    if (UpdateShapeFusionMatrix(post_node, &shape_fusion_matrix) != RET_OK) {
      MS_LOG(WARNING) << "UpdateShapeMatrix failed while fuse op: " << post_node->name_;
      return RET_ERROR;
    }
  }

  // generate matrix_tensor, and update input_indices and output_indices
  auto tensor = BuildTensorFromShapeFusionMatrix(shape_fusion_matrix);
  MS_CHECK_TRUE_RET(tensor != nullptr, RET_ERROR);
  shape_fusion->input_indices_.push_back(src_tensors_->size());
  src_tensors_->push_back(tensor);
  auto output_index = post_node->output_indices_.front();
  shape_fusion->output_indices_.push_back(output_index);
  shape_fusion_matrices_[output_index] = shape_fusion_matrix;

  MS_CHECK_TRUE_RET(subgraph_index < lite_model_->graph_.sub_graphs_.size(), RET_ERROR);
  auto subgraph = lite_model_->graph_.sub_graphs_.at(subgraph_index);
  MS_CHECK_TRUE_RET(subgraph != nullptr, RET_ERROR);
  auto &subgraph_node_indices = subgraph->node_indices_;
  size_t node_index = std::find(all_nodes_->begin(), all_nodes_->end(), post_node) - all_nodes_->begin();
  MS_CHECK_TRUE_RET(node_index != all_nodes_->size(), RET_ERROR);
  auto indice_itr = std::find(subgraph_node_indices.begin(), subgraph_node_indices.end(), node_index);
  MS_CHECK_TRUE_RET(indice_itr != subgraph_node_indices.end(), RET_ERROR);
  subgraph_node_indices.erase(indice_itr);
  for (auto idx : *input_indices) {
    MS_CHECK_TRUE_RET(used_nodes_.find(idx) != used_nodes_.end(), RET_ERROR);
    auto &used_nodes = used_nodes_[idx];
    auto itr = std::find(used_nodes.begin(), used_nodes.end(), post_node);
    MS_CHECK_TRUE_RET(itr != used_nodes.end(), RET_ERROR);
    used_nodes.erase(itr);
  }
  return RET_OK;
}

int ShapeFusionPass::GenerateFusedShapeFusionMatrix(LiteGraph::Node *shape_fusion, const LiteGraph::Node *post_node,
                                                    std::vector<uint32_t> *input_indices,
                                                    ShapeFusionMatrix *shape_fusion_matrix) {
  MS_ASSERT(shape_fusion != nullptr && post_node != nullptr && shape_fusion_matrix != nullptr);
  std::vector<uint32_t> fused_inputs;
  std::set<uint32_t> shape_fusion_outputs(shape_fusion->output_indices_.begin(), shape_fusion->output_indices_.end());
  std::set<uint32_t> post_inputs(post_node->input_indices_.begin(), post_node->input_indices_.end());
  std::set_intersection(post_inputs.begin(), post_inputs.end(), shape_fusion_outputs.begin(),
                        shape_fusion_outputs.end(), std::inserter(fused_inputs, fused_inputs.begin()));
  MS_CHECK_TRUE_RET(!fused_inputs.empty(), RET_ERROR);
  MS_CHECK_TRUE_RET(shape_fusion_matrices_.find(fused_inputs.at(0)) != shape_fusion_matrices_.end(), RET_ERROR);

  *shape_fusion_matrix = shape_fusion_matrices_[fused_inputs.at(0)];
  for (size_t i = 0; i < post_node->input_indices_.size(); i++) {
    ShapeFusionMatrix const_matrix;
    auto input_index = post_node->input_indices_.at(i);
    if (std::find(shape_fusion->output_indices_.begin(), shape_fusion->output_indices_.end(), input_index) !=
        shape_fusion->output_indices_.end()) {
      MS_CHECK_TRUE_RET(shape_fusion_matrices_.find(input_index) != shape_fusion_matrices_.end(), RET_ERROR);
      const_matrix = shape_fusion_matrices_[input_index];
      input_indices->push_back(input_index);
    } else {
      std::vector<size_t> shape = {shape_fusion_matrix->shape_matrix.size(),
                                   shape_fusion_matrix->shape_matrix.front().size()};
      auto const_tensor = src_tensors_->at(input_index);
      MS_CHECK_TRUE_RET(const_tensor != nullptr && const_tensor->data() != nullptr, RET_ERROR);
      if (GetFusionMatrixFromConstantTensor(const_tensor, shape, post_node->node_type_, &const_matrix) != RET_OK) {
        MS_LOG(ERROR) << "GetMatrixFromConstantTensor failed.";
        return RET_ERROR;
      }
    }
    if (i == 0) {
      *shape_fusion_matrix = const_matrix;
      continue;
    }
    if (post_node->node_type_ == schema::PrimitiveType_Concat) {
      shape_fusion_matrix->Append(const_matrix);
    } else {
      shape_fusion_matrix->Arithmetic(const_matrix, static_cast<schema::PrimitiveType>(post_node->node_type_));
    }
  }
  return RET_OK;
}

int ShapeFusionPass::UpdateShapeFusionMatrix(const LiteGraph::Node *post_node, ShapeFusionMatrix *shape_fusion_matrix) {
  MS_ASSERT(post_node != nullptr && shape_fusion_matrix != nullptr);
  switch (post_node->node_type_) {
    case schema::PrimitiveType_Cast:
      break;
    case schema::PrimitiveType_Gather: {
      auto indices_tensor = src_tensors_->at(post_node->input_indices_.at(1));
      MS_CHECK_TRUE_RET(indices_tensor != nullptr && indices_tensor->data() != nullptr, RET_ERROR);
      MS_CHECK_TRUE_RET(
        indices_tensor->data_type() == kNumberTypeInt || indices_tensor->data_type() == kNumberTypeInt32, RET_ERROR);
      std::vector<int> indices(indices_tensor->ElementsNum());
      memcpy(indices.data(), indices_tensor->data(), indices_tensor->Size());
      if (shape_fusion_matrix->Gather(indices) != RET_OK) {
        MS_LOG(ERROR) << "Fuse gather failed.";
        return RET_ERROR;
      }
      shape_fusion_matrix->scalar = indices_tensor->category() == CONST_SCALAR ? true : false;
    } break;
    case schema::PrimitiveType_Squeeze: {
      MS_CHECK_TRUE_RET(shape_fusion_matrix->scalar == false, RET_ERROR);
      shape_fusion_matrix->scalar = true;
    } break;
    case schema::PrimitiveType_Unsqueeze: {
      MS_CHECK_TRUE_RET(shape_fusion_matrix->scalar == true, RET_ERROR);
      shape_fusion_matrix->scalar = false;
    } break;
    case schema::PrimitiveType_Shape: {
      std::vector<float> shape_vec(shape_fusion_matrix->shape_matrix.front().size(), 0);
      shape_vec.at(shape_vec.size() - 1) = static_cast<float>(shape_fusion_matrix->shape_matrix.size());
      shape_fusion_matrix->shape_matrix = {shape_vec};
      shape_fusion_matrix->scalar = true;
    } break;
    default:
      MS_LOG(WARNING) << "Unsupported to fuse op: " << post_node->node_type_;
      return RET_ERROR;
  }
  return RET_OK;
}

int ShapeFusionPass::GetFusionMatrixFromConstantTensor(const lite::Tensor *tensor, const std::vector<size_t> &shape,
                                                       int node_type, ShapeFusionMatrix *constant_matrix) {
  MS_ASSERT(tensor != nullptr && tensor->data() != nullptr && constant_matrix != nullptr);
  MS_CHECK_TRUE_RET(tensor->data_type() == kNumberTypeInt || tensor->data_type() == kNumberTypeInt32, RET_ERROR);
  std::vector<int> value(tensor->ElementsNum());
  memcpy(value.data(), tensor->data(), tensor->Size());
  std::vector<std::vector<float>> shape_matrix;
  switch (node_type) {
    case schema::PrimitiveType_AddFusion:
    case schema::PrimitiveType_SubFusion: {
      std::vector<float> row_vec(shape.at(1));
      if (value.size() == shape.at(0)) {
        std::transform(value.begin(), value.end(), std::back_inserter(shape_matrix), [&row_vec](int ele) {
          row_vec.at(row_vec.size() - 1) = static_cast<float>(ele);
          return row_vec;
        });
      } else {
        MS_CHECK_TRUE_RET(value.size() == 1, RET_ERROR);
        row_vec.at(row_vec.size() - 1) = static_cast<float>(value.at(0));
        shape_matrix = std::vector<std::vector<float>>(shape.at(0), row_vec);
      }
    } break;
    case schema::PrimitiveType_MulFusion:
    case schema::PrimitiveType_DivFusion: {
      if (value.size() == shape.at(0)) {
        std::transform(value.begin(), value.end(), std::back_inserter(shape_matrix), [&shape](int ele) {
          std::vector<float> row_vec(shape.at(1), static_cast<float>(ele));
          return row_vec;
        });
      } else {
        MS_CHECK_TRUE_RET(value.size() == 1, RET_ERROR);
        std::vector<float> row_vec(shape.at(1), static_cast<float>(value.at(0)));
        shape_matrix = std::vector<std::vector<float>>(shape.at(0), row_vec);
      }
    } break;
    case schema::PrimitiveType_Concat: {
      std::vector<float> row_vec(shape.at(1));
      std::transform(value.begin(), value.end(), std::back_inserter(shape_matrix), [&row_vec](int ele) {
        row_vec.at(row_vec.size() - 1) = static_cast<float>(ele);
        return row_vec;
      });
    } break;
    default:
      MS_LOG(ERROR) << "Unsupported to generate constant shape matrix for node type: " << node_type;
      return RET_ERROR;
  }
  constant_matrix->shape_matrix = shape_matrix;
  return RET_OK;
}
}  // namespace mindspore::lite
#endif
