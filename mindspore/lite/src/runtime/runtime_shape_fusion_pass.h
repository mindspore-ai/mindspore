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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_RUNTIME_SHAPE_FUSION_PASS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_RUNTIME_SHAPE_FUSION_PASS_H_

#include <map>
#include <vector>
#include <algorithm>
#include "src/runtime/lite_model.h"
#include "src/common/tensor_util.h"
#include "schema/ops_generated.h"
#include "schema/model_generated.h"

namespace mindspore::lite {
#ifndef RUNTIME_PASS_CLIP
struct ShapeFusionMatrix {
  ShapeFusionMatrix() {}
  explicit ShapeFusionMatrix(size_t dim) {
    for (size_t i = 0; i < dim; ++i) {
      std::vector<float> row;
      for (size_t j = 0; j < dim; ++j) {
        row.push_back(i == j ? 1 : 0);
      }
      row.push_back(0);
      shape_matrix.push_back(row);
    }
  }

  void Append(const ShapeFusionMatrix &other) {
    for (auto row : other.shape_matrix) {
      shape_matrix.push_back(row);
    }
  }

  void Gather(const std::vector<int> &indices) {
    auto src_matrix = shape_matrix;
    shape_matrix.clear();
    for (auto idx : indices) {
      shape_matrix.push_back(src_matrix.at(idx));
    }
  }

  void Arithmetic(const ShapeFusionMatrix &other, schema::PrimitiveType type) {
    for (size_t i = 0; i < shape_matrix.size(); i++) {
      for (size_t j = 0; j < shape_matrix.front().size(); j++) {
        switch (type) {
          case schema::PrimitiveType_AddFusion:
            shape_matrix[i][j] += other.shape_matrix[i][j];
            break;
          case schema::PrimitiveType_SubFusion:
            shape_matrix[i][j] -= other.shape_matrix[i][j];
            break;
          case schema::PrimitiveType_MulFusion:
            shape_matrix[i][j] *= other.shape_matrix[i][j];
            break;
          case schema::PrimitiveType_DivFusion:
            shape_matrix[i][j] /= other.shape_matrix[i][j];
            break;
          default:
            break;
        }
      }
    }
  }
  std::vector<std::vector<float>> shape_matrix;
  bool scalar = false;
};
#endif

class ShapeFusionPass {
 public:
  ShapeFusionPass(LiteModel *model, std::vector<lite::Tensor *> *src_tensors)
      : lite_model_(model), all_nodes_(&(model->all_nodes_)), src_tensors_(src_tensors) {
    MS_ASSERT(model != nullptr && src_tensors != nullptr);
    for (auto node : model->all_nodes_) {
      for (auto input_idx : node->input_indices_) {
        used_nodes_[input_idx].push_back(node);
      }
    }
  }
  ~ShapeFusionPass() = default;

  void Run(Model::Node *node, size_t subgraph_index) {
#ifndef RUNTIME_PASS_CLIP
    if (ConvertToShapeFusion(node) != RET_OK) {
      MS_LOG(WARNING) << "Convert to built-in shape failed: " << node->name_;
    } else if (FusePostNodes(node, subgraph_index) != RET_OK) {
      MS_LOG(WARNING) << "Fused to built-in shape failed: " << node->name_;
    }
    std::transform(node->output_indices_.begin(), node->output_indices_.end(),
                   std::back_inserter(shape_fusion_outputs_),
                   [&](uint32_t idx) { return this->src_tensors_->at(idx); });
#endif
  }
  void FreeOutputTensorDataOfFusedShape() {
#if !defined(RUNTIME_PASS_CLIP)
    for (auto tensor : shape_fusion_outputs_) {
      tensor->FreeData();
      tensor->set_category(VAR);
    }
#endif
  }

 private:
#ifndef RUNTIME_PASS_CLIP
  int ConvertToShapeFusion(Model::Node *node);
  int FusePostNodes(Model::Node *node, size_t subgraph_index);
  Tensor *BuildTensorFromShapeFusionMatrix(const ShapeFusionMatrix &shape_fusion_matrix);
  bool CheckCanFused(const Model::Node *shape_fusion, const Model::Node *post_node, uint32_t input_idx,
                     size_t subgraph_index);
  int DoFuse(Model::Node *shape_fusion, const Model::Node *post_node, std::vector<uint32_t> *input_indices,
             size_t subgraph_index);
  int GenerateFusedShapeFusionMatrix(Model::Node *shape_fusion, const Model::Node *post_node,
                                     std::vector<uint32_t> *input_indices, ShapeFusionMatrix *shape_fusion_matrix);
  int UpdateShapeFusionMatrix(const Model::Node *post_node, ShapeFusionMatrix *shape_fusion_matrix);
  int GetFusionMatrixFromConstantTensor(const lite::Tensor *tensor, const std::vector<size_t> &shape, int node_type,
                                        ShapeFusionMatrix *constant_matrix);

 private:
  std::map<uint32_t, ShapeFusionMatrix> shape_fusion_matrices_;
  std::vector<lite::Tensor *> shape_fusion_outputs_;
#endif
  LiteModel *lite_model_ = nullptr;
  const std::vector<Model::Node *> *all_nodes_ = nullptr;
  std::vector<lite::Tensor *> *src_tensors_ = nullptr;
  std::map<uint32_t, std::vector<Model::Node *>> used_nodes_;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_RUNTIME_SHAPE_FUSION_PASS_H_
