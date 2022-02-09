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

#ifndef RUNTIME_PASS_CLIP
#include <map>
#include <vector>
#include "src/lite_model.h"
#include "src/common/tensor_util.h"
#include "schema/ops_generated.h"
#include "schema/model_generated.h"

namespace mindspore::lite {
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

  int ConvertToShapeFusion(Model::Node *node);
  int FusePostNodes(Model::Node *node, size_t subgraph_index);

 private:
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
  LiteModel *lite_model_ = nullptr;
  const std::vector<Model::Node *> *all_nodes_ = nullptr;
  std::vector<lite::Tensor *> *src_tensors_ = nullptr;
  std::map<uint32_t, std::vector<Model::Node *>> used_nodes_;
  std::map<uint32_t, ShapeFusionMatrix> shape_fusion_matrices_;
};
}  // namespace mindspore::lite
#endif
#endif  // MINDSPORE_LITE_SRC_RUNTIME_RUNTIME_SHAPE_FUSION_PASS_H_
