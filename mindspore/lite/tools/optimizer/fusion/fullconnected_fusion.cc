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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/fullconnected_fusion.h"
#include <memory>
#include <vector>
#include "tools/common/tensor_util.h"
#include "ops/fusion/full_connection.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore::opt {
namespace {
constexpr size_t kFcWeightIndex = 2;
constexpr size_t kFcParameterDims = 2;

template <typename T>
void Segmm(bool transb, T *A, T *B, T *bias, T *C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      T value = 0;
      for (int z = 0; z < K; z++) {
        if (transb) {
          value += A[i * K + z] * B[j * K + z];
        } else {
          value += A[i * K + z] * B[z * N + j];
        }
      }
      if (bias != nullptr) {
        value += bias[i * N + j];
      }
      C[i * N + j] = value;
    }
  }
}

int CalNewCnodeScale(const AnfNodePtr &curr_weight_node, const AnfNodePtr &prev_weight_node) {
  std::shared_ptr<tensor::Tensor> curr_weight_tensor = GetTensorInfo(curr_weight_node);
  MS_CHECK_TRUE_RET(curr_weight_tensor != nullptr, RET_ERROR);
  if (curr_weight_tensor->data_type() != kNumberTypeFloat32) {
    MS_LOG(INFO) << "only support float32 data type";
    return RET_ERROR;
  }
  std::vector<int64_t> curr_tensor_shape = curr_weight_tensor->shape();
  auto curr_weight_data = reinterpret_cast<float *>(curr_weight_tensor->data_c());
  MS_CHECK_TRUE_RET(curr_weight_data != nullptr, RET_ERROR);

  std::shared_ptr<tensor::Tensor> prev_weight_tensor = GetTensorInfo(prev_weight_node);
  MS_CHECK_TRUE_RET(prev_weight_tensor != nullptr, RET_ERROR);
  if (prev_weight_tensor->data_type() != kNumberTypeFloat32) {
    MS_LOG(INFO) << "only support float32 data type";
    return RET_ERROR;
  }
  std::vector<int64_t> prev_tensor_shape = prev_weight_tensor->shape();
  auto prev_weight_data = reinterpret_cast<float *>(prev_weight_tensor->data_c());
  MS_CHECK_TRUE_RET(prev_weight_data != nullptr, RET_ERROR);

  if (curr_tensor_shape.size() != kFcParameterDims || prev_tensor_shape.size() != kFcParameterDims ||
      curr_tensor_shape[1] != prev_tensor_shape[0]) {
    MS_LOG(ERROR) << "previous fullconnection node shape don't match with current node";
    return RET_ERROR;
  }

  std::vector<int64_t> new_shape{curr_tensor_shape[0], prev_tensor_shape[1]};
  float *new_weight_data = new (std::nothrow) float[curr_tensor_shape[0] * prev_tensor_shape[1]];
  if (new_weight_data == nullptr) {
    MS_LOG(ERROR) << "alloc failed";
    return RET_ERROR;
  }
  auto status = memset_s(new_weight_data, curr_tensor_shape[0] * prev_tensor_shape[1] * sizeof(float), 0,
                         curr_tensor_shape[0] * prev_tensor_shape[1] * sizeof(float));
  if (status != EOK) {
    MS_LOG(ERROR) << "memset_s failed";
    delete[] new_weight_data;
    return RET_ERROR;
  }
  Segmm<float>(false, curr_weight_data, prev_weight_data, nullptr, new_weight_data, curr_tensor_shape[0],
               prev_tensor_shape[1], curr_tensor_shape[1]);
  auto parameter_node = curr_weight_node->cast<ParameterPtr>();
  auto tensor_info =
    lite::CreateTensorInfo(new_weight_data, curr_tensor_shape[0] * prev_tensor_shape[1] * sizeof(float), new_shape,
                           curr_weight_tensor->data_type());
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "create tensor info failed";
    delete[] new_weight_data;
    return RET_ERROR;
  }
  delete[] new_weight_data;
  auto ret = lite::InitParameterFromTensorInfo(parameter_node, tensor_info);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "failed to initialize parameter from tensor info";
    return ret;
  }

  parameter_node->set_name(curr_weight_node->fullname_with_scope() + "_fusion");
  return RET_OK;
}

int CalNewCnodeBias(const FuncGraphPtr &func_graph, const CNodePtr &curr_cnode, const AnfNodePtr &prev_bias_node) {
  std::shared_ptr<tensor::Tensor> prev_bias_tensor = GetTensorInfo(prev_bias_node);
  MS_CHECK_TRUE_RET(prev_bias_tensor != nullptr, RET_ERROR);
  if (prev_bias_tensor->data_type() != kNumberTypeFloat32) {
    MS_LOG(INFO) << "only support float32 data type";
    return RET_ERROR;
  }
  std::vector<int64_t> prev_bias_shape = prev_bias_tensor->shape();
  MS_CHECK_TRUE_RET(prev_bias_shape.size() > 0, RET_ERROR);
  auto prev_bias_data = reinterpret_cast<float *>(prev_bias_tensor->data_c());
  MS_CHECK_TRUE_RET(prev_bias_data != nullptr, RET_ERROR);

  AnfNodePtr curr_weight_node = curr_cnode->input(kInputIndexTwo);
  std::shared_ptr<tensor::Tensor> curr_weight_tensor = GetTensorInfo(curr_weight_node);
  MS_CHECK_TRUE_RET(curr_weight_tensor != nullptr, RET_ERROR);
  if (curr_weight_tensor->data_type() != kNumberTypeFloat32) {
    MS_LOG(INFO) << "only support float32 data type";
    return RET_ERROR;
  }
  std::vector<int64_t> curr_weight_shape = curr_weight_tensor->shape();
  auto curr_weight_data = reinterpret_cast<float *>(curr_weight_tensor->data_c());
  MS_CHECK_TRUE_RET(curr_weight_data != nullptr, RET_ERROR);

  AnfNodePtr curr_bias_node = nullptr;
  if (curr_cnode->size() > kInputIndexThree) {
    curr_bias_node = curr_cnode->input(kInputIndexThree);
  }
  float *curr_bias_data = nullptr;
  if (curr_bias_node != nullptr) {
    std::shared_ptr<tensor::Tensor> curr_bias_tensor = GetTensorInfo(curr_bias_node);
    MS_CHECK_TRUE_RET(curr_bias_tensor != nullptr, RET_ERROR);
    if (curr_bias_tensor->data_type() != kNumberTypeFloat32) {
      MS_LOG(INFO) << "only support float32 data type";
      return RET_ERROR;
    }
    std::vector<int64_t> curr_bias_shape = curr_bias_tensor->shape();
    curr_bias_data = reinterpret_cast<float *>(curr_bias_tensor->data_c());
    MS_CHECK_TRUE_RET(curr_bias_data != nullptr, RET_ERROR);
    if (curr_bias_shape[0] != curr_weight_shape[0]) {
      MS_LOG(ERROR) << "weight with bias shape of fullconnected node don't match.";
      return RET_ERROR;
    }
  } else {
    auto new_bias_node = func_graph->add_parameter();
    std::vector<int64_t> shape_vector(curr_weight_shape[0]);
    auto tensor_info = lite::CreateTensorInfo(nullptr, 0, shape_vector, prev_bias_tensor->data_type());
    MS_CHECK_TRUE_RET(tensor_info != nullptr, RET_ERROR);
    curr_bias_data = reinterpret_cast<float *>(tensor_info->data_c());
    MS_CHECK_TRUE_RET(
      memset_s(curr_bias_data, curr_weight_shape[0] * sizeof(float), 0, curr_weight_shape[0] * sizeof(float)) == EOK,
      RET_ERROR);
    auto status = lite::InitParameterFromTensorInfo(new_bias_node, tensor_info);
    MS_CHECK_TRUE_RET(status != RET_OK, RET_ERROR);
    auto manager = func_graph->manager();
    manager->AddEdge(curr_cnode, new_bias_node);
  }

  // bias = curr_weight * prev_bias + curr_bias
  int row = prev_bias_shape.size() == 1 ? 1 : prev_bias_shape[0];
  Segmm<float>(true, prev_bias_data, curr_weight_data, curr_bias_data, curr_bias_data, row, curr_weight_shape[0],
               curr_weight_shape[1]);
  auto parameter_node = curr_bias_node->cast<ParameterPtr>();
  parameter_node->set_name(curr_bias_node->fullname_with_scope() + "_fusion");
  return RET_OK;
}

bool IsPrimitiveProper(const CNodePtr &curr_fc_cnode, const CNodePtr &prev_fc_cnode) {
  MS_CHECK_TRUE_RET(curr_fc_cnode != nullptr, false);
  MS_CHECK_TRUE_RET(prev_fc_cnode != nullptr, false);
  auto prev_primc = GetValueNode<PrimitiveCPtr>(prev_fc_cnode->input(0));  // previous fc primitive
  MS_CHECK_TRUE_RET(prev_primc != nullptr, false);
  if (IsQuantParameterNode(prev_primc)) {
    MS_LOG(INFO) << prev_fc_cnode->fullname_with_scope() << "is quant node";
    return false;
  }

  if (CheckPrimitiveType(prev_fc_cnode, prim::kPrimFullConnection)) {
    auto pre_fc_weight_node = prev_fc_cnode->input(kFcWeightIndex);
    if (!IsParamNode(pre_fc_weight_node)) {
      MS_LOG(INFO) << pre_fc_weight_node->fullname_with_scope() << "'s weight is not parameter";
      return false;
    }
    auto full_prim = api::MakeShared<mindspore::ops::FullConnection>(prev_primc);
    MS_ASSERT(full_prim != nullptr);
    auto full_prim_c = full_prim->GetPrim();
    MS_ASSERT(full_prim_c != nullptr);
    if (full_prim_c->GetAttr(ops::kActivationType) != nullptr) {
      auto activate_type = full_prim->get_activation_type();
      if (activate_type != NO_ACTIVATION) {
        MS_LOG(INFO) << pre_fc_weight_node->fullname_with_scope() << " has activation operator";
        return false;
      }
    }
  }

  auto curr_primc = GetValueNode<PrimitiveCPtr>(curr_fc_cnode->input(0));  // previous fc primitive
  MS_CHECK_TRUE_RET(curr_primc != nullptr, false);
  if (IsQuantParameterNode(curr_primc)) {
    MS_LOG(INFO) << curr_fc_cnode->fullname_with_scope() << "is quant node";
    return false;
  }
  if (CheckPrimitiveType(curr_fc_cnode, prim::kPrimFullConnection)) {
    auto fc_weight_node = curr_fc_cnode->input(kFcWeightIndex);
    if (!IsParamNode(fc_weight_node)) {
      MS_LOG(INFO) << curr_fc_cnode->fullname_with_scope() << "'s weight is not parameter";
      return false;
    }
  }
  return true;
}
}  // namespace

const BaseRef FullConnectedFusion::DefinePattern() const {
  auto is_fc1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimFullConnection>);
  MS_CHECK_TRUE_RET(is_fc1 != nullptr, {});
  auto is_fc2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimFullConnection>);
  MS_CHECK_TRUE_RET(is_fc2 != nullptr, {});
  auto is_seq_var = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var != nullptr, {});
  return VectorRef({is_fc1, is_fc2, is_seq_var});
}

const AnfNodePtr FullConnectedFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                              const EquivPtr &) const {
  if (func_graph == nullptr || node == nullptr) {
    return nullptr;
  }

  auto curr_cnode = node->cast<CNodePtr>();
  if (curr_cnode == nullptr || curr_cnode->size() < kInputSizeThree) {
    return nullptr;
  }
  if (IsMarkedTrainOp(curr_cnode)) {
    return nullptr;
  }

  auto prev_fc_node = curr_cnode->input(1);
  auto prev_fc_cnode = prev_fc_node->cast<CNodePtr>();
  if (prev_fc_cnode == nullptr || IsMultiOutputTensors(func_graph, prev_fc_cnode) ||
      !IsPrimitiveProper(curr_cnode, prev_fc_cnode)) {
    return nullptr;
  }
  if (IsMarkedTrainOp(prev_fc_cnode)) {
    return nullptr;
  }

  auto curr_weight_node = curr_cnode->input(kInputIndexTwo);
  auto prev_weight_node = prev_fc_cnode->input(kInputIndexTwo);

  AnfNodePtr prev_bias_node = nullptr;
  if (prev_fc_cnode->size() > kInputIndexThree) {
    prev_bias_node = prev_fc_cnode->input(kInputIndexThree);
  }

  if (prev_bias_node != nullptr && CalNewCnodeBias(func_graph, curr_cnode, prev_bias_node) != RET_OK) {
    MS_LOG(ERROR) << "failed to fusion bias";
    return nullptr;
  }

  if (CalNewCnodeScale(curr_weight_node, prev_weight_node) != RET_OK) {
    MS_LOG(ERROR) << "failed to fusion weight";
    return nullptr;
  }

  // delete prev_fc_cnode
  auto manager = func_graph->manager();
  (void)manager->Replace(prev_fc_cnode, prev_fc_cnode->input(1));
  MS_LOG(INFO) << curr_cnode->fullname_with_scope() << " fusion success";
  return nullptr;
}
}  // namespace mindspore::opt
