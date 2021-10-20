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

#include "tools/converter/parser/lstm_adjust_pass.h"
#include "ops/lstm.h"
#include "ops/reshape.h"
#include "ops/transpose.h"
#include "src/common/utils.h"
#include "tools/anf_exporter/fetch_content.h"
#include "tools/common/tensor_util.h"
#include "utils/check_convert_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kMindLstmInputs = 5;
constexpr size_t kLSTMWeightIndex = 4;
constexpr size_t kGateNums = 4;  // it,ft,gt,ot
constexpr size_t kBiasNums = 2;  // b_ih,b_hh
constexpr int kWeightInputIndex = 2;
constexpr int kBirectionalNums = 2;
constexpr int kHiddenIndex = 2;
constexpr int kCellIndex = 3;
constexpr int kOutputBatchIndex = 2;
constexpr int kOutputHiddenIndex = 3;
AnfNodePtr GetRealLstmWeightNode(const FuncGraphPtr &graph, const CNodePtr &cnode, size_t weight_index) {
  MS_CHECK_TRUE_MSG(graph != nullptr, nullptr, "graph is nullptr.");
  MS_CHECK_TRUE_MSG(cnode != nullptr, nullptr, "cnode is nullptr.");
  if (!opt::CheckPrimitiveType(cnode, prim::kPrimLstm)) {
    return nullptr;
  }
  auto weight_node = cnode->input(weight_index);
  MS_CHECK_TRUE_MSG(weight_node != nullptr, nullptr, "weight_node is nullptr.");
  bool is_real_weight =
    !opt::CheckPrimitiveType(weight_node, opt::kPrimIdentity) && !opt::CheckPrimitiveType(weight_node, prim::kPrimLoad);
  while (!is_real_weight) {
    if (!utils::isa<CNode>(weight_node)) {
      MS_LOG(ERROR) << "weight node is invalid.";
      return nullptr;
    }
    auto weight_cnode = weight_node->cast<CNodePtr>();
    MS_ASSERT(weight_cnode != nullptr);
    weight_node = weight_cnode->input(1);
    MS_CHECK_TRUE_MSG(weight_node != nullptr, nullptr, "weight_node is nullptr.");
    is_real_weight = !opt::CheckPrimitiveType(weight_node, opt::kPrimIdentity) &&
                     !opt::CheckPrimitiveType(weight_node, prim::kPrimLoad);
  }
  auto manager = Manage(graph);
  MS_ASSERT(manager != nullptr);
  (void)manager->Replace(cnode->input(weight_index), weight_node);
  return weight_node;
}

// split flatten weight to ih,hh weight
int InitLstmWeight(const ParameterPtr &parameter, void *data, size_t data_size, const std::vector<int64_t> &shape,
                   TypeId data_type, bool is_bias = false, size_t num_directions = 1) {
  MS_CHECK_TRUE_MSG(parameter != nullptr, RET_ERROR, "parameter is nullptr.");
  MS_CHECK_TRUE_MSG(data != nullptr, RET_ERROR, "data is nullptr.");
  auto tensor_info = lite::CreateTensorInfo(nullptr, 0, shape, data_type);
  MS_CHECK_TRUE_MSG(tensor_info != nullptr, RET_ERROR, "Create tensor info failed.");
  // lite input weight order should wii,wio,wif,wig
  size_t combined_num = is_bias ? static_cast<size_t>(kBiasNums) : 1;  // ih_bias and hh_bias are combined
  const auto &weight_order = kNH2NC;
  size_t weight_batch = num_directions * combined_num;
  size_t weight_size = data_size / (kGateNums * weight_batch);
  for (size_t k = 0; k < num_directions; k++) {
    auto start_addr_x = static_cast<char *>(data) + data_size / num_directions * k;
    auto start_addr_y = static_cast<char *>(tensor_info->data_c()) + data_size / num_directions * k;
    for (size_t i = 0; i < combined_num; i++) {
      start_addr_x = start_addr_x + data_size / weight_batch * i;
      start_addr_y = start_addr_y + data_size / weight_batch * i;
      for (size_t j = 0; j < kGateNums; j++) {
        if (EOK != memcpy_s(start_addr_y + j * weight_size, weight_size,
                            start_addr_x + static_cast<size_t>(weight_order[j]) * weight_size, weight_size)) {
          MS_LOG(ERROR) << "memcpy_s data failed";
          return RET_ERROR;
        }
      }
    }
  }
  auto status = lite::InitParameterFromTensorInfo(parameter, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvertBiWeight(char *flatten_weight, size_t hh_weight_size, size_t ih_weight_size) {
  MS_CHECK_TRUE_MSG(flatten_weight != nullptr, RET_ERROR, "flatten_weight is nullptr.");
  // convert weight
  if (hh_weight_size == 0) {
    return RET_ERROR;
  }
  auto hh_temp_ptr = reinterpret_cast<char *>(malloc(hh_weight_size));
  if (hh_temp_ptr == nullptr) {
    MS_LOG(ERROR) << "tensor_data is nullptr";
    return RET_ERROR;
  }
  if (memcpy_s(hh_temp_ptr, hh_weight_size, flatten_weight + ih_weight_size, hh_weight_size) != EOK) {
    free(hh_temp_ptr);
    MS_LOG(ERROR) << "memcpy error: hh_fd_weight to temp";
    return RET_ERROR;
  }
  if (memcpy_s(flatten_weight + ih_weight_size, ih_weight_size, flatten_weight + ih_weight_size + hh_weight_size,
               ih_weight_size) != EOK) {
    free(hh_temp_ptr);
    MS_LOG(ERROR) << "memcpy error: ih_bd_weight to hh_fd_weight";
    return RET_ERROR;
  }
  if (memcpy_s(flatten_weight + ih_weight_size + ih_weight_size, hh_weight_size, hh_temp_ptr, hh_weight_size) != EOK) {
    free(hh_temp_ptr);
    MS_LOG(ERROR) << "memcpy error: ih_bd_weight to hh_fd_weight";
    return RET_ERROR;
  }
  free(hh_temp_ptr);
  return RET_OK;
}
int ReplaceLstmNode(const FuncGraphManagerPtr &manager, const FuncGraphPtr &func_graph, const AnfNodePtr &lstm_node) {
  MS_CHECK_TRUE_MSG(manager != nullptr, RET_ERROR, "manager is nullptr.");
  MS_CHECK_TRUE_MSG(func_graph != nullptr, RET_ERROR, "func_graph is nullptr.");
  MS_CHECK_TRUE_MSG(lstm_node != nullptr, RET_ERROR, "lstm_node is nullptr.");
  auto lstm_cnode = lstm_node->cast<CNodePtr>();
  if (lstm_cnode == nullptr) {
    return RET_ERROR;
  }
  MS_CHECK_TRUE_MSG(lstm_cnode->input(0) != nullptr, RET_ERROR, "lstm_cnode->input(0) is nullptr.");
  auto primitive_c = GetValueNode<std::shared_ptr<mindspore::ops::LSTM>>(lstm_cnode->input(0));
  if (primitive_c == nullptr) {
    return RET_ERROR;
  }

  auto inputs = lstm_cnode->inputs();
  if (inputs.size() != kMindLstmInputs) {
    return RET_ERROR;
  }
  size_t input_size = static_cast<size_t>(primitive_c->get_input_size());
  size_t hidden_size = static_cast<size_t>(primitive_c->get_hidden_size());
  size_t num_directions = primitive_c->get_bidirectional() ? kBirectionalNums : 1;
  size_t byte_num = sizeof(float);
  size_t ih_weight_size = kGateNums * hidden_size * input_size * byte_num;
  size_t hh_weight_size = kGateNums * hidden_size * hidden_size * byte_num;
  size_t bias_size = kGateNums * hidden_size * kBiasNums * byte_num;
  std::vector<int64_t> ih_weight_shape = {static_cast<int64_t>(num_directions),
                                          static_cast<int64_t>(kGateNums * hidden_size),
                                          static_cast<int64_t>(input_size)};
  std::vector<int64_t> hh_weight_shape = {static_cast<int64_t>(num_directions),
                                          static_cast<int64_t>(kGateNums * hidden_size),
                                          static_cast<int64_t>(hidden_size)};
  // bias include ih bias and hh bias
  std::vector<int64_t> bias_shape = {static_cast<int64_t>(num_directions),
                                     static_cast<int64_t>(kGateNums * kBiasNums * hidden_size)};

  auto lstm_weight_node = GetRealLstmWeightNode(func_graph, lstm_cnode, kLSTMWeightIndex);
  MS_CHECK_TRUE_MSG(lstm_weight_node != nullptr, RET_ERROR, "lstm_weight_node is nullptr.");
  lite::DataInfo data_info;
  if (lstm_weight_node->isa<Parameter>()) {
    auto ret = FetchDataFromParameterNode(lstm_cnode, kLSTMWeightIndex, converter::kFmkTypeMs, false, &data_info);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "parse const node failed.";
      return RET_ERROR;
    }
  } else if (lstm_weight_node->isa<ValueNode>()) {
    auto ret = FetchDataFromValueNode(lstm_cnode, kLSTMWeightIndex, converter::kFmkTypeMs, false, &data_info);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "parse const node failed.";
      return RET_ERROR;
    }
  } else {
    return RET_ERROR;
  }
  if (data_info.data_type_ != kNumberTypeFloat32) {
    MS_LOG(DEBUG) << "default param is not fp32";
    return RET_ERROR;
  }
  auto data_type = static_cast<TypeId>(data_info.data_type_);
  auto *flatten_weight = reinterpret_cast<char *>(data_info.data_.data());
  // bidirection flatten weight arrange is ih_fd weight,hh_fd_weight,ih_bd_weight,hh_bd_weight,fd_bias,bd_bias
  // need first convert to ih_fd weight,ih_bd weight,hh_fd_weight,hh_bd_weight,fd_bias,bd_bias
  auto ret = RET_OK;
  if (num_directions == kBirectionalNums) {
    ret = ConvertBiWeight(flatten_weight, hh_weight_size, ih_weight_size);
    if (ret != RET_OK) {
      MS_LOG(DEBUG) << "convert birectional weight failed";
      return ret;
    }
  }

  auto ih_weight_paramter = func_graph->add_parameter();
  MS_CHECK_TRUE_MSG(ih_weight_paramter != nullptr, lite::RET_NULL_PTR, "ih_weight_paramter is nullptr.");
  ih_weight_paramter->fullname_with_scope() = lstm_weight_node->fullname_with_scope() + "_ih";
  ret = InitLstmWeight(ih_weight_paramter, flatten_weight, ih_weight_size * num_directions, ih_weight_shape, data_type,
                       false, num_directions);
  if (ret != RET_OK) {
    return ret;
  }

  auto hh_weight_paramter = func_graph->add_parameter();
  MS_CHECK_TRUE_MSG(hh_weight_paramter != nullptr, lite::RET_NULL_PTR, "hh_weight_paramter is nullptr.");
  hh_weight_paramter->fullname_with_scope() = lstm_weight_node->fullname_with_scope() + "_hh";
  ret = InitLstmWeight(hh_weight_paramter, flatten_weight + ih_weight_size * num_directions,
                       hh_weight_size * num_directions, hh_weight_shape, data_type, false, num_directions);
  if (ret != RET_OK) {
    return ret;
  }
  auto bias_paramter = func_graph->add_parameter();
  MS_CHECK_TRUE_MSG(bias_paramter != nullptr, lite::RET_NULL_PTR, "bias_paramter is nullptr.");
  bias_paramter->fullname_with_scope() = lstm_weight_node->fullname_with_scope() + "_bias";
  ret = InitLstmWeight(bias_paramter, flatten_weight + (hh_weight_size + ih_weight_size) * num_directions,
                       bias_size * num_directions, bias_shape, data_type, true, num_directions);
  if (ret != RET_OK) {
    return ret;
  }
  auto lstm_name = lstm_cnode->fullname_with_scope();
  std::vector<AnfNodePtr> new_lstm_inputs = {inputs.at(0),         inputs.at(1),  ih_weight_paramter,
                                             hh_weight_paramter,   bias_paramter, inputs.at(kHiddenIndex),
                                             inputs.at(kCellIndex)};
  auto new_lstm_cnode = func_graph->NewCNode(new_lstm_inputs);
  MS_CHECK_TRUE_MSG(new_lstm_cnode != nullptr, lite::RET_NULL_PTR, "New lstm cnode failed.");
  new_lstm_cnode->set_fullname_with_scope(lstm_name);
  new_lstm_cnode->set_abstract(lstm_cnode->abstract()->Clone());
  (void)manager->Replace(lstm_cnode, new_lstm_cnode);
  return RET_OK;
}
}  // namespace

bool LstmAdjustPass::Run(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, false, "func_graph is nullptr.");
  auto node_list = TopoSort(func_graph->get_return());
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return false;
  }
  // lstm weight is combined of ih_weight,hh_weight,bias
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (CheckPrimitiveType(node, prim::kPrimLstm)) {
      auto ret = ReplaceLstmNode(manager, func_graph, node);
      if (ret != RET_OK) {
        return false;
      }
      continue;
    }
    if (!CheckPrimitiveType(node, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr || cnode->inputs().size() != kTupleGetItemInputSize || cnode->input(1) == nullptr) {
      return false;
    }
    if (!CheckPrimitiveType(cnode->input(1)->cast<CNodePtr>(), prim::kPrimLstm)) {
      continue;
    }
    auto item_index_node = cnode->input(kWeightInputIndex)->cast<ValueNodePtr>();
    if (item_index_node == nullptr) {
      return false;
    }
    int index_value = CastToInt(item_index_node->value()).front();
    if (index_value != 0) {
      continue;
    }
    auto get_item_name = cnode->fullname_with_scope();
    auto transpose_prim = std::make_shared<ops::Transpose>();
    MS_CHECK_TRUE_MSG(transpose_prim != nullptr, false, "transpose_prim is nullptr.");
    std::vector<int> perm_value = {0, kOutputBatchIndex, 1, kOutputHiddenIndex};
    auto transpose_perm = BuildIntVecParameterNode(func_graph, perm_value, "transpose_" + get_item_name + "_perm");
    auto new_transpose_node = func_graph->NewCNode(transpose_prim, {cnode, transpose_perm});
    MS_CHECK_TRUE_MSG(new_transpose_node != nullptr, false, "New transpose node failed.");

    auto reshape_prim = std::make_shared<ops::Reshape>();
    MS_CHECK_TRUE_MSG(reshape_prim != nullptr, false, "reshape_prim is nullptr.");
    auto reshape_perm = BuildIntVecParameterNode(func_graph, {0, 0, -1}, "reshape_" + get_item_name + "_perm");
    auto new_reshape_node = func_graph->NewCNode(reshape_prim, {new_transpose_node, reshape_perm});
    MS_CHECK_TRUE_MSG(new_reshape_node != nullptr, false, "New reshape node failed.");
    (void)manager->Replace(cnode, new_reshape_node);
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
