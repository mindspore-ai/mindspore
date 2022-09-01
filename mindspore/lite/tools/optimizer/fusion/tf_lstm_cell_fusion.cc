/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/fusion/tf_lstm_cell_fusion.h"
#include <memory>
#include "ops/lstm.h"
#include "src/common/utils.h"
#include "tools/common/tensor_util.h"
#include "include/common/utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/fusion/tflite_lstm_cell_fusion.h"
#include "tools/optimizer/common/helper.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
namespace {
constexpr int kNumInPlaceHolder = 10;
constexpr int kNumGetItem = 4;
constexpr size_t kLstmInputsLength = 13;
constexpr size_t kLstmInputsVarNum = 11;
constexpr size_t kCondNodesNum = 12;
constexpr size_t kCondCNodesNum = 4;
constexpr size_t kBodyNodesNum = 82;
constexpr size_t kBodyCNodesNum = 30;
constexpr auto kUnidirectionalGateNum = 4;
constexpr auto kBidirectionalGateNum = 8;
const auto &p1 = std::placeholders::_1;
bool IsParameterNode(const BaseRef &n) { return utils::isa<ParameterPtr>(n); }

std::vector<VectorRef> GenerateBodyGraphHiddenPattern(const VarPtr &forget_bias_input,
                                                      const std::vector<CondVarPtr> &placeholders) {
  MS_CHECK_TRUE_RET(placeholders.size() >= kNumInPlaceHolder, {});
  auto is_var_getitem = std::make_shared<Var>("GetItem");
  MS_CHECK_TRUE_RET(is_var_getitem != nullptr, {});
  auto is_param3 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param3 != nullptr, {});
  VectorRef get_item = VectorRef({is_var_getitem, placeholders[7], placeholders[2], is_param3});
  auto is_var1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var1 != nullptr, {});
  VectorRef concat_input_h = VectorRef({is_var1, get_item, placeholders[5]});

  auto is_var2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var2 != nullptr, {});
  VectorRef matmul = VectorRef({is_var2, concat_input_h, placeholders[8]});
  auto is_var3 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var3 != nullptr, {});
  VectorRef bias = VectorRef({is_var3, matmul, placeholders[9]});
  auto is_var4 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var4 != nullptr, {});
  VectorRef split = VectorRef({is_var4, bias});

  std::vector<VectorRef> get_items;
  for (int i = 0; i < kNumGetItem; ++i) {
    auto is_var_loop1 = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(is_var_loop1 != nullptr, {});
    auto is_var_loop2 = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(is_var_loop2 != nullptr, {});
    VectorRef get_item_loop = VectorRef({is_var_loop1, split, is_var_loop2});
    get_items.push_back(get_item_loop);
  }

  auto is_var_sigmoid1 = std::make_shared<Var>("Sigmoid");
  MS_CHECK_TRUE_RET(is_var_sigmoid1 != nullptr, {});
  VectorRef input_gate = VectorRef({is_var_sigmoid1, get_items[0]});
  auto is_var_tanh1 = std::make_shared<Var>("Tanh");
  MS_CHECK_TRUE_RET(is_var_tanh1 != nullptr, {});
  VectorRef input_to_cell = VectorRef({is_var_tanh1, get_items[1]});
  auto is_var_add1 = std::make_shared<Var>("Add");
  MS_CHECK_TRUE_RET(is_var_add1 != nullptr, {});
  VectorRef forget_bias = VectorRef({is_var_add1, get_items[2], forget_bias_input});
  auto is_var_sigmoid2 = std::make_shared<Var>("Sigmoid");
  MS_CHECK_TRUE_RET(is_var_sigmoid2 != nullptr, {});
  VectorRef forget_gate = VectorRef({is_var_sigmoid2, forget_bias});
  auto is_var_sigmoid3 = std::make_shared<Var>("Sigmoid");
  MS_CHECK_TRUE_RET(is_var_sigmoid3 != nullptr, {});
  VectorRef output_gate = VectorRef({is_var_sigmoid3, get_items[3]});

  auto is_var5 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var5 != nullptr, {});
  VectorRef forgetted_cell = VectorRef({is_var5, forget_gate, placeholders[4]});
  auto is_var6 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var6 != nullptr, {});
  VectorRef inputted_cell = VectorRef({is_var6, input_gate, input_to_cell});
  auto is_var_add2 = std::make_shared<Var>("Add");
  MS_CHECK_TRUE_RET(is_var_add2 != nullptr, {});
  VectorRef input_forget_cell = VectorRef({is_var_add2, forgetted_cell, inputted_cell});
  auto is_var_tanh2 = std::make_shared<Var>("Tanh");
  MS_CHECK_TRUE_RET(is_var_tanh2 != nullptr, {});
  VectorRef to_new_hidden = VectorRef({is_var_tanh2, input_forget_cell});
  auto is_var_mul = std::make_shared<Var>("Mul");
  MS_CHECK_TRUE_RET(is_var_mul != nullptr, {});
  VectorRef new_hidden = VectorRef({is_var_mul, output_gate, to_new_hidden});
  return {input_forget_cell, new_hidden};
}
}  // namespace

TfLstmCellFusion::TfLstmCellFusion(const std::string &name, bool multigraph)
    : TfliteLstmCellFusion(name, multigraph, kLstmInputsLength, kLstmInputsVarNum, kCondNodesNum, kCondCNodesNum,
                           kBodyNodesNum, kBodyCNodesNum) {
  /*
   * vars for lstm cell input
   * 0:cond 1:body 2:index 3:limit1 4:output 5:cell 6:hidden 7:limit2 8:input 9:kernel 10:bias
   */
}

AnfNodePtr TfLstmCellFusion::GetBodyGraphPattern(const PrimitiveVarMapPtr &primitive_vars) const {
  std::vector<CondVarPtr> placeholders;
  for (int i = 0; i < kNumInPlaceHolder; ++i) {
    auto is_param_holder = std::make_shared<CondVar>(IsParameterNode);
    MS_CHECK_TRUE_RET(is_param_holder != nullptr, nullptr);
    placeholders.emplace_back(is_param_holder);
  }
  auto is_var1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var1 != nullptr, {});
  auto is_param1 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param1 != nullptr, nullptr);
  VectorRef add2 = VectorRef({is_var1, placeholders[2], is_param1});
  auto is_var2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var2 != nullptr, {});
  auto is_param2 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param2 != nullptr, nullptr);
  VectorRef add3 = VectorRef({is_var2, placeholders[0], is_param2});

  forget_bias_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(forget_bias_ != nullptr, nullptr);
  auto hidden_cells = GenerateBodyGraphHiddenPattern(forget_bias_, placeholders);
  MS_CHECK_TRUE_RET(hidden_cells.size() == kInputSizeTwo, {});

  auto is_var_mul1 = std::make_shared<Var>("Mul");
  MS_CHECK_TRUE_RET(is_var_mul1 != nullptr, {});
  auto input_forget_cell = hidden_cells[0];
  MS_CHECK_TRUE_RET(!input_forget_cell.empty(), {});
  VectorRef new_to_cell = VectorRef({is_var_mul1, cell_zoneout_new_, input_forget_cell});
  auto is_var_mul2 = std::make_shared<Var>("Mul");
  MS_CHECK_TRUE_RET(is_var_mul2 != nullptr, {});
  VectorRef old_to_cell = VectorRef({is_var_mul2, cell_zoneout_old_, placeholders[4]});
  auto is_var_add1 = std::make_shared<Var>("Add");
  MS_CHECK_TRUE_RET(is_var_add1 != nullptr, {});
  VectorRef output_cell = VectorRef({is_var_add1, new_to_cell, old_to_cell});

  auto new_hidden = hidden_cells[1];
  MS_CHECK_TRUE_RET(!new_hidden.empty(), {});
  auto is_var_mul3 = std::make_shared<Var>("Mul");
  MS_CHECK_TRUE_RET(is_var_mul3 != nullptr, {});
  VectorRef new_to_hidden = VectorRef({is_var_mul3, hidden_zoneout_new_, new_hidden});
  auto is_var_mul4 = std::make_shared<Var>("Mul");
  MS_CHECK_TRUE_RET(is_var_mul4 != nullptr, {});
  VectorRef old_to_hidden = VectorRef({is_var_mul4, hidden_zoneout_old_, placeholders[5]});
  auto is_var_add2 = std::make_shared<Var>("Add");
  MS_CHECK_TRUE_RET(is_var_add2 != nullptr, {});
  VectorRef output_hidden = VectorRef({is_var_add2, new_to_hidden, old_to_hidden});

  auto is_var3 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var3 != nullptr, {});
  VectorRef set_item = VectorRef({is_var3, placeholders[3], placeholders[2], new_hidden});

  auto is_make_tuple = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMakeTuple));
  MS_CHECK_TRUE_RET(is_make_tuple != nullptr, nullptr);
  std::vector<BaseRef> outputs = {is_make_tuple, add3, placeholders[1], add2, set_item, output_cell, output_hidden};
  outputs.insert(outputs.end(), placeholders.begin() + 6, placeholders.end());
  VectorRef make_tuple_node = VectorRef(outputs);
  auto is_return = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReturn));
  MS_CHECK_TRUE_RET(is_return != nullptr, nullptr);
  VectorRef return_node = VectorRef({is_return, make_tuple_node});

  VarPtr is_fg = std::make_shared<Var>("RootG");
  MS_CHECK_TRUE_RET(is_fg != nullptr, nullptr);
  auto pattern = Helper::SexpToNode(return_node, is_fg, primitive_vars.get(), true);
  return pattern;
}

STATUS TfLstmCellFusion::SetWeightAbstractAndDefault(const ParameterPtr &weight, const std::vector<int64_t> &shape,
                                                     const float *const data_ptr, const int hidden_size) {
  MS_ASSERT(weight != nullptr);
  MS_ASSERT(data_ptr != nullptr);
  if (shape.size() != kInputSizeThree) {
    MS_LOG(ERROR) << "lstm weight shape must have 3 dims";
    return RET_ERROR;
  }
  const auto param_num = shape[0] * shape[1] * shape[kInputIndexTwo];
  auto tensor_data = new (std::nothrow) float[static_cast<size_t>(param_num) * sizeof(float)];
  std::vector<int> data_diff{0, 3, 2, 1};
  if (tensor_data == nullptr) {
    MS_LOG(DEBUG) << "new data failed";
    return RET_ERROR;
  }
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < hidden_size; ++j) {
      for (int t = 0; t < shape[2]; ++t) {
        tensor_data[(i * hidden_size + j) * shape[2] + t] = data_ptr[t * shape[1] + data_diff[i] * hidden_size + j];
      }
    }
  }
  auto tensor_info =
    lite::CreateTensorInfo(tensor_data, static_cast<size_t>(param_num) * sizeof(float), shape, kNumberTypeFloat32);
  delete[] tensor_data;
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "create tensor info failed.";
    return RET_ERROR;
  }
  auto status = lite::InitParameterFromTensorInfo(weight, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS TfLstmCellFusion::SplitWeights(const AnfNodePtr &weight, const ParameterPtr &weight_i,
                                      const ParameterPtr &weight_c, int hidden_size) {
  // split input_size and hidden_size at dim 0
  // transform i,c,f,o to i,o,f,c at dim 1
  MS_ASSERT(weight != nullptr);
  MS_ASSERT(weight_i != nullptr);
  MS_ASSERT(weight_c != nullptr);
  if (!utils::isa<ParameterPtr>(weight)) {
    return RET_ERROR;
  }
  auto weight_param = utils::cast<ParameterPtr>(weight);
  if (!weight_param->has_default() || weight_param->default_param() == nullptr) {
    MS_LOG(DEBUG) << "weight not have default value";
    return RET_ERROR;
  }
  if (!utils::isa<tensor::TensorPtr>(weight_param->default_param())) {
    MS_LOG(DEBUG) << "default value is not tensor::Tensor";
    return RET_FAILED;
  }
  auto origin_tensor = std::dynamic_pointer_cast<tensor::Tensor>(weight_param->default_param());
  if (origin_tensor->data_type() != kNumberTypeFloat32 && origin_tensor->data_type() != kNumberTypeFloat) {
    MS_LOG(DEBUG) << "origin_tensor is not float32 type";
    return RET_ERROR;
  }
  auto data_ptr = reinterpret_cast<float *>(origin_tensor->data_c());
  auto data_shape = origin_tensor->shape();
  if (data_shape.size() != kInputSizeTwo) {
    MS_LOG(ERROR) << "weight data shape invalid";
    return RET_ERROR;
  }
  if (data_shape[0] <= hidden_size) {
    MS_LOG(ERROR) << "weight data shape[0] invalid";
    return RET_ERROR;
  }
  if (hidden_size * 4 != data_shape[1]) {
    MS_LOG(ERROR) << "weight data shape[1] invalid";
    return RET_ERROR;
  }
  const auto input_size = data_shape[0] - hidden_size;

  std::vector<int64_t> shape_i{1, kUnidirectionalGateNum * hidden_size, input_size};
  if (SetWeightAbstractAndDefault(weight_i, shape_i, data_ptr, hidden_size) != RET_OK) {
    MS_LOG(ERROR) << "get weight_i failed";
    return RET_ERROR;
  }

  std::vector<int64_t> shape_c{1, kUnidirectionalGateNum * hidden_size, hidden_size};
  if (SetWeightAbstractAndDefault(weight_c, shape_c, data_ptr + input_size * data_shape[1], hidden_size) != RET_OK) {
    MS_LOG(ERROR) << "get weight_i failed";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS TfLstmCellFusion::PopulateBiasNode(const EquivPtr &body_equiv, const ParameterPtr &new_bias,
                                          const AnfNodePtr &old_bias, const int hidden_size) const {
  MS_ASSERT(body_equiv != nullptr);
  MS_ASSERT(new_bias != nullptr);
  MS_ASSERT(old_bias != nullptr);
  if (!utils::isa<ParameterPtr>(old_bias)) {
    MS_LOG(DEBUG) << "old_bias is not parameter";
    return RET_ERROR;
  }
  auto old_bias_param = utils::cast<ParameterPtr>(old_bias);
  if (!old_bias_param->has_default() || old_bias_param->default_param() == nullptr) {
    MS_LOG(DEBUG) << "bias not have default value";
    return RET_ERROR;
  }
  if (!utils::isa<tensor::TensorPtr>(old_bias_param->default_param())) {
    MS_LOG(DEBUG) << "default value is not tensor::Tensor";
    return RET_FAILED;
  }
  auto origin_tensor = std::dynamic_pointer_cast<tensor::Tensor>(old_bias_param->default_param());
  MS_CHECK_TRUE_RET(origin_tensor != nullptr, RET_ERROR);
  if (origin_tensor->data_type() != kNumberTypeFloat32 && origin_tensor->data_type() != kNumberTypeFloat) {
    MS_LOG(DEBUG) << "origin_tensor is not float32 type";
    return RET_ERROR;
  }
  auto data_ptr = reinterpret_cast<float *>(origin_tensor->data_c());
  MS_CHECK_TRUE_RET(data_ptr != nullptr, RET_ERROR);
  auto data_shape = origin_tensor->shape();
  MS_CHECK_GE(hidden_size, 0, RET_ERROR);
  if (data_shape.size() != 1 || data_shape[0] != 4 * hidden_size) {
    MS_LOG(DEBUG) << "bias data shape illegal";
    return RET_ERROR;
  }

  std::vector<int64_t> shape{1, kBidirectionalGateNum * hidden_size};
  auto tensor_data = std::make_unique<float[]>(static_cast<size_t>(hidden_size) * 8);
  MS_CHECK_TRUE_RET(tensor_data != nullptr, lite::RET_ERROR);
  auto forget_bias_node = utils::cast<AnfNodePtr>((*body_equiv)[forget_bias_]);
  if (forget_bias_node == nullptr) {
    MS_LOG(ERROR) << "forget bias node is nullptr";
    return RET_ERROR;
  }
  float forget_bias_value = 0.0f;
  if (GetFloatScalarFromTensorInfo(forget_bias_node, &forget_bias_value) != RET_OK) {
    return RET_ERROR;
  }

  std::vector<int> data_diff{0, 3, 2, 1};
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < hidden_size; ++j) {
      if (i < 4) {
        tensor_data[i * hidden_size + j] = data_ptr[data_diff[i] * hidden_size + j];
        if (i == 2) {  // forget bias
          tensor_data[i * hidden_size + j] += forget_bias_value;
        }
      } else {
        tensor_data[i * hidden_size + j] = 0.0f;
      }
    }
  }

  auto tensor_info =
    lite::CreateTensorInfo(tensor_data.get(), static_cast<size_t>(hidden_size) * kBidirectionalGateNum * sizeof(float),
                           shape, kNumberTypeFloat32);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "create tensor info failed.";
    return RET_ERROR;
  }

  auto status = lite::InitParameterFromTensorInfo(new_bias, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return RET_ERROR;
  }

  return RET_OK;
}

CNodePtr TfLstmCellFusion::CreateLSTMNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                          const EquivPtr &body_equiv, const std::string &base_name,
                                          const float zoneout_cell, const float zoneout_hidden) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(equiv != nullptr);
  auto lstm_prim = std::make_shared<ops::LSTM>();
  MS_CHECK_TRUE_RET(lstm_prim != nullptr, nullptr);
  auto lstm_prim_c = lstm_prim->GetPrim();
  MS_CHECK_TRUE_RET(lstm_prim_c != nullptr, nullptr);
  lstm_prim->set_bidirectional(false);
  lstm_prim->set_zoneout_cell(zoneout_cell);
  lstm_prim->set_zoneout_hidden(zoneout_hidden);
  auto value_node = NewValueNode(lstm_prim_c);
  MS_CHECK_TRUE_RET(value_node != nullptr, nullptr);

  auto &vars = while_input_vars_;
  auto weight = utils::cast<AnfNodePtr>((*equiv)[vars[9]]);
  MS_ASSERT(weight);
  auto bias = utils::cast<AnfNodePtr>((*equiv)[vars[10]]);
  MS_ASSERT(bias);
  auto input = utils::cast<AnfNodePtr>((*equiv)[vars[8]]);
  MS_ASSERT(input);
  auto cell = utils::cast<AnfNodePtr>((*equiv)[vars[5]]);
  MS_ASSERT(cell);
  auto hidden = utils::cast<AnfNodePtr>((*equiv)[vars[6]]);
  MS_ASSERT(hidden);

  if (!utils::isa<ParameterPtr>(hidden)) {
    MS_LOG(DEBUG) << "hidden is not parameter";
    return nullptr;
  }
  auto hidden_param = utils::cast<ParameterPtr>(hidden);
  if (!utils::isa<abstract::AbstractTensorPtr>(hidden_param->abstract())) {
    MS_LOG(DEBUG) << "hidden abstract is not AbstractTensor";
    return nullptr;
  }
  auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(hidden_param->abstract());
  MS_CHECK_TRUE_MSG(abstract_tensor != nullptr, nullptr, "Cast to abstract tensor failed!");
  auto hidden_shape = utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape();
  if (hidden_shape.empty()) {
    MS_LOG(DEBUG) << "can't get hidden shape";
    return nullptr;
  }

  auto i_weight = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(i_weight != nullptr, nullptr);
  i_weight->set_name(base_name + "_weight_i");
  if (weight->abstract() != nullptr) {
    i_weight->set_abstract(weight->abstract()->Clone());
  }

  auto c_weight = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(c_weight != nullptr, nullptr);
  c_weight->set_name(base_name + "_weight_c");
  if (weight->abstract() != nullptr) {
    c_weight->set_abstract(weight->abstract()->Clone());
  }

  if (SplitWeights(weight, i_weight, c_weight, static_cast<int>(hidden_shape.back())) != RET_OK) {
    MS_LOG(DEBUG) << "split weight to i_weight and c_weight failed";
    return nullptr;
  }

  auto bias_node = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(bias_node != nullptr, nullptr);
  MS_CHECK_TRUE_RET(bias_node != nullptr, nullptr);
  bias_node->set_name(base_name + "_bias");
  if (bias->abstract() != nullptr) {
    bias_node->set_abstract(bias->abstract()->Clone());
  }

  if (PopulateBiasNode(body_equiv, bias_node, bias, static_cast<int>(hidden_shape.back())) != RET_OK) {
    MS_LOG(DEBUG) << "reorder bias failed";
    return nullptr;
  }

  if (!utils::isa<CNodePtr>(input) || !CheckPrimitiveType(input, prim::kPrimTensorListFromTensor)) {
    MS_LOG(DEBUG) << "input is not tensorlistfromtensor op";
    return nullptr;
  }
  auto tensor_list_cnode = utils::cast<CNodePtr>(input);
  auto input_tensor_node = tensor_list_cnode->input(1);

  std::vector<AnfNodePtr> new_node_inputs = {value_node, input_tensor_node, i_weight, c_weight, bias_node, hidden,
                                             cell};
  auto new_node = func_graph->NewCNode(new_node_inputs);
  MS_CHECK_TRUE_RET(new_node != nullptr, nullptr);
  new_node->set_fullname_with_scope(base_name);
  return new_node;
}
}  // namespace opt
}  // namespace mindspore
