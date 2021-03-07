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
#include "tools/optimizer/fusion/tf_lstm_cell_fusion.h"
#include <memory>
#include "ops/lstm.h"
#include "src/common/utils.h"
#include "src/param_value_lite.h"
#include "utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "securec/include/securec.h"
#include "tools/optimizer/fusion/tflite_lstm_cell_fusion.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kLstmInputsLength = 13;
constexpr size_t kLstmInputsVarNum = 11;
constexpr size_t kCondNodesNum = 12;
constexpr size_t kCondCNodesNum = 4;
constexpr size_t kBodyNodesNum = 82;
constexpr size_t kBodyCNodesNum = 30;
const auto &p1 = std::placeholders::_1;

bool IsParameterNode(const BaseRef &n) { return utils::isa<ParameterPtr>(n); }

bool IsOpType(const BaseRef &n, const PrimitivePtr &prim) {
  if (utils::isa<AnfNodePtr>(n)) {
    auto anf_node = utils::cast<AnfNodePtr>(n);
    return CheckPrimitiveType(anf_node, prim);
  }
  return false;
}
}  // namespace

TfLstmCellFusion::TfLstmCellFusion(const std::string &name, bool multigraph)
    : TfliteLstmCellFusion(name, multigraph, kLstmInputsLength, kLstmInputsVarNum, kCondNodesNum, kCondCNodesNum,
                           kBodyNodesNum, kBodyCNodesNum) {
  /*
   * vars for lstm cell input
   * 0:cond 1:body 2:index 3:limit1 4:output 5:cell 6:hidden 7:limit2 8:input 9:kernel 10:bias
   */
  forget_bias_ = std::make_shared<Var>();
}

AnfNodePtr TfLstmCellFusion::GetBodyGraphPattern(const PrimitiveVarMapPtr &primitive_vars) const {
  std::vector<CondVarPtr> placeholders;
  for (int i = 0; i < 10; ++i) {
    placeholders.emplace_back(std::make_shared<CondVar>(IsParameterNode));
  }
  VectorRef add2 = VectorRef({std::make_shared<Var>(), placeholders[2], std::make_shared<CondVar>(IsParameterNode)});
  VectorRef add3 = VectorRef({std::make_shared<Var>(), placeholders[0], std::make_shared<CondVar>(IsParameterNode)});

  VectorRef get_item = VectorRef(
    {std::make_shared<Var>("GetItem"), placeholders[7], placeholders[2], std::make_shared<CondVar>(IsParameterNode)});
  VectorRef concat_input_h = VectorRef({std::make_shared<Var>(), get_item, placeholders[5]});

  VectorRef matmul = VectorRef({std::make_shared<Var>(), concat_input_h, placeholders[8]});
  VectorRef bias = VectorRef({std::make_shared<Var>(), matmul, placeholders[9]});
  VectorRef split = VectorRef({std::make_shared<Var>(), bias});

  VectorRef get_item1 = VectorRef({std::make_shared<Var>(), split, std::make_shared<Var>()});
  VectorRef get_item2 = VectorRef({std::make_shared<Var>(), split, std::make_shared<Var>()});
  VectorRef get_item3 = VectorRef({std::make_shared<Var>(), split, std::make_shared<Var>()});
  VectorRef get_item4 = VectorRef({std::make_shared<Var>(), split, std::make_shared<Var>()});

  VectorRef input_gate = VectorRef({std::make_shared<Var>("Sigmoid"), get_item1});
  VectorRef input_to_cell = VectorRef({std::make_shared<Var>("Tanh"), get_item2});
  VectorRef forget_bias = VectorRef({std::make_shared<Var>("Add"), get_item3, forget_bias_});
  VectorRef forget_gate = VectorRef({std::make_shared<Var>("Sigmoid"), forget_bias});
  VectorRef output_gate = VectorRef({std::make_shared<Var>("Sigmoid"), get_item4});

  VectorRef forgetted_cell = VectorRef({std::make_shared<Var>(""), forget_gate, placeholders[4]});
  VectorRef inputted_cell = VectorRef({std::make_shared<Var>(""), input_gate, input_to_cell});
  VectorRef input_forget_cell = VectorRef({std::make_shared<Var>("Add"), forgetted_cell, inputted_cell});
  VectorRef to_new_hidden = VectorRef({std::make_shared<Var>("Tanh"), input_forget_cell});
  VectorRef new_hidden = VectorRef({std::make_shared<Var>("Mul"), output_gate, to_new_hidden});

  VectorRef new_to_cell = VectorRef({std::make_shared<Var>("Mul"), cell_zoneout_new_, input_forget_cell});
  VectorRef old_to_cell = VectorRef({std::make_shared<Var>("Mul"), cell_zoneout_old_, placeholders[4]});
  VectorRef output_cell = VectorRef({std::make_shared<Var>("Add"), new_to_cell, old_to_cell});

  VectorRef new_to_hidden = VectorRef({std::make_shared<Var>("Mul"), hidden_zoneout_new_, new_hidden});
  VectorRef old_to_hidden = VectorRef({std::make_shared<Var>("Mul"), hidden_zoneout_old_, placeholders[5]});
  VectorRef output_hidden = VectorRef({std::make_shared<Var>("Add"), new_to_hidden, old_to_hidden});

  VectorRef set_item = VectorRef({std::make_shared<Var>(""), placeholders[3], placeholders[2], new_hidden});

  auto is_make_tuple = std::make_shared<CondVar>(std::bind(IsOpType, p1, kPrimMakeTuple));
  std::vector<BaseRef> outputs = {is_make_tuple, add3, placeholders[1], add2, set_item, output_cell, output_hidden};
  outputs.insert(outputs.end(), placeholders.begin() + 6, placeholders.end());
  VectorRef make_tuple_node = VectorRef(outputs);
  auto is_return = std::make_shared<CondVar>(std::bind(IsOpType, p1, kPrimReturn));
  VectorRef return_node = VectorRef({is_return, make_tuple_node});

  VarPtr fg = std::make_shared<Var>("RootG");
  auto pattern = SexpToNode(return_node, fg, primitive_vars.get(), true);
  return pattern;
}

STATUS TfLstmCellFusion::SetWeightAbstractAndDefault(const ParameterPtr &weight, const std::vector<int> &shape,
                                                     const float *const data_ptr, const int hidden_size) const {
  MS_ASSERT(weight != nullptr);
  MS_ASSERT(data_ptr != nullptr);
  auto default_param = std::make_shared<ParamValueLite>();
  if (default_param == nullptr) {
    MS_LOG(ERROR) << "new_default is nullptr";
    return RET_ERROR;
  }
  default_param->set_tensor_shape(shape);
  default_param->set_tensor_type(kNumberTypeFloat32);
  default_param->set_format(schema::Format_NHWC);

  if (shape.size() != 3) {
    MS_LOG(ERROR) << "lstm weight shape must have 3 dims";
    return RET_ERROR;
  }
  const auto param_num = shape[0] * shape[1] * shape[2];
  auto tensor_data = new (std::nothrow) float[param_num * 4];
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
  default_param->SetTensorData(tensor_data, param_num * 4);
  weight->set_default_param(default_param);
  std::vector<int64_t> shape_vector_i(shape.begin(), shape.end());
  auto abstract_tensor_i = std::make_shared<abstract::AbstractTensor>(kFloat32, shape_vector_i);
  if (abstract_tensor_i == nullptr) {
    MS_LOG(ERROR) << "abstract_tensor is nullptr";
    delete[] tensor_data;
    return RET_ERROR;
  }
  weight->set_abstract(abstract_tensor_i);
  return RET_OK;
}

STATUS TfLstmCellFusion::SplitWeights(const AnfNodePtr &weight, const ParameterPtr &weight_i,
                                      const ParameterPtr &weight_c, int hidden_size) const {
  // split input_size and hidden_size at dim 0
  // transform i,c,f,o to i,o,f,c at dim 1
  MS_ASSERT(weight != nullptr);
  MS_ASSERT(wiehgt_i != nullptr);
  MS_ASSERT(wiehgt_c != nullptr);
  if (!utils::isa<ParameterPtr>(weight)) {
    return RET_ERROR;
  }
  auto weight_param = utils::cast<ParameterPtr>(weight);
  if (!weight_param->has_default()) {
    MS_LOG(DEBUG) << "weight not have default value";
    return RET_ERROR;
  }
  if (!utils::isa<ParamValueLitePtr>(weight_param->default_param())) {
    MS_LOG(DEBUG) << "default value is not ParamValueLite";
    return RET_FAILED;
  }
  auto origin_tensor = std::dynamic_pointer_cast<ParamValueLite>(weight_param->default_param());
  if (origin_tensor->tensor_type() != kNumberTypeFloat32 && origin_tensor->tensor_type() != kNumberTypeFloat) {
    MS_LOG(DEBUG) << "origin_tensor is not float32 type";
    return RET_ERROR;
  }
  auto data_ptr = reinterpret_cast<float *>(origin_tensor->tensor_addr());
  auto data_shape = origin_tensor->tensor_shape();
  if (data_shape.size() != 2) {
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

  std::vector<int> shape_i{1, 4 * hidden_size, input_size};
  if (SetWeightAbstractAndDefault(weight_i, shape_i, data_ptr, hidden_size) != RET_OK) {
    MS_LOG(ERROR) << "get weight_i failed";
    return RET_ERROR;
  }

  std::vector<int> shape_c{1, 4 * hidden_size, hidden_size};
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
  if (!old_bias_param->has_default()) {
    MS_LOG(DEBUG) << "bias not have default value";
    return RET_ERROR;
  }
  if (!utils::isa<ParamValueLitePtr>(old_bias_param->default_param())) {
    MS_LOG(DEBUG) << "default value is not ParamValueLite";
    return RET_FAILED;
  }
  auto origin_tensor = std::dynamic_pointer_cast<ParamValueLite>(old_bias_param->default_param());
  if (origin_tensor->tensor_type() != kNumberTypeFloat32 && origin_tensor->tensor_type() != kNumberTypeFloat) {
    MS_LOG(DEBUG) << "origin_tensor is not float32 type";
    return RET_ERROR;
  }
  auto data_ptr = reinterpret_cast<float *>(origin_tensor->tensor_addr());
  auto data_shape = origin_tensor->tensor_shape();
  if (data_shape.size() != 1 || data_shape[0] != 4 * hidden_size) {
    MS_LOG(DEBUG) << "bias data shape illegal";
    return RET_ERROR;
  }
  std::vector<int> shape{1, 8 * hidden_size};

  auto default_param = std::make_shared<ParamValueLite>();
  if (default_param == nullptr) {
    MS_LOG(ERROR) << "new_default is nullptr";
    return RET_ERROR;
  }
  default_param->set_tensor_shape(shape);
  default_param->set_tensor_type(kNumberTypeFloat32);
  default_param->set_format(schema::Format_NHWC);

  std::unique_ptr<float[]> tensor_data(new (std::nothrow) float[hidden_size * 8]);

  auto forget_bias_node = utils::cast<AnfNodePtr>((*body_equiv)[forget_bias_]);
  if (forget_bias_node == nullptr) {
    MS_LOG(ERROR) << "forget bias node is nullptr";
    return RET_ERROR;
  }
  float forget_bias_value = 0.0f;
  if (GetFloatScalarFromParamValueLite(forget_bias_node, &forget_bias_value) != RET_OK) {
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
  default_param->SetTensorData(tensor_data.release(), hidden_size * 8 * 4);
  new_bias->set_default_param(default_param);
  std::vector<int64_t> shape_vector_i(shape.begin(), shape.end());
  auto abstract_tensor_i = std::make_shared<abstract::AbstractTensor>(kFloat32, shape_vector_i);
  if (abstract_tensor_i == nullptr) {
    MS_LOG(ERROR) << "abstract_tensor is nullptr";
    return RET_ERROR;
  }
  new_bias->set_abstract(abstract_tensor_i);
  return RET_OK;
}

CNodePtr TfLstmCellFusion::CreateLSTMNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                          const EquivPtr &body_equiv, const std::string &base_name,
                                          const float zoneout_cell, const float zoneout_hidden) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(equiv != nullptr);
  auto lstm_prim = std::make_shared<ops::LSTM>();
  lstm_prim->set_bidirectional(false);
  lstm_prim->set_zoneout_cell(zoneout_cell);
  lstm_prim->set_zoneout_hidden(zoneout_hidden);
  auto value_node = NewValueNode(lstm_prim);

  auto &vars = while_input_vars_;

  auto limit1 = utils::cast<AnfNodePtr>((*equiv)[vars[3]]);
  MS_ASSERT(limit1);
  auto limit2 = utils::cast<AnfNodePtr>((*equiv)[vars[7]]);
  MS_ASSERT(limit2);
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
  auto hidden_shape = utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape();
  if (hidden_shape.size() == 0) {
    MS_LOG(DEBUG) << "can't get hidden shape";
    return nullptr;
  }

  auto i_weight = func_graph->add_parameter();
  i_weight->set_name(base_name + "_weight_i");
  i_weight->set_abstract(weight->abstract()->Clone());

  auto c_weight = func_graph->add_parameter();
  c_weight->set_name(base_name + "_weight_c");
  c_weight->set_abstract(weight->abstract()->Clone());

  if (SplitWeights(weight, i_weight, c_weight, hidden_shape.back()) != RET_OK) {
    MS_LOG(DEBUG) << "split weight to i_weight and c_weight failed";
    return nullptr;
  }

  auto bias_node = func_graph->add_parameter();
  bias_node->set_name(base_name + "_bias");
  bias_node->set_abstract(bias->abstract()->Clone());

  if (PopulateBiasNode(body_equiv, bias_node, bias, hidden_shape.back()) != RET_OK) {
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
  new_node->set_fullname_with_scope(base_name);
  return new_node;
}
}  // namespace opt
}  // namespace mindspore
