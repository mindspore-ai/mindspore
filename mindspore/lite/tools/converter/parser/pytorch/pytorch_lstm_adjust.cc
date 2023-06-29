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
#define USE_DEPRECATED_API
#include "tools/converter/parser/pytorch/pytorch_lstm_adjust.h"
#include "ops/lstm.h"
#include "src/common/utils.h"
#include "tools/lite_exporter/fetch_content.h"
#include "tools/common/tensor_util.h"
#include "utils/check_convert_utils.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/sequence_ops.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kTorchLstmInputSize = 4;
constexpr size_t kStateTupleInputSize = 3;
constexpr size_t kWeightTupleInputSize = 5;
constexpr size_t kTotalBiasNum = 8;
constexpr size_t kOnnxOutputGateIndex = 1;
constexpr size_t kOnnxForgetGateIndex = 2;
constexpr size_t kOnnxCellGateIndex = 3;
constexpr size_t kInputNumWithProject = 6;
}  // namespace
bool PytorchLstmAdjustPass::Run(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, false, "func_graph is nullptr.");
  auto node_list = TopoSort(func_graph->get_return());
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return false;
  }
  // reorder and adjust torch lstm inputs to conform onnx lstm input specification
  // new inputs are (input, weight_input, weight_hidden, bias, hidden_state, cell_state, project_weight)
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (CheckPrimitiveType(node, prim::kPrimLstm)) {
      auto lstm = node->cast<CNodePtr>();
      MS_CHECK_TRUE_RET(lstm != nullptr, false);
      auto lstm_prim = ops::GetOperator<ops::LSTM>(lstm->input(kAnfPrimitiveIndex));
      MS_CHECK_TRUE_RET(lstm != nullptr, false);

      auto bidirectional = lstm_prim->get_bidirectional();
      int64_t bidirectional_dim = bidirectional ? 2 : 1;
      MS_CHECK_TRUE_MSG(bidirectional == false, false, "torch lstm adjust pass doesn't support bidirectional");
      MS_CHECK_TRUE_MSG(lstm->inputs().size() == kTorchLstmInputSize, false, "torch lstm inputs size is not 4.");
      // unpack state tuple
      MS_CHECK_TRUE_MSG(CheckPrimitiveType(lstm->input(THIRD_INPUT), prim::kPrimMakeTuple), false,
                        "lstm's second input is not make tuple");
      auto state_tuple = utils::cast<CNodePtr>(lstm->input(THIRD_INPUT));
      MS_CHECK_TRUE_MSG(state_tuple->size() == kStateTupleInputSize, false, "lstm's state tuple size is not 3");
      auto hidden_state = state_tuple->input(SECOND_INPUT);
      auto cell_state = state_tuple->input(THIRD_INPUT);
      // unpack weight and bias tuple
      MS_CHECK_TRUE_MSG(CheckPrimitiveType(lstm->input(FOURTH_INPUT), prim::kPrimMakeTuple), false,
                        "lstm's third input is not make tuple");
      auto weight_tuple = utils::cast<CNodePtr>(lstm->input(FOURTH_INPUT));
      MS_CHECK_TRUE_MSG(weight_tuple->size() >= kWeightTupleInputSize, false, "weight tuple input size is less than 5");
      auto weight_input = utils::cast<ParameterPtr>(weight_tuple->input(SECOND_INPUT));
      MS_CHECK_TRUE_RET(AdjustDataFormat(weight_input), false);
      auto weight_hidden = utils::cast<ParameterPtr>(weight_tuple->input(THIRD_INPUT));
      MS_CHECK_TRUE_RET(AdjustDataFormat(weight_hidden), false);
      auto bias_input = utils::cast<ParameterPtr>(weight_tuple->input(FOURTH_INPUT));
      MS_CHECK_TRUE_RET(AdjustDataFormat(bias_input), false);
      auto bias_hidden = utils::cast<ParameterPtr>(weight_tuple->input(FIFTH_INPUT));
      MS_CHECK_TRUE_RET(AdjustDataFormat(bias_hidden), false);
      // process project weight
      bool has_project = false;
      ParameterPtr project_weight = nullptr;
      if (weight_tuple->size() >= kInputNumWithProject) {
        has_project = true;
        project_weight = utils::cast<ParameterPtr>(weight_tuple->input(SIXTH_INPUT));
        MS_CHECK_TRUE_RET(project_weight != nullptr, false);
      }

      int64_t hidden_size = 0;
      if (!GetAndSetHiddenSize(weight_input, lstm_prim, &hidden_size)) {
        MS_LOG(ERROR) << "GetAndSetHiddenSize failed";
        return false;
      }
      if (AdjustInputShape(weight_input, weight_hidden, bidirectional_dim) != RET_OK) {
        MS_LOG(ERROR) << "Adjust input shape failed";
        return false;
      }
      auto bias_parameter =
        CombineTwoBiasInput(bias_input, bias_hidden, func_graph, lstm, bidirectional_dim, hidden_size);
      if (bias_parameter == nullptr) {
        MS_LOG(ERROR) << "CombineTwoBiasInput failed";
        return false;
      }
      std::vector<AnfNodePtr> op_inputs{lstm->input(kAnfPrimitiveIndex),
                                        lstm->input(SECOND_INPUT),
                                        weight_input,
                                        weight_hidden,
                                        bias_parameter,
                                        hidden_state,
                                        cell_state};
      if (has_project) {
        auto proj_param = project_weight->default_param();
        MS_CHECK_TRUE_RET(proj_param != nullptr, false);
        auto proj_tensor = proj_param->cast<tensor::TensorPtr>();
        auto proj_shape = proj_tensor->shape();
        proj_shape.insert(proj_shape.begin(), bidirectional_dim);
        proj_tensor->set_shape(proj_shape);
        project_weight->set_abstract(proj_tensor->ToAbstract());
        op_inputs.emplace_back(project_weight);
      }

      auto adjusted_lstm = func_graph->NewCNode(op_inputs);
      MS_CHECK_TRUE_RET(adjusted_lstm != nullptr, false);
      adjusted_lstm->set_abstract(lstm->abstract()->Clone());
      auto ret = manager->Replace(lstm, adjusted_lstm);
      if (!ret) {
        MS_LOG(ERROR) << "Replace old lstm to new lstm failed.";
        return false;
      }
    }
  }
  return true;
}

// torch lstm weight whose format is ifco (input forget cell output) will be converted to onnx data format iofc
bool PytorchLstmAdjustPass::AdjustDataFormat(const ParameterPtr &parameter) {
  MS_CHECK_TRUE_RET(parameter != nullptr, false);
  auto param = parameter->default_param();
  MS_CHECK_TRUE_RET(param != nullptr, false);
  auto weight = param->cast<tensor::TensorPtr>();
  MS_CHECK_TRUE_RET(weight != nullptr, false);
  auto data_size = weight->Size();
  auto gate_data_size = data_size / kInputSizeFour;
  auto data = weight->data_c();
  auto input_gate_data = reinterpret_cast<uint8_t *>(data);
  auto forget_gate_data = input_gate_data + gate_data_size;
  auto cell_gate_data = forget_gate_data + gate_data_size;
  auto output_gate_data = cell_gate_data + gate_data_size;

  if (data_size > MAX_MALLOC_SIZE) {
    MS_LOG(ERROR) << "allocate data size exceeds MAX_MALLOC_SIZE";
    return false;
  }
  auto new_data = reinterpret_cast<uint8_t *>(malloc(data_size));
  if (new_data == nullptr) {
    MS_LOG(ERROR) << "malloc new buffer failed.";
    return false;
  }

  if (memcpy_s(new_data, data_size, input_gate_data, gate_data_size) != EOK) {
    MS_LOG(ERROR) << "memcpy data to new buffer failed.";
    free(new_data);
    return false;
  }

  if (memcpy_s(new_data + gate_data_size * kOnnxOutputGateIndex, data_size - gate_data_size * kOnnxOutputGateIndex,
               output_gate_data, gate_data_size) != EOK) {
    MS_LOG(ERROR) << "memcpy data to new buffer failed.";
    free(new_data);
    return false;
  }

  if (memcpy_s(new_data + gate_data_size * kOnnxForgetGateIndex, data_size - gate_data_size * kOnnxForgetGateIndex,
               forget_gate_data, gate_data_size) != EOK) {
    MS_LOG(ERROR) << "memcpy data to new buffer failed.";
    free(new_data);
    return false;
  }

  if (memcpy_s(new_data + gate_data_size * kOnnxCellGateIndex, data_size - gate_data_size * kOnnxCellGateIndex,
               cell_gate_data, gate_data_size) != EOK) {
    MS_LOG(ERROR) << "memcpy data to new buffer failed.";
    free(new_data);
    return false;
  }

  auto new_tensor = std::make_shared<tensor::Tensor>(weight->data_type(), weight->shape(), new_data, data_size);
  MS_CHECK_TRUE_RET(new_tensor != nullptr, false);
  parameter->set_default_param(new_tensor);

  return true;
}

bool PytorchLstmAdjustPass::AdjustInputShape(const ParameterPtr &weight_input, const ParameterPtr &weight_hidden,
                                             const int64_t &bidirectional_dim) {
  auto weight_param = weight_input->default_param();
  MS_CHECK_TRUE_RET(weight_param != nullptr, false);
  auto weight_input_tensor = weight_param->cast<tensor::TensorPtr>();
  MS_CHECK_TRUE_RET(weight_input_tensor != nullptr, false);
  auto input_weight_shape = weight_input_tensor->shape();

  input_weight_shape.insert(input_weight_shape.begin(), bidirectional_dim);
  weight_input_tensor->set_shape(input_weight_shape);
  weight_input->set_abstract(weight_input_tensor->ToAbstract());

  auto weight_hidden_param = weight_hidden->default_param();
  MS_CHECK_TRUE_RET(weight_hidden_param != nullptr, false);
  auto weight_hidden_tensor = weight_hidden_param->cast<tensor::TensorPtr>();
  MS_CHECK_TRUE_RET(weight_hidden_tensor != nullptr, false);
  auto hidden_weight_shape = weight_hidden_tensor->shape();
  hidden_weight_shape.insert(hidden_weight_shape.begin(), bidirectional_dim);
  weight_hidden_tensor->set_shape(hidden_weight_shape);
  weight_hidden->set_abstract(weight_hidden_tensor->ToAbstract());
  return true;
}

ParameterPtr PytorchLstmAdjustPass::CombineTwoBiasInput(const ParameterPtr &bias_input, const ParameterPtr &bias_hidden,
                                                        const FuncGraphPtr &func_graph, const CNodePtr &lstm,
                                                        const int64_t &bidirectional_dim, const int64_t &hidden_size) {
  ShapeVector bias_shape{bidirectional_dim, static_cast<int64_t>(kTotalBiasNum * hidden_size)};

  auto bias_input_param = bias_input->default_param();
  if (bias_input_param == nullptr) {
    return nullptr;
  }
  auto bias_input_tensor = bias_input_param->cast<tensor::TensorPtr>();
  auto bias_hidden_param = bias_hidden->default_param();
  if (bias_hidden_param == nullptr) {
    return nullptr;
  }
  auto bias_hidden_tensor = bias_hidden_param->cast<tensor::TensorPtr>();
  // combine two bias tensor
  auto bias_input_data_type = bias_input_tensor->data_type();
  auto data_size =
    bidirectional_dim * kTotalBiasNum * hidden_size * mindspore::abstract::TypeIdSize(bias_input_data_type);
  auto combined_buffer = reinterpret_cast<uint8_t *>(malloc(data_size));
  if (combined_buffer == nullptr) {
    MS_LOG(ERROR) << "malloc lstm bias buffer failed!";
    return nullptr;
  }

  if (memcpy_s(combined_buffer, data_size, bias_input_tensor->data_c(), data_size / Num2) != EOK) {
    MS_LOG(ERROR) << "memcpy data to combined_buffer failed.";
    free(combined_buffer);
    return nullptr;
  }
  if (memcpy_s(combined_buffer + data_size / Num2, data_size / Num2, bias_hidden_tensor->data_c(), data_size / Num2) !=
      EOK) {
    MS_LOG(ERROR) << "memcpy data to combined_buffer failed.";
    free(combined_buffer);
    return nullptr;
  }

  auto tensor_info = lite::CreateTensorInfo(combined_buffer, data_size, bias_shape, bias_input_data_type);
  auto bias_parameter = opt::BuildParameterNode(func_graph, tensor_info, lstm->fullname_with_scope() + "_bias");
  return bias_parameter;
}

bool PytorchLstmAdjustPass::GetAndSetHiddenSize(const ParameterPtr &weight_input,
                                                const api::SharedPtr<ops::LSTM> &lstm_prim, int64_t *hidden_size) {
  MS_ASSERT(hidden_size != nullptr);
  auto weight_param = weight_input->default_param();
  MS_CHECK_TRUE_RET(weight_param != nullptr, false);
  auto weight_input_tensor = weight_param->cast<tensor::TensorPtr>();
  MS_CHECK_TRUE_RET(weight_input_tensor != nullptr, false);
  auto input_weight_shape = weight_input_tensor->shape();
  *hidden_size = input_weight_shape[kDim0] / kInputSizeFour;
  lstm_prim->set_hidden_size(*hidden_size);
  return true;
}
}  // namespace opt
}  // namespace mindspore
