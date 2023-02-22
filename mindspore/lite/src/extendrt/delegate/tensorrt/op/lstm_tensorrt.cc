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

#include <memory>
#include "src/extendrt/delegate/tensorrt/op/lstm_tensorrt.h"
#include "src/extendrt/delegate/tensorrt/op/lstm_plugin.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_runtime.h"
#include "ops/lstm.h"

namespace mindspore::lite {
int LSTMTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                            const std::vector<TensorInfo> &out_tensors) {
#if TRT_VERSION_GE(7, 0)
  if (in_tensors.size() != INPUT_TENSOR_SIZE && in_tensors.size() != INPUT_SIZE4) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != OUTPUT_TENSOR_SIZE && out_tensors.size() != INPUT_SIZE5) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  dynamic_shape_params_.support_dynamic_ = false;
  dynamic_shape_params_.support_hw_dynamic_ = false;
  return RET_OK;
#else
  MS_LOG(WARNING) << "low TensorRT version don't support LSTM op, please upgrade TensorRT version to 7 or higher";
  return RET_ERROR;
#endif
}

int LSTMTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid";
    return RET_ERROR;
  }
  int input_data_dims_cnt = input(ctx, 0).trt_tensor_->getDimensions().nbDims;
  if (input_data_dims_cnt != DIMENSION_3D) {
    MS_LOG(ERROR) << "invalid input data shape dims for " << op_name_;
    return RET_ERROR;
  }
  int ret = PreProcess(ctx);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PreProcess for " << op_name_;
    return ret;
  }
  if (in_tensors_.size() == INPUT_SIZE4) {
    return RunLSTMPlugin(ctx);
  }

  ret = AddLSTMLayers(ctx);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "AddLSTMLayers for " << op_name_;
    return RET_ERROR;
  }

  if (op_data_out_ == nullptr) {
    MS_LOG(ERROR) << "layers final output tensor is invalid for " << op_name_;
    return RET_ERROR;
  }
  MS_LOG(DEBUG) << "lstm op_data_out_ " << GetTensorFormat(op_data_out_);
  MS_LOG(DEBUG) << "lstm op_hidden_out_ " << GetTensorFormat(op_hidden_out_);
  MS_LOG(DEBUG) << "lstm op_cell_out_ " << GetTensorFormat(op_cell_out_);
  ctx->RegisterTensor(ITensorHelper{op_data_out_}, out_tensors_[0].Name());
  ctx->RegisterTensor(ITensorHelper{op_hidden_out_}, out_tensors_[OUTPUT_HIDDEN_INDEX].Name());
  ctx->RegisterTensor(ITensorHelper{op_cell_out_}, out_tensors_[OUTPUT_CELL_INDEX].Name());
  return RET_OK;
}

int LSTMTensorRT::RunLSTMPlugin(TensorRTContext *ctx) {
  auto lstm_op = AsOps<ops::LSTM>();
  if (params_.directional_cnt_ == BIDIRECTIONAL) {
    MS_LOG(ERROR) << "mindir lstm with bidirectional not support yet";
    return RET_ERROR;
  }
  auto plugin = std::make_shared<LSTMPlugin>(op_name_, params_.layer_count_, params_.batch_size_,
                                             params_.sequence_size_, params_.input_data_size_, params_.hidden_size_,
                                             0.f, lstm_op->get_has_bias(), false, device_id_);
  CHECK_NULL_RETURN(plugin);
  nvinfer1::ITensor *inputTensors[] = {input(ctx, 0).trt_tensor_, input(ctx, 1).trt_tensor_,
                                       input(ctx, INPUT_SIZE2).trt_tensor_, input(ctx, INPUT_SIZE3).trt_tensor_};
  nvinfer1::IPluginV2Layer *lstm_layer = ctx->network()->addPluginV2(inputTensors, INPUT_SIZE4, *plugin);
  CHECK_NULL_RETURN(lstm_layer);
  ctx->RegisterTensor(ITensorHelper{lstm_layer->getOutput(0)}, out_tensors_[0].Name());
  return RET_OK;
}

int LSTMTensorRT::PreProcess(TensorRTContext *ctx) {
  auto ms_input_shape = ConvertMSShape(input(ctx, 0).trt_tensor_->getDimensions());
  params_.sequence_size_ = ms_input_shape[0];
  params_.batch_size_ = ms_input_shape[1];
  params_.input_data_size_ = ms_input_shape[INPUT_SIZE_INDEX];
  if (params_.batch_size_ != 1) {
    MS_LOG(WARNING) << op_name_ << " lstm has batchsize " << params_.batch_size_ << ", needs further verify";
  }
  auto lstm_op = AsOps<ops::LSTM>();
  params_.layer_count_ = lstm_op->get_num_layers() == 0 ? 1 : lstm_op->get_num_layers();
  params_.hidden_size_ = lstm_op->get_hidden_size();
  params_.directional_cnt_ = lstm_op->get_bidirectional() ? BIDIRECTIONAL : 1;
  params_.data_type_ = ConvertDataType(in_tensors_[1].DataType());
  if (in_tensors_.size() == INPUT_SIZE4) {
    return RET_OK;
  }
  // ms: 0 sequence size, 1 batch size, 2 input size -> tensorrt: 0 batch size, 1 sequence size, 2 input size
  auto transpose_in_layer = ctx->network()->addShuffle(*input(ctx, 0).trt_tensor_);
  if (transpose_in_layer == nullptr) {
    MS_LOG(ERROR) << "create transpose_in_layer failed for " << op_name_;
    return RET_ERROR;
  }
  this->layer_ = transpose_in_layer;
  nvinfer1::Permutation transpose_perm{{1, 0, INPUT_SIZE_INDEX}};
  transpose_in_layer->setFirstTranspose(transpose_perm);
  transpose_in_layer->setName((op_name_ + "transpose_in").c_str());
  input_data_ = transpose_in_layer->getOutput(0);
  MS_LOG(DEBUG) << "lstm input " << GetTensorFormat(input_data_);

  return RET_OK;
}

int LSTMTensorRT::AddLSTMLayers(TensorRTContext *ctx) {
  nvinfer1::ITensor *data_out{nullptr};
  nvinfer1::ITensor *hidden_init{nullptr};
  nvinfer1::ITensor *cell_init{nullptr};
  if (in_tensors_[HIDDEN_IN_TENSOR_INIT].Data() != nullptr && in_tensors_[CELL_IN_TENSOR_INIT].Data() != nullptr) {
    TensorInfo &hidden_in_init = in_tensors_[HIDDEN_IN_TENSOR_INIT];
    TensorInfo &cell_in_init = in_tensors_[CELL_IN_TENSOR_INIT];
    hidden_init_name_ = hidden_in_init.Name() + "_hidden_init";

    hidden_init = ctx->network()->addInput(
      hidden_init_name_.c_str(), nvinfer1::DataType::kFLOAT,
      nvinfer1::Dims3(params_.layer_count_ * params_.directional_cnt_, params_.batch_size_, params_.hidden_size_));
    if (hidden_init == nullptr) {
      MS_LOG(ERROR) << "add hidden_init input tensor failed for " << op_name_;
      return RET_ERROR;
    }
    op_binding_tensor_.push_back(BindingHelper{hidden_init_name_, hidden_in_init.MutableData(),
                                               nvinfer1::DataType::kFLOAT, hidden_in_init.DataSize()});
    cell_init_name_ = cell_in_init.Name() + "_cell_init";
    cell_init = ctx->network()->addInput(
      cell_init_name_.c_str(), nvinfer1::DataType::kFLOAT,
      nvinfer1::Dims3(params_.layer_count_ * params_.directional_cnt_, params_.batch_size_, params_.hidden_size_));
    if (cell_init == nullptr) {
      MS_LOG(ERROR) << "add cell_init input tensor failed for " << op_name_;
      return RET_ERROR;
    }
    op_binding_tensor_.push_back(
      BindingHelper{cell_init_name_, cell_in_init.MutableData(), nvinfer1::DataType::kFLOAT, cell_in_init.DataSize()});

    sequence_size_input_ =
      ctx->network()->addInput((op_name_ + "_seq_input").c_str(), nvinfer1::DataType::kINT32, nvinfer1::Dims{});
    if (sequence_size_input_ == nullptr) {
      MS_LOG(ERROR) << "add sequence_size_input_ input tensor failed for " << op_name_;
      return RET_ERROR;
    }
    op_binding_tensor_.push_back(
      BindingHelper{(op_name_ + "_seq_input"), &params_.sequence_size_, nvinfer1::DataType::kINT32, sizeof(int)});
  } else {
    hidden_init = input(ctx, HIDDEN_IN_TENSOR_INIT).trt_tensor_;
    cell_init = input(ctx, CELL_IN_TENSOR_INIT).trt_tensor_;
    sequence_size_input_ = ctx->ConvertTo0DTensor(input(ctx, 0).trt_tensor_->getDimensions().d[0]);
  }

  nvinfer1::ITensor *max_sequence_size =
    ctx->network()
      ->addConstant(nvinfer1::Dims{}, nvinfer1::Weights{nvinfer1::DataType::kINT32, &params_.sequence_size_, 1})
      ->getOutput(0);
  if (max_sequence_size == nullptr) {
    MS_LOG(ERROR) << "add max_sequence_size constant tensor failed for " << op_name_;
    return RET_ERROR;
  }
  LstmState next_state{input_data_, nullptr, nullptr};  // init states
  std::vector<nvinfer1::ITensor *> hidden_outputs;
  std::vector<nvinfer1::ITensor *> cell_outputs;
  int input_weight_offset = 0;
  int state_weight_offset = 0;
  int bias_offset = 0;

  if (params_.layer_count_ != 1) {
    MS_LOG(WARNING) << op_name_ << " needs verify for layer cnt: " << params_.layer_count_;
  }
  for (int i = 0; i < params_.layer_count_; i++) {
    LstmState layer_input_states[BIDIRECTIONAL];
    LstmWeights layer_weights[BIDIRECTIONAL];
    layer_weights[0].max_seq_size_ = max_sequence_size;
    int ret = ParseLSTMCellInputs(ctx, i, hidden_init, cell_init, layer_input_states, &input_weight_offset,
                                  &state_weight_offset, &bias_offset, layer_weights, next_state);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "ParseLSTMCellInputs failed for " << op_name_;
      return RET_ERROR;
    }
    data_out = AddLSTMCell(ctx, layer_input_states, layer_weights, &next_state);
    hidden_outputs.push_back(next_state.hidden_);
    cell_outputs.push_back(next_state.cell_);
    if (data_out == nullptr || next_state.hidden_ == nullptr || next_state.cell_ == nullptr) {
      MS_LOG(ERROR) << "AddLSTMCell failed for " << op_name_;
      return RET_ERROR;
    }
  }

  op_hidden_out_ = ConcateAll(ctx, hidden_outputs);
  if (op_hidden_out_ == nullptr) {
    MS_LOG(ERROR) << "concat hidden output failed for " << op_name_;
    return RET_ERROR;
  }
  op_cell_out_ = ConcateAll(ctx, cell_outputs);
  if (op_cell_out_ == nullptr) {
    MS_LOG(ERROR) << "concat cell output failed for " << op_name_;
    return RET_ERROR;
  }
  op_data_out_ = data_out;
  return RET_OK;
}

int LSTMTensorRT::ParseLSTMCellInputs(TensorRTContext *ctx, int layer_index, nvinfer1::ITensor *hidden_init,
                                      nvinfer1::ITensor *cell_init, LstmState *layer_input_states,
                                      int *input_weight_offset, int *state_weight_offset, int *bias_offset,
                                      LstmWeights *layer_weights, const LstmState &next_state) {
  nvinfer1::Dims2 dim_input_weight(LSTM_GATE_NUM * params_.hidden_size_, params_.input_data_size_);
  nvinfer1::Dims2 dim_state_weight(LSTM_GATE_NUM * params_.hidden_size_, params_.hidden_size_);
  nvinfer1::Dims dim_bias{1, {LSTM_GATE_NUM * params_.hidden_size_}};

  TensorInfo &input_weight = in_tensors_[INPUT_WEIGHT];
  TensorInfo &state_weight = in_tensors_[STATE_WEIGHT];
  TensorInfo &bias = in_tensors_[BIAS];

  nvinfer1::Dims dimW = layer_index == 0 ? dim_input_weight : dim_state_weight;

  for (int direction_index = 0; direction_index < params_.directional_cnt_; direction_index++) {
    nvinfer1::ITensor *index =
      ctx->network()
        ->addConstant(nvinfer1::Dims{},
                      nvinfer1::Weights{nvinfer1::DataType::kINT32,
                                        &INDICES[layer_index * params_.directional_cnt_ + direction_index], 1})
        ->getOutput(0);
    MS_ASSERT(index);
    layer_input_states[direction_index].data_ = next_state.data_;
    layer_input_states[direction_index].hidden_ = ctx->network()->addGather(*hidden_init, *index, 0)->getOutput(0);
    layer_input_states[direction_index].cell_ = ctx->network()->addGather(*cell_init, *index, 0)->getOutput(0);
    MS_ASSERT(layer_input_states[direction_index].hidden_);
    MS_ASSERT(layer_input_states[direction_index].cell_);

    // weight order: input, output, forget, cell
    if (params_.data_type_ != nvinfer1::DataType::kFLOAT) {
      MS_LOG(WARNING) << "more data type need to be done";
      return RET_ERROR;
    }
    const float *input_weight_ptr = static_cast<const float *>(input_weight.Data());
    const float *state_weight_ptr = static_cast<const float *>(state_weight.Data());
    const float *bias_ptr = static_cast<const float *>(bias.Data());
    nvinfer1::Weights slice_input_weight{params_.data_type_, input_weight_ptr + *input_weight_offset,
                                         GetDimsVolume(dimW)};
    (*input_weight_offset) += slice_input_weight.count;
    nvinfer1::Weights slice_state_weight{params_.data_type_, state_weight_ptr + *state_weight_offset,
                                         GetDimsVolume(dim_state_weight)};
    (*state_weight_offset) += slice_state_weight.count;
    layer_weights[direction_index].input_weights_ = ctx->network()->addConstant(dimW, slice_input_weight)->getOutput(0);
    layer_weights[direction_index].state_weights_ =
      ctx->network()->addConstant(dim_state_weight, slice_state_weight)->getOutput(0);
    MS_ASSERT(layer_weights[direction_index].input_weights_);
    MS_ASSERT(layer_weights[direction_index].state_weights_);

    // bias
    nvinfer1::Weights slice_input_bias{params_.data_type_, bias_ptr + *bias_offset, GetDimsVolume(dim_bias)};
    (*bias_offset) += slice_input_bias.count;
    nvinfer1::Weights slice_state_bias{params_.data_type_, bias_ptr + *bias_offset, GetDimsVolume(dim_bias)};
    (*bias_offset) += slice_state_bias.count;
    layer_weights[direction_index].input_bias_ = ctx->network()->addConstant(dim_bias, slice_input_bias)->getOutput(0);
    layer_weights[direction_index].state_bias_ = ctx->network()->addConstant(dim_bias, slice_state_bias)->getOutput(0);
    MS_ASSERT(layer_weights[direction_index].input_bias_);
    MS_ASSERT(layer_weights[direction_index].state_bias_);
  }
  if (params_.directional_cnt_ == BIDIRECTIONAL) {
    layer_weights[1].max_seq_size_ = layer_weights[0].max_seq_size_;
  }
  return RET_OK;
}

nvinfer1::ITensor *LSTMTensorRT::Reshape(TensorRTContext *ctx, nvinfer1::ITensor *tensor, nvinfer1::Dims dims) {
  nvinfer1::IShuffleLayer *shuffle = ctx->network()->addShuffle(*tensor);
  shuffle->setReshapeDimensions(dims);
  return shuffle->getOutput(0);
}

nvinfer1::ITensor *LSTMTensorRT::ConcateAll(TensorRTContext *ctx, std::vector<nvinfer1::ITensor *> all_tensor,
                                            int axis) {
  if (all_tensor.size() == 1) {
    return all_tensor[0];
  }
  nvinfer1::IConcatenationLayer *concat = ctx->network()->addConcatenation(all_tensor.data(), all_tensor.size());
  if (concat == nullptr) {
    MS_LOG(ERROR) << "addConcatenation failed for " << op_name_;
    return nullptr;
  }
  if (axis >= all_tensor[0]->getDimensions().nbDims) {
    MS_LOG(ERROR) << op_name_ << " concat axis is " << axis << ", larger than tensor dims "
                  << all_tensor[0]->getDimensions().nbDims;
    return nullptr;
  }
  concat->setAxis(axis);
  this->layer_ = concat;
  return concat->getOutput(0);
}

nvinfer1::ITensor *LSTMTensorRT::AddLSTMCell(TensorRTContext *ctx, const LstmState *layer_input_states,
                                             const LstmWeights *layer_weights, LstmState *next_state) {
  nvinfer1::ITensor *backward_output = nullptr;
  nvinfer1::ITensor *backward_hidden_out = nullptr;
  nvinfer1::ITensor *backward_cell_out = nullptr;
  nvinfer1::ITensor *forward_hidden_out = nullptr;
  nvinfer1::ITensor *forward_cell_out = nullptr;

  nvinfer1::ITensor *forward_output =
    AddLSTMCalculation(ctx, layer_input_states[0], layer_weights[0], &forward_hidden_out, &forward_cell_out);
  if (params_.directional_cnt_ == BIDIRECTIONAL) {
    backward_output =
      AddLSTMCalculation(ctx, layer_input_states[1], layer_weights[1], &backward_hidden_out, &backward_cell_out, true);
  }

  // concate forward and backward
  nvinfer1::ITensor *output_tensor = forward_output;
  nvinfer1::ITensor *cell_out = forward_cell_out;
  nvinfer1::ITensor *hidden_out = forward_hidden_out;
  if (backward_output != nullptr && backward_hidden_out != nullptr && backward_cell_out != nullptr) {
    nvinfer1::ITensor *output_concat_input[BIDIRECTIONAL] = {forward_output, backward_output};
    auto ouput_out_layer = ctx->network()->addConcatenation(output_concat_input, BIDIRECTIONAL);
    this->layer_ = ouput_out_layer;
    if (ouput_out_layer == nullptr) {
      MS_LOG(ERROR) << "create one loop output concat failed for " << op_name_;
      return nullptr;
    }
    ouput_out_layer->setAxis(1);  // ms: 0 sequence size, 1 layer * direction, 2 batchsize, 3 hidden
    output_tensor = ouput_out_layer->getOutput(0);

    nvinfer1::ITensor *hidden_concat_input[BIDIRECTIONAL] = {forward_hidden_out, backward_hidden_out};
    auto hidden_out_layer = ctx->network()->addConcatenation(hidden_concat_input, BIDIRECTIONAL);
    hidden_out_layer->setAxis(0);
    hidden_out = hidden_out_layer->getOutput(0);

    nvinfer1::ITensor *cell_concat_input[BIDIRECTIONAL] = {forward_cell_out, backward_cell_out};
    auto cell_out_layer = ctx->network()->addConcatenation(cell_concat_input, BIDIRECTIONAL);
    cell_out_layer->setAxis(0);
    cell_out = cell_out_layer->getOutput(0);
  }
  if (hidden_out == nullptr || cell_out == nullptr) {
    MS_LOG(ERROR) << "get one loop hidden_out and cell_out failed for " << op_name_;
    return nullptr;
  }
  *next_state = LstmState{output_tensor, hidden_out, cell_out};
  return output_tensor;
}
nvinfer1::ITensor *LSTMTensorRT::AddLSTMCalculation(TensorRTContext *ctx, const LstmState &input_state,
                                                    const LstmWeights &lstm_weights, nvinfer1::ITensor **hidden_out,
                                                    nvinfer1::ITensor **cell_out, bool is_backward) {
  std::vector<nvinfer1::ITensor *> all_batch_outputs;
  std::vector<nvinfer1::ITensor *> all_batch_hidden;
  std::vector<nvinfer1::ITensor *> all_batch_cell;
  for (int batch_index = 0; batch_index < params_.batch_size_; batch_index++) {
    LstmState one_batch_input_state;
    nvinfer1::ITensor *batch_index_tensor =
      ctx->network()
        ->addConstant(nvinfer1::Dims{}, nvinfer1::Weights{nvinfer1::DataType::kINT32, &INDICES[batch_index], 1})
        ->getOutput(0);
    one_batch_input_state.data_ = ctx->network()->addGather(*input_state.data_, *batch_index_tensor, 0)->getOutput(0);
    one_batch_input_state.hidden_ =
      ctx->network()->addGather(*input_state.hidden_, *batch_index_tensor, 0)->getOutput(0);
    one_batch_input_state.cell_ = ctx->network()->addGather(*input_state.cell_, *batch_index_tensor, 0)->getOutput(0);
    nvinfer1::ITensor *one_batch_hidden = nullptr;
    nvinfer1::ITensor *one_batch_cell = nullptr;
    nvinfer1::ITensor *one_batch_output =
      AddLSTMOneLoop(ctx, one_batch_input_state, lstm_weights, &one_batch_hidden, &one_batch_cell, is_backward);
    if (one_batch_output == nullptr || one_batch_cell == nullptr || one_batch_hidden == nullptr) {
      MS_LOG(ERROR) << "AddLSTMOneLoop failed for " << op_name_ << " at batch index " << batch_index;
      return nullptr;
    }
    all_batch_outputs.push_back(one_batch_output);
    all_batch_hidden.push_back(one_batch_hidden);
    all_batch_cell.push_back(one_batch_cell);
  }
  *hidden_out = ConcateAll(ctx, all_batch_hidden, 1);
  *cell_out = ConcateAll(ctx, all_batch_cell, 1);
  return ConcateAll(ctx, all_batch_outputs, BATCH_SIZE_INDEX);
}

nvinfer1::ITensor *LSTMTensorRT::AddLSTMOneLoop(TensorRTContext *ctx, const LstmState &input_state,
                                                const LstmWeights &lstm_weights, nvinfer1::ITensor **hidden_out,
                                                nvinfer1::ITensor **cell_out, bool is_backward) {
#if TRT_VERSION_GE(7, 0)
  nvinfer1::ILoop *sequence_loop = ctx->network()->addLoop();
  if (sequence_loop == nullptr) {
    MS_LOG(ERROR) << "add sequence_loop layer failed for " << op_name_;
    return nullptr;
  }
  std::string loop_name = op_name_ + "_loop" + (is_backward ? "_backward" : "_forward");
  sequence_loop->setName(loop_name.c_str());
  sequence_loop->addTripLimit(*sequence_size_input_, nvinfer1::TripLimit::kCOUNT);
  nvinfer1::ITensor *input = sequence_loop->addIterator(*input_state.data_, 0, is_backward)->getOutput(0);

  nvinfer1::ILayer *hidden_mid = sequence_loop->addRecurrence(*input_state.hidden_);
  if (hidden_mid == nullptr) {
    MS_LOG(ERROR) << "add hidden layer failed for " << op_name_;
    return nullptr;
  }
  nvinfer1::ILayer *cell_mid = sequence_loop->addRecurrence(*input_state.cell_);
  if (cell_mid == nullptr) {
    MS_LOG(ERROR) << "add cell layer failed for " << op_name_;
    return nullptr;
  }

  nvinfer1::ITensor *input_matmul =
    ctx->network()
      ->addMatrixMultiply(*input, nvinfer1::MatrixOperation::kVECTOR, *lstm_weights.input_weights_,
                          nvinfer1::MatrixOperation::kTRANSPOSE)
      ->getOutput(0);

  nvinfer1::ITensor *hidden_matmul =
    ctx->network()
      ->addMatrixMultiply(*hidden_mid->getOutput(0), nvinfer1::MatrixOperation::kVECTOR, *lstm_weights.state_weights_,
                          nvinfer1::MatrixOperation::kTRANSPOSE)
      ->getOutput(0);

  nvinfer1::ITensor *weights_add =
    ctx->network()->addElementWise(*input_matmul, *hidden_matmul, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);

  nvinfer1::ITensor *bias =
    ctx->network()
      ->addElementWise(*lstm_weights.input_bias_, *lstm_weights.state_bias_, nvinfer1::ElementWiseOperation::kSUM)
      ->getOutput(0);

  nvinfer1::ITensor *gates_calculate =
    ctx->network()->addElementWise(*weights_add, *bias, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);

  const auto isolateGate = [&](nvinfer1::ITensor &gates, int gateIndex) -> nvinfer1::ITensor * {
    nvinfer1::ISliceLayer *slice =
      ctx->network()->addSlice(gates, nvinfer1::Dims{1, {gateIndex * params_.hidden_size_}},
                               nvinfer1::Dims{1, {params_.hidden_size_}}, nvinfer1::Dims{1, {1}});
    return Reshape(ctx, slice->getOutput(0), nvinfer1::Dims{1, {params_.hidden_size_}});
  };
  // weight order: input, output, forget, cell
  nvinfer1::ITensor *i =
    ctx->network()->addActivation(*isolateGate(*gates_calculate, 0), nvinfer1::ActivationType::kSIGMOID)->getOutput(0);

  nvinfer1::ITensor *o =
    ctx->network()->addActivation(*isolateGate(*gates_calculate, 1), nvinfer1::ActivationType::kSIGMOID)->getOutput(0);

  nvinfer1::ITensor *f =
    ctx->network()
      ->addActivation(*isolateGate(*gates_calculate, FORGET_GATE), nvinfer1::ActivationType::kSIGMOID)
      ->getOutput(0);

  nvinfer1::ITensor *c = ctx->network()
                           ->addActivation(*isolateGate(*gates_calculate, CELL_GATE), nvinfer1::ActivationType::kTANH)
                           ->getOutput(0);

  nvinfer1::ITensor *C =
    ctx->network()
      ->addElementWise(*ctx->network()
                          ->addElementWise(*f, *cell_mid->getOutput(0), nvinfer1::ElementWiseOperation::kPROD)
                          ->getOutput(0),
                       *ctx->network()->addElementWise(*i, *c, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0),
                       nvinfer1::ElementWiseOperation::kSUM)
      ->getOutput(0);
  nvinfer1::ITensor *H =
    ctx->network()
      ->addElementWise(*o, *ctx->network()->addActivation(*C, nvinfer1::ActivationType::kTANH)->getOutput(0),
                       nvinfer1::ElementWiseOperation::kPROD)
      ->getOutput(0);

  // Recurrent backedge input for hidden and cell.
  cell_mid->setInput(1, *C);
  hidden_mid->setInput(1, *H);
  // outputs
  nvinfer1::LoopOutput output_mode = is_backward ? nvinfer1::LoopOutput::kREVERSE : nvinfer1::LoopOutput::kCONCATENATE;
  nvinfer1::ILoopOutputLayer *output_layer = sequence_loop->addLoopOutput(*H, output_mode);
  output_layer->setInput(1, *lstm_weights.max_seq_size_);
  *hidden_out = Reshape(
    ctx, sequence_loop->addLoopOutput(*hidden_mid->getOutput(0), nvinfer1::LoopOutput::kLAST_VALUE)->getOutput(0),
    nvinfer1::Dims3(1, 1, params_.hidden_size_));
  *cell_out =
    Reshape(ctx, sequence_loop->addLoopOutput(*cell_mid->getOutput(0), nvinfer1::LoopOutput::kLAST_VALUE)->getOutput(0),
            nvinfer1::Dims3(1, 1, params_.hidden_size_));
  return Reshape(ctx, output_layer->getOutput(0), nvinfer1::Dims4(params_.sequence_size_, 1, 1, params_.hidden_size_));
#else
  MS_LOG(ERROR) << "low TensorRT version don't support LSTM op, please upgrade TensorRT version to 7 or higher";
  return nullptr;
#endif
}

int LSTMTensorRT::Prepare(void **network_tensor_bindings, nvinfer1::ICudaEngine *engine) {
  if (in_tensors_.size() == INPUT_SIZE4) {
    return RET_OK;
  }
  if (in_tensors_[HIDDEN_IN_TENSOR_INIT].Data() == nullptr && in_tensors_[CELL_IN_TENSOR_INIT].Data() == nullptr) {
    return RET_OK;
  }

  if (op_binding_tensor_.size() == 0) {
    MS_LOG(DEBUG) << "using serialized engine, add input tensor for " << op_name_;
    TensorInfo &hidden_in_init = in_tensors_[HIDDEN_IN_TENSOR_INIT];
    TensorInfo &cell_in_init = in_tensors_[CELL_IN_TENSOR_INIT];

    hidden_init_name_ = hidden_in_init.Name() + "_hidden_init";
    cell_init_name_ = cell_in_init.Name() + "_cell_init";
    op_binding_tensor_.push_back(
      BindingHelper{hidden_init_name_, hidden_in_init.Data(), nvinfer1::DataType::kFLOAT, hidden_in_init.DataSize()});
    op_binding_tensor_.push_back(
      BindingHelper{cell_init_name_, cell_in_init.Data(), nvinfer1::DataType::kFLOAT, cell_in_init.DataSize()});
    params_.sequence_size_ = in_tensors_[0].Shape()[0];
    op_binding_tensor_.push_back(
      BindingHelper{(op_name_ + "_seq_input"), &params_.sequence_size_, nvinfer1::DataType::kINT32, sizeof(int)});
  }
  for (auto tensor : op_binding_tensor_) {
    auto device_ptr = runtime_->GetAllocator()->MallocDeviceMem(tensor.name_, tensor.size_, tensor.data_type_);
    if (device_ptr == nullptr) {
      MS_LOG(ERROR) << "malloc for inputs tensor device memory failed " << tensor.name_;
      return RET_ERROR;
    }
    int index = engine->getBindingIndex(tensor.name_.c_str());
    network_tensor_bindings[index] = device_ptr;
    runtime_->GetAllocator()->SyncMemInHostAndDevice(const_cast<void *>(tensor.data_), tensor.name_, tensor.size_,
                                                     true);
    runtime_->GetAllocator()->MarkMemValid(tensor.name_, true);
  }
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(ops::kNameLSTM, LSTMTensorRT)
}  // namespace mindspore::lite
