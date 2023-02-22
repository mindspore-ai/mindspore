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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_LSTM_TENSORRT_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_LSTM_TENSORRT_H_
#include <string>
#include <vector>
#include <array>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"

namespace mindspore::lite {
constexpr int INPUT_TENSOR_SIZE = 6;
constexpr int OUTPUT_TENSOR_SIZE = 3;
constexpr int INPUT_WEIGHT = 1;
constexpr int STATE_WEIGHT = 2;
constexpr int BIAS = 3;
constexpr int HIDDEN_IN_TENSOR_INIT = 4;
constexpr int CELL_IN_TENSOR_INIT = 5;
constexpr int LSTM_GATE_NUM = 4;
constexpr int BIDIRECTIONAL = 2;
constexpr int OUTPUT_HIDDEN_INDEX = 1;
constexpr int OUTPUT_CELL_INDEX = 2;
constexpr int INPUT_SIZE_INDEX = 2;
constexpr int FORGET_GATE = 2;
constexpr int CELL_GATE = 3;
constexpr int BATCH_SIZE_INDEX = 2;
static const std::array<int, 4> INDICES{0, 1, 2, 3};

struct LSTMParams {
  int sequence_size_;
  int input_data_size_;
  int batch_size_;
  int layer_count_;
  int hidden_size_;
  nvinfer1::DataType data_type_;
  int directional_cnt_;
};

struct LstmState {
  nvinfer1::ITensor *data_{nullptr};
  nvinfer1::ITensor *hidden_{nullptr};
  nvinfer1::ITensor *cell_{nullptr};
};

struct LstmWeights {
  nvinfer1::ITensor *input_weights_{nullptr};
  nvinfer1::ITensor *state_weights_{nullptr};
  nvinfer1::ITensor *input_bias_{nullptr};
  nvinfer1::ITensor *state_bias_{nullptr};
  nvinfer1::ITensor *max_seq_size_{nullptr};
};

class LSTMTensorRT : public TensorRTOp {
 public:
  LSTMTensorRT(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
               const std::vector<TensorInfo> &out_tensors, std::string name)
      : TensorRTOp(base_operator, in_tensors, out_tensors, name) {}

  ~LSTMTensorRT() override = default;

  int AddInnerOp(TensorRTContext *ctx) override;

  bool IsWeightInputHanledInner() const override { return false; }

  int IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                const std::vector<TensorInfo> &out_tensors) override;

  int Prepare(void **network_tensor_bindings, nvinfer1::ICudaEngine *engine) override;

 private:
  int PreProcess(TensorRTContext *ctx);
  int RunLSTMPlugin(TensorRTContext *ctx);

  int AddLSTMLayers(TensorRTContext *ctx);

  nvinfer1::ITensor *AddLSTMCell(TensorRTContext *ctx, const LstmState *layer_input_states,
                                 const LstmWeights *layer_weights, LstmState *next_state);

  nvinfer1::ITensor *Reshape(TensorRTContext *ctx, nvinfer1::ITensor *tensor, nvinfer1::Dims dims);

  nvinfer1::ITensor *ConcateAll(TensorRTContext *ctx, std::vector<nvinfer1::ITensor *> all_tensort, int axis = 0);

  nvinfer1::ITensor *AddLSTMCalculation(TensorRTContext *ctx, const LstmState &input_state,
                                        const LstmWeights &lstm_weights, nvinfer1::ITensor **hidden_out,
                                        nvinfer1::ITensor **cell_out, bool is_backward = false);
  nvinfer1::ITensor *AddLSTMOneLoop(TensorRTContext *ctx, const LstmState &input_state, const LstmWeights &lstm_weights,
                                    nvinfer1::ITensor **hidden_out, nvinfer1::ITensor **cell_out,
                                    bool is_backward = false);

  int ParseLSTMCellInputs(TensorRTContext *ctx, int layer_index, nvinfer1::ITensor *hidden_init,
                          nvinfer1::ITensor *cell_init, LstmState *input_state, int *input_weight_offset,
                          int *state_weight_offset, int *bias_offset, LstmWeights *lstm_weights,
                          const LstmState &next_state);

  nvinfer1::ITensor *input_data_{nullptr};
  nvinfer1::ITensor *sequence_size_input_{nullptr};
  nvinfer1::ITensor *op_data_out_{nullptr};
  nvinfer1::ITensor *op_hidden_out_{nullptr};
  nvinfer1::ITensor *op_cell_out_{nullptr};
  LSTMParams params_;
  std::string hidden_init_name_;
  std::string cell_init_name_;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_LSTM_TENSORRT_H_
