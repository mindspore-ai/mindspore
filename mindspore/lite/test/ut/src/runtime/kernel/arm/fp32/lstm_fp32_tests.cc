/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <iostream>
#include <memory>
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/nnacl/fp32/lstm_fp32.h"
#include "mindspore/lite/src/kernel_registry.h"

namespace mindspore {
class LstmFp32 : public mindspore::CommonTest {
 public:
  LstmFp32() = default;
};

void InitLstmParam(LstmParameter *lstm_param) {
  lstm_param->seq_len_ = 4;
  lstm_param->batch_ = 1;
  lstm_param->input_size_ = 2;
  lstm_param->hidden_size_ = 3;
  lstm_param->bidirectional_ = false;
}

void InitLstmForwardCreator(std::vector<lite::Tensor *> *inputs, std::vector<lite::Tensor *> *outputs,
                            const LstmParameter *lstm_param) {
  // prepare input
  std::vector<float> input_data = {1.3889, -0.3006, -0.1787, 2.1504, -0.3181, 0.4945, -0.4758, -0.8187};
  auto *input = new lite::Tensor;
  input->set_data_type(kNumberTypeFloat32);
  input->set_shape({lstm_param->seq_len_, lstm_param->batch_, lstm_param->input_size_});
  input->MallocData();
  memcpy(input->MutableData(), input_data.data(), input_data.size() * sizeof(float));

  // prepare weight_i
  std::vector<float> weight_i_data = {0.21368974,  -0.3778776,  0.05025542, 0.09011161,  0.18355745,  0.5491228,
                                      -0.14186832, -0.4655916,  0.49541366, -0.44039622, 0.5625571,   0.23325664,
                                      0.3449825,   -0.42750397, 0.01911497, -0.4125802,  -0.56690466, 0.50593233,
                                      -0.29129684, -0.27841482, 0.01964372, -0.42543447, 0.41720617,  -0.30054367};
  auto *weight_i = new lite::Tensor;
  weight_i->set_data_type(kNumberTypeFloat32);
  weight_i->set_format(schema::Format_NHWC);
  weight_i->set_shape({1, lstm_param->hidden_size_ * 4, lstm_param->input_size_});
  weight_i->MallocData();
  memcpy(weight_i->MutableData(), weight_i_data.data(), weight_i_data.size() * sizeof(float));

  // prepare weight_r
  std::vector<float> weight_h_data = {
    -0.03424168, 0.00643545,  0.36867607, -0.08598137, 0.19804275,  -0.11319417, -0.0244593,  -0.16440144, -0.07268238,
    0.09828371,  0.33358777,  0.53381383, -0.39431244, -0.06005383, -0.3520246,  0.42687547,  0.5772828,   0.5380008,
    -0.16130409, -0.24737108, 0.42409766, -0.50648475, 0.48223662,  -0.5221103,  -0.49216837, -0.29084128, 0.3408438,
    0.34080023,  0.49467337,  0.23473483, 0.01759732,  0.04691631,  0.45574808,  -0.29481018, 0.29442167,  -0.36718};
  auto *weight_h = new lite::Tensor;
  weight_h->set_data_type(kNumberTypeFloat32);
  weight_h->set_format(schema::Format_NHWC);
  weight_h->set_shape({1, lstm_param->hidden_size_ * 4, lstm_param->hidden_size_});
  weight_h->MallocData();
  memcpy(weight_h->MutableData(), weight_h_data.data(), weight_h_data.size() * sizeof(float));

  // prepare bias
  std::vector<float> bias_data = {-0.00207639, 0.16391152,  -0.00069344, -0.32945693, -0.367423,   0.28301108,
                                  -0.17930457, 0.5278388,   0.12598747,  -0.53130764, 0.1479364,   0.16695255,
                                  -0.00708795, -0.46417096, -0.23966661, -0.17496741, -0.19166365, -0.50466555,
                                  -0.23593256, -0.3911457,  0.51128435,  0.5128727,   0.253451,    -0.51891875};
  auto *bias = new lite::Tensor;
  bias->set_data_type(kNumberTypeFloat32);
  bias->set_format(schema::Format_NHWC);
  bias->set_shape({1, lstm_param->hidden_size_ * 4 * 2});
  bias->MallocData();
  memcpy(bias->MutableData(), bias_data.data(), bias_data.size() * sizeof(float));

  // prepare state
  std::vector<float> state_data = {0, 0, 0};
  auto *state = new lite::Tensor;
  state->set_data_type(kNumberTypeFloat32);
  state->set_format(schema::Format_NHWC);
  state->set_shape({1, lstm_param->batch_, lstm_param->hidden_size_});
  state->MallocData();
  memcpy(state->MutableData(), state_data.data(), state_data.size() * sizeof(float));

  inputs->push_back(input);
  inputs->push_back(weight_i);
  inputs->push_back(weight_h);
  inputs->push_back(bias);
  inputs->push_back(state);
  inputs->push_back(state);

  // malloc output buffer, for arm cpu, format: N C4 H W 4
  auto *output = new lite::Tensor;
  output->set_data_type(kNumberTypeFloat32);
  output->set_shape({lstm_param->seq_len_, lstm_param->batch_, lstm_param->hidden_size_});
  output->set_format(schema::Format_NHWC);
  output->MallocData();
  memset(output->MutableData(), 0, output->ElementsNum() * sizeof(float));

  auto *cell_state = new lite::Tensor;
  cell_state->set_data_type(kNumberTypeFloat32);
  cell_state->set_shape({1, lstm_param->batch_, lstm_param->hidden_size_});
  cell_state->set_format(schema::Format_NHWC);
  cell_state->MallocData();
  memset(cell_state->MutableData(), 0, cell_state->ElementsNum() * sizeof(float));

  auto *hidden_state = new lite::Tensor;
  hidden_state->set_data_type(kNumberTypeFloat32);
  hidden_state->set_shape({1, lstm_param->batch_, lstm_param->hidden_size_});
  hidden_state->set_format(schema::Format_NHWC);
  hidden_state->MallocData();
  memset(hidden_state->MutableData(), 0, hidden_state->ElementsNum() * sizeof(float));

  outputs->push_back(output);
  outputs->push_back(cell_state);
  outputs->push_back(hidden_state);
}

void CompareResult(lite::Tensor *output, std::vector<float> data) {
  for (int i = 0; i < output->ElementsNum(); i++) {
    std::cout << reinterpret_cast<float *>(output->MutableData())[i] << ", ";
  }
  std::cout << std::endl;

  CommonTest::CompareOutputData(reinterpret_cast<float *>(output->MutableData()), data.data(), output->ElementsNum(),
                                0.0001);
}

TEST_F(LstmFp32, LstmForwardFp32Accuracy) {
  // prepare stage
  auto lstm_param = new LstmParameter();
  InitLstmParam(lstm_param);

  // init ctx
  auto ctx = new lite::InnerContext();
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  // init tensor
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  InitLstmForwardCreator(&inputs, &outputs, lstm_param);

  // register op
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, mindspore::schema::PrimitiveType_LSTM};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(lstm_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  // op run
  kernel->Run();

  std::cout << "==================output data=================" << std::endl;
  std::vector<float> output0_data = {-0.0702, 0.1225,  0.0876,  -0.0357, -0.0227, -0.2294,
                                     -0.0345, -0.0108, -0.2002, 0.0451,  0.0853,  -0.1205};
  CompareResult(outputs[0], output0_data);

  std::vector<float> output1_data = {0.0451, 0.0853, -0.1205};
  CompareResult(outputs[1], output1_data);

  std::vector<float> output2_data = {0.0989, 0.2094, -0.4132};
  CompareResult(outputs[2], output2_data);

  delete lstm_param;
  for (unsigned int i = 0; i < inputs.size() - 1; i++) {
    delete inputs[i];
  }
  for (auto &output : outputs) {
    delete output;
  }
  delete kernel;
  MS_LOG(INFO) << "LstmFp32 forward accuracy passed";
}

void InitLstmBackwardCreator(std::vector<lite::Tensor *> *inputs, std::vector<lite::Tensor *> *outputs,
                             const LstmParameter *lstm_param) {
  // prepare input
  std::vector<float> input_data = {1.4305, 0.5342, -0.9221, 0.0527, 2.3770, -0.3697, -0.2833, -2.1285};
  auto *input = new lite::Tensor;
  input->set_data_type(kNumberTypeFloat32);
  input->set_shape({lstm_param->seq_len_, lstm_param->batch_, lstm_param->input_size_});
  input->MallocData();
  memcpy(input->MutableData(), input_data.data(), input_data.size() * sizeof(float));

  // prepare weight_i
  std::vector<float> weight_i_data = {
    -0.19253477, -0.007966279, -0.06039094, 0.27697134,  -0.5071223,  0.18996351,  0.20472168,  -0.1007814,
    0.04282999,  0.20836472,   -0.4654655,  0.050321221, -0.3431457,  0.22256428,  0.29294532,  0.45042896,
    0.20468240,  0.13078391,   -0.20987969, -0.3173505,  -0.3813517,  0.10205835,  0.21858131,  -0.0386473,
    0.5512280,   -0.2763766,   -0.3593936,  -0.5181975,  0.3469863,   -0.38533931, 0.010202527, -0.46598294,
    -0.5740513,  0.06127524,   -0.03960543, 0.2478809,   -0.17296993, 0.19159525,  -0.4976995,  0.05985528,
    0.3653409,   0.386924,     0.3170289,   -0.08830952, -0.31105759, 0.3110240,   0.15174299,  0.287579894};
  auto *weight_i = new lite::Tensor;
  weight_i->set_data_type(kNumberTypeFloat32);
  weight_i->set_format(schema::Format_NHWC);
  weight_i->set_shape({2, lstm_param->hidden_size_ * 4, lstm_param->input_size_});
  weight_i->MallocData();
  memcpy(weight_i->MutableData(), weight_i_data.data(), weight_i_data.size() * sizeof(float));

  // prepare weight_r
  std::vector<float> weight_h_data = {
    0.106934666,  -0.50430017,  0.33296257,   -0.288117021, -0.38019785,  -0.147071093, 0.422707557,  0.41497004,
    -0.5329730,   -0.430150926, -0.032713949, 0.35401260,   0.179495036,  -0.14158579,  0.380428612,  -0.175597071,
    0.54088723,   -0.403292059, -0.287720531, -0.51250511,  -0.15405902,  -0.440592586, 0.16726928,   -0.0163397789,
    0.51673841,   0.5094323,    -0.137105107, -0.181070089, -0.47221425,  -0.38046866,  -0.206725060, 0.248537719,
    -0.23961094,  -0.117781728, 0.426800847,  0.0266208052, -0.197408229, 0.54831492,   -0.280048757, -0.125062286,
    -0.29929456,  0.42354834,   -0.401066303, 0.356340110,  0.54629492,   -0.15852552,  0.131406366,  -0.101815432,
    0.0121276974, -0.53553336,  0.121099889,  0.060554087,  0.46259057,   -0.49666053,  0.090806663,  0.20542401,
    -0.38674920,  -0.23874849,  -0.5222138,   0.57537007,   0.113343358,  -0.35233467,  -0.25532332,  0.159506142,
    0.35996592,   -0.201961308, -0.16323345,  0.119177639,  -0.12677872,  -0.175229549, -0.160024613, -0.21058899};
  auto *weight_h = new lite::Tensor;
  weight_h->set_data_type(kNumberTypeFloat32);
  weight_h->set_format(schema::Format_NHWC);
  weight_h->set_shape({2, lstm_param->hidden_size_ * 4, lstm_param->hidden_size_});
  weight_h->MallocData();
  memcpy(weight_h->MutableData(), weight_h_data.data(), weight_h_data.size() * sizeof(float));

  // prepare bias
  std::vector<float> bias_data = {
    0.57061123,   -0.25357073,  -0.146834075, 0.412972748,  -0.27809411,  -0.0542128682, -0.45384609,  -0.53261917,
    0.222133636,  -0.18093895,  -0.045559883, 0.09109061,   0.080319643,  0.455167174,   0.36235427,   -0.00164419412,
    -0.135566502, 0.41905909,   -0.450117409, 0.50565385,   -0.077815443, -0.47051778,   -0.141349375, -0.338519752,
    0.48683023,   0.282384872,  0.13399660,   -0.382526844, -0.23370727,  -0.184681564,  0.45679104,   -0.339453905,
    0.452010273,  0.0552094578, 0.328843057,  0.127738714,  -0.127084732, -0.334061294,  -0.46742400,  -0.401568055,
    0.23712641,   -0.052937567, 0.272351622,  0.42767739,   0.303884744,  -0.46025499,   -0.43985402,  0.256422877};
  auto *bias = new lite::Tensor;
  bias->set_data_type(kNumberTypeFloat32);
  bias->set_format(schema::Format_NHWC);
  bias->set_shape({2, lstm_param->hidden_size_ * 4 * 2});
  bias->MallocData();
  memcpy(bias->MutableData(), bias_data.data(), bias_data.size() * sizeof(float));

  // prepare state
  std::vector<float> state_data = {0, 0, 0, 0, 0, 0};
  auto *state = new lite::Tensor;
  state->set_data_type(kNumberTypeFloat32);
  state->set_format(schema::Format_NHWC);
  state->set_shape({2, lstm_param->batch_, lstm_param->hidden_size_});
  state->MallocData();
  memcpy(state->MutableData(), state_data.data(), state_data.size() * sizeof(float));

  inputs->push_back(input);
  inputs->push_back(weight_i);
  inputs->push_back(weight_h);
  inputs->push_back(bias);
  inputs->push_back(state);
  inputs->push_back(state);

  // malloc output buffer, for arm cpu, format: N C4 H W 4
  auto *output = new lite::Tensor;
  output->set_data_type(kNumberTypeFloat32);
  output->set_shape({lstm_param->seq_len_, 2, lstm_param->batch_, lstm_param->hidden_size_});
  output->set_format(schema::Format_NHWC);
  output->MallocData();
  memset(output->MutableData(), 0, output->ElementsNum() * sizeof(float));

  auto *cell_state = new lite::Tensor;
  cell_state->set_data_type(kNumberTypeFloat32);
  cell_state->set_shape({2, lstm_param->batch_, lstm_param->hidden_size_});
  cell_state->set_format(schema::Format_NHWC);
  cell_state->MallocData();
  memset(cell_state->MutableData(), 0, cell_state->ElementsNum() * sizeof(float));

  auto *hidden_state = new lite::Tensor;
  hidden_state->set_data_type(kNumberTypeFloat32);
  hidden_state->set_shape({2, lstm_param->batch_, lstm_param->hidden_size_});
  hidden_state->set_format(schema::Format_NHWC);
  hidden_state->MallocData();
  memset(hidden_state->MutableData(), 0, hidden_state->ElementsNum() * sizeof(float));

  outputs->push_back(output);
  outputs->push_back(cell_state);
  outputs->push_back(hidden_state);
}

TEST_F(LstmFp32, LstmBackwardFp32Accuracy) {
  // prepare stage
  auto lstm_param = new LstmParameter();
  InitLstmParam(lstm_param);
  lstm_param->bidirectional_ = true;

  // init ctx
  auto ctx = new lite::InnerContext();
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  // init tensor
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  InitLstmBackwardCreator(&inputs, &outputs, lstm_param);

  // register op
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, mindspore::schema::PrimitiveType_LSTM};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(lstm_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  // op run
  kernel->Run();

  std::cout << "==================output data=================" << std::endl;
  std::vector<float> output0_data = {-0.2922, -0.1416, 0.0077,  -0.0422, -0.0585, 0.2061,  -0.2385, -0.0146,
                                     -0.1796, -0.0554, -0.0973, 0.1013,  -0.3062, -0.1516, -0.0310, 0.0459,
                                     -0.0784, 0.0949,  0.0249,  -0.0653, -0.0869, -0.1113, -0.2155, -0.0500};
  CompareResult(outputs[0], output0_data);

  std::vector<float> output1_data = {0.0249, -0.0653, -0.0869, -0.0422, -0.0585, 0.2061};
  CompareResult(outputs[1], output1_data);

  std::vector<float> output2_data = {0.0373, -0.2322, -0.1477, -0.1621, -0.1808, 0.5146};
  CompareResult(outputs[2], output2_data);

  delete lstm_param;
  for (unsigned int i = 0; i < inputs.size() - 1; i++) {
    delete inputs[i];
  }
  for (auto &output : outputs) {
    delete output;
  }
  delete kernel;
  MS_LOG(INFO) << "LstmFp32 backward accuracy passed";
}

}  // namespace mindspore
