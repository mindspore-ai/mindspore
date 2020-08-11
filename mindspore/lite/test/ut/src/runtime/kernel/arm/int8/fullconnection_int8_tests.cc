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
#include "utils/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/src/runtime/kernel/arm/int8/fullconnection_int8.h"
#include "mindspore/lite/src/runtime/kernel/arm/nnacl/common_func.h"
#include "mindspore/lite/src/runtime/kernel/arm/nnacl/quantization/quantize.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/src/lite_kernel.h"

namespace mindspore {
using lite::tensor::Tensor;
class TestFcInt8 : public mindspore::CommonTest {
 public:
  TestFcInt8() {}
};

int FcInt8TestInit(std::vector<lite::tensor::Tensor *> *inputs_, std::vector<lite::tensor::Tensor *> *outputs_,
                   MatMulParameter *matmal_param, float **correct, double *scale, int *zeropoint) {
  float input_max = 20;
  float input_min = -20;
  float weight_max = 1;
  float weight_min = -1;
  float output_max = 20;
  float output_min = -20;

  double input_scale =
    (input_max - input_min) / (std::numeric_limits<int8_t>::max() - std::numeric_limits<int8_t>::min());
  int input_zp = std::numeric_limits<int8_t>::max() - input_max / input_scale;
  double weight_scale =
    (weight_max - weight_min) / (std::numeric_limits<int8_t>::max() - std::numeric_limits<int8_t>::min());
  int weight_zp = std::numeric_limits<int8_t>::max() - weight_max / weight_scale;
  double output_scale =
    (output_max - output_min) / (std::numeric_limits<int8_t>::max() - std::numeric_limits<int8_t>::min());
  int output_zp = std::numeric_limits<int8_t>::max() - output_max / output_scale;
  *scale = output_scale;
  *zeropoint = output_zp;

  Tensor *in_t = new Tensor(kNumberTypeInt8, {2, 2, 2, 2}, schema::Format_NHWC, static_cast<schema::NodeType>(1));
  in_t->MallocData();
  float in[] = {-3.2366564, -4.7733846, -7.8329225, 16.146885, 5.060793,  -6.1471,  -1.7680453, -6.5721383,
                17.87506,   -5.1192183, 10.742863,  1.4536934, 19.693445, 19.45783, 5.063163,   0.5234792};
  Quantize(in, in_t->ElementsNum(), input_scale, input_zp, reinterpret_cast<int8_t *>(in_t->Data()));
  auto in_quant_arg = new mindspore::lite::tensor::QuantArg();
  in_quant_arg->zeroPoint = input_zp;
  in_quant_arg->scale = input_scale;
  in_t->AddQuantParam(*in_quant_arg);
  inputs_->push_back(in_t);

  Tensor *weight_t = new Tensor(kNumberTypeInt8, {3, 8}, schema::Format_NHWC, static_cast<schema::NodeType>(1));
  weight_t->MallocData();
  float weight[] = {-0.24438887,  0.06738146,  -0.8169129,   0.21510671,   -0.012470592, -0.053063435,
                    0.6050155,    0.8656233,   0.12911413,   -0.028635843, -0.034080597, -0.10622552,
                    -0.012254699, -0.01312836, 0.25241964,   -0.4706142,   0.2451482,    -0.9558459,
                    0.4481974,    0.33251503,  -0.011705584, -0.1720293,   -0.39410214,  -0.73637343};
  Quantize(weight, weight_t->ElementsNum(), weight_scale, weight_zp, reinterpret_cast<int8_t *>(weight_t->Data()));
  auto weight_quant_arg = new mindspore::lite::tensor::QuantArg();
  weight_quant_arg->zeroPoint = weight_zp;
  weight_quant_arg->scale = weight_scale;
  weight_t->AddQuantParam(*weight_quant_arg);
  inputs_->push_back(weight_t);

  Tensor *bias_t = new Tensor(kNumberTypeInt32, {3}, schema::Format_NHWC, static_cast<schema::NodeType>(1));
  bias_t->MallocData();
  memset(bias_t->Data(), 0, sizeof(int) * bias_t->ElementsNum());
  inputs_->push_back(bias_t);

  Tensor *out_t = new Tensor(kNumberTypeInt8, {2, 3}, schema::Format_NHWC, static_cast<schema::NodeType>(1));
  out_t->MallocData();
  auto output_quant_arg = new mindspore::lite::tensor::QuantArg();
  output_quant_arg->zeroPoint = output_zp;
  output_quant_arg->scale = output_scale;
  out_t->AddQuantParam(*output_quant_arg);
  outputs_->push_back(out_t);

  *correct = reinterpret_cast<float *>(malloc(out_t->ElementsNum() * sizeof(float)));
  float nchw_co[] = {3.84586822, 0.93586633, 12.16212629, -10.93835061, 2.46887183, 8.61480108};
  memcpy(*correct, nchw_co, out_t->ElementsNum() * sizeof(float));

  matmal_param->b_transpose_ = true;
  matmal_param->a_transpose_ = false;
  matmal_param->has_bias_ = true;
  matmal_param->act_type_ = ActType_No;
  return out_t->ElementsNum();
}

TEST_F(TestFcInt8, fcint8) {
  std::vector<lite::tensor::Tensor *> inputs_;
  std::vector<lite::tensor::Tensor *> outputs_;
  auto matmul_param = new MatMulParameter();
  float *correct;
  double output_scale;
  int output_zp;
  int total_size = FcInt8TestInit(&inputs_, &outputs_, matmul_param, &correct, &output_scale, &output_zp);
  lite::Context *ctx = new lite::Context;
  ctx->thread_num_ = 2;
  kernel::FullconnectionInt8CPUKernel *fc = new kernel::FullconnectionInt8CPUKernel(
    reinterpret_cast<OpParameter *>(matmul_param), inputs_, outputs_, ctx, nullptr);

  fc->Init();
  fc->Run();
  float fout[6] = {0};
  Dequantize(reinterpret_cast<int8_t *>(outputs_[0]->Data()), outputs_[0]->ElementsNum(), output_scale, output_zp,
             fout);
  CompareOutputData(fout, correct, 6, 0.2);
  delete matmul_param;
  delete fc;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
  free(correct);
}

}  // namespace mindspore
