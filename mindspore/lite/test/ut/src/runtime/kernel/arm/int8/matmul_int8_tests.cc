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
#include "mindspore/lite/src/runtime/kernel/arm/int8/matmul_int8.h"
#include "mindspore/lite/src/runtime/kernel/arm/nnacl/quantization/quantize.h"
#include "mindspore/lite/src/runtime/kernel/arm/nnacl/common_func.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/src/lite_kernel.h"

namespace mindspore {
class TestMatmulInt8 : public mindspore::CommonTest {
 public:
  TestMatmulInt8() {}
};

int MMInt8TestInit(std::vector<lite::tensor::Tensor *> *inputs_, std::vector<lite::tensor::Tensor *> *outputs_,
                   MatMulParameter *matmal_param, float **correct, double *scale, int *zeropoint) {
  float input_max = 20;
  float input_min = -20;
  float weight_max = 1;
  float weight_min = -1;
  float output_max = 30;
  float output_min = -30;

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

  auto in_t =
    new lite::tensor::Tensor(kNumberTypeInt8, {1, 2, 8}, schema::Format_NHWC, static_cast<schema::NodeType>(1));
  in_t->MallocData();
  float in[] = {6.583835634764597,   11.337275140963907,  -4.125256949459629, 10.994337291530833,
                19.086065139532636,  3.620842999158455,   13.167624585590346, -18.326739299407755,
                14.877693740734841,  -17.092677920571653, 19.24147072807235,  -15.14805323833401,
                -18.075654829688737, -0.9164404591894204, -3.836646280336332, -10.870298671273918};
  Quantize(in, in_t->ElementsNum(), input_scale, input_zp, reinterpret_cast<int8_t *>(in_t->Data()));
  auto in_quant_arg = new mindspore::lite::tensor::QuantArg();
  in_quant_arg->zeroPoint = input_zp;
  in_quant_arg->scale = input_scale;
  in_t->AddQuantParam(*in_quant_arg);
  inputs_->push_back(in_t);

  auto weight_t =
    new lite::tensor::Tensor(kNumberTypeInt8, {1, 3, 8}, schema::Format_NHWC, static_cast<schema::NodeType>(1));
  weight_t->MallocData();
  float weight[] = {0.3651070698591563,    -0.5856943921727129,  -0.7472032663840145,  0.9489992871641959,
                    -0.8179490270358738,   -0.873058811259344,   0.39876672713807215,  -0.1816769383004213,
                    -0.13584645926733696,  -0.7614673836659709,  -0.2535825872616164,  -0.05265760030895916,
                    0.28558728305658754,   0.15404213943520118,  -0.1634824450738006,  -0.5068199082730189,
                    -0.026961256849111326, -0.1508441942453307,  0.9375335677537737,   0.3304690744194263,
                    -0.5091563780251127,   0.029887336278646925, -0.39540496207319276, 0.46094065001445084};
  Quantize(weight, weight_t->ElementsNum(), weight_scale, weight_zp, reinterpret_cast<int8_t *>(weight_t->Data()));
  auto weight_quant_arg = new mindspore::lite::tensor::QuantArg();
  weight_quant_arg->zeroPoint = weight_zp;
  weight_quant_arg->scale = weight_scale;
  weight_t->AddQuantParam(*weight_quant_arg);
  inputs_->push_back(weight_t);

  auto out_t =
    new lite::tensor::Tensor(kNumberTypeInt8, {1, 2, 3}, schema::Format_NHWC, static_cast<schema::NodeType>(1));
  out_t->MallocData();
  auto output_quant_arg = new mindspore::lite::tensor::QuantArg();
  output_quant_arg->zeroPoint = output_zp;
  output_quant_arg->scale = output_scale;
  out_t->AddQuantParam(*output_quant_arg);
  outputs_->push_back(out_t);

  *correct = reinterpret_cast<float *>(malloc(out_t->ElementsNum() * sizeof(float)));
  float nchw_co[] = {-0.912632942, 4.08398056, -25.385608673, 2.720281124, 7.745952606, 20.893184662};
  memcpy(*correct, nchw_co, out_t->ElementsNum() * sizeof(float));

  matmal_param->b_transpose_ = true;
  matmal_param->a_transpose_ = false;
  matmal_param->has_bias_ = false;
  return out_t->ElementsNum();
}

TEST_F(TestMatmulInt8, mmint8) {
  std::vector<lite::tensor::Tensor *> inputs_;
  std::vector<lite::tensor::Tensor *> outputs_;
  auto matmul_param = new MatMulParameter();
  float *correct;
  double output_scale;
  int output_zp;
  int total_size = MMInt8TestInit(&inputs_, &outputs_, matmul_param, &correct, &output_scale, &output_zp);
  auto ctx = new lite::Context;
  ctx->thread_num_ = 2;
  kernel::MatmulInt8CPUKernel *mm =
    new kernel::MatmulInt8CPUKernel(reinterpret_cast<OpParameter *>(matmul_param), inputs_, outputs_, ctx, nullptr);

  mm->Init();
  mm->Run();
  float fout[6] = {0};
  Dequantize(reinterpret_cast<int8_t *>(outputs_[0]->Data()), outputs_[0]->ElementsNum(), output_scale, output_zp,
             fout);
  CompareOutputData(fout, correct, 6, 0.3);
  delete matmul_param;
  delete mm;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
  free(correct);
}

}  // namespace mindspore
