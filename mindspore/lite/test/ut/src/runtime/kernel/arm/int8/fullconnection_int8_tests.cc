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
#include "schema/inner/model_generated.h"
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/src/runtime/kernel/arm/int8/fullconnection_int8.h"
#include "mindspore/lite/nnacl/common_func.h"
#include "mindspore/lite/nnacl/int8/quantize.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/src/lite_kernel.h"

namespace mindspore {
using lite::Tensor;
class TestFcInt8 : public mindspore::CommonTest {
 public:
  TestFcInt8() {}
};

struct TensorInfo {
  float *data;
  int *data_int;
  float min;
  float max;
  int len;
  std::vector<int> *shape;
};

extern void QuantProcess(float *input, int len, float min, float max, float *scale, int *zero_point, int8_t *output);
extern lite::Tensor *MakeQuantTensor(int8_t *data, int len, std::vector<int> *shape, float scale, int zp);

lite::Tensor *MakeIntTensor(int *data, int len, std::vector<int> *shape) {
  auto tensor = new lite::Tensor(kNumberTypeInt32, *shape, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  tensor->MallocData();
  auto tensor_ptr = reinterpret_cast<int *>(tensor->MutableData());
  memcpy(tensor_ptr, data, len * sizeof(int));
  return tensor;
}

void FcInt8TestInit(std::vector<lite::Tensor *> *inputs, std::vector<lite::Tensor *> *outputs, TensorInfo *in,
                    TensorInfo *weight, TensorInfo *bias, TensorInfo *out) {
  float in_scale, weight_scale, out_scale;
  int in_zp, weight_zp, out_zp;
  int8_t *in_data = new int8_t[in->len];
  int8_t *weight_data = new int8_t[weight->len];
  QuantProcess(in->data, in->len, in->min, in->max, &in_scale, &in_zp, in_data);
  auto in_tensor = MakeQuantTensor(in_data, in->len, in->shape, in_scale, in_zp);
  inputs->push_back(in_tensor);
  QuantProcess(weight->data, weight->len, weight->min, weight->max, &weight_scale, &weight_zp, weight_data);
  auto weight_tensor = MakeQuantTensor(weight_data, weight->len, weight->shape, weight_scale, weight_zp);
  inputs->push_back(weight_tensor);
  auto bias_tensor = MakeIntTensor(bias->data_int, bias->len, bias->shape);
  inputs->push_back(bias_tensor);
  QuantProcess(out->data, out->len, out->min, out->max, &out_scale, &out_zp, nullptr);
  auto out_tensor = MakeQuantTensor(nullptr, out->len, out->shape, out_scale, out_zp);
  outputs->push_back(out_tensor);
  delete[] in_data;
  delete[] weight_data;
}

TEST_F(TestFcInt8, fctest1) {
  float in[] = {4.259103407444801,   5.992151035772917,   -9.495343223733581,  3.0509999931426215, -16.635707833991095,
                -14.72005749234452,  2.8290916795754093,  -15.827977973039049, -16.98208477063347, 2.8801101778935347,
                -0.5905297521382735, 18.042746010536085,  3.913511213700396,   11.571264917136105, 19.084257392926148,
                8.571560238377568,   17.58868010598305,   12.433311533838427,  4.548078598583526,  15.609650071521138,
                6.663372887795717,   17.581323475674594,  1.453277207446778,   -6.119351424589654, -16.87310296820285,
                11.906066592064796,  -13.290100998834653, 19.627129875430548,  16.034262583959162, 10.255738135902781,
                12.134650347811792,  -5.5882066903433305, 15.554050723026322,  15.288481461776783, 17.651080309797287,
                -9.258779162183215,  4.218532791445092,   -6.205309122668545,  1.2220458021156908, 1.6800736573947326};
  TensorInfo in_params;
  in_params.data = in;
  in_params.len = 40;
  std::vector<int> in_shape{5, 2, 2, 2};
  in_params.shape = &in_shape;
  in_params.min = -20;
  in_params.max = 20;

  float weight[] = {
    -0.586269014312498,   0.10845796767603733,  0.8455159907124523,   0.20261291069007226,  0.7564258582027543,
    0.4505005038790615,   -0.607259232240795,   -0.6962171798923924,  0.7967573009922135,   -0.46069496925353715,
    -0.2967638879316592,  -0.7025557337565955,  -0.5313515272071268,  0.07584168670764102,  -0.6860034691410029,
    0.9218806800279316,   -0.07408538201953907, -0.7933652717840096,  0.6636691558029275,   -0.30198695606477477,
    0.790225747868754,    -0.9478140254555916,  0.4537316306461665,   0.1776848732022871,   -0.7492316745474277,
    -0.5825825240770948,  0.5680842804542614,   -0.9255552309192772,  0.20866577718844725,  0.9570928647172854,
    0.18172570688854406,  -0.26442830241827253, -0.24765169216720873, -0.19512285277145702, 0.1120696020054861,
    0.7558578199370625,   -0.15032457481135109, -0.08485585411928809, 0.6343014796699504,   0.026380085222785787,
    -0.40516674259120444, -0.7407588590646037,  -0.28521396461492454, 0.2555841827858194,   0.023640857478332444,
    -0.6540694390119834,  0.7439705499824205,   -0.7579774562590929};
  TensorInfo weight_params;
  weight_params.data = weight;
  weight_params.len = 48;
  std::vector<int> weight_shape{6, 8};
  weight_params.shape = &weight_shape;
  weight_params.min = -1;
  weight_params.max = 1;

  int bias[6] = {0};
  TensorInfo bias_params;
  bias_params.data_int = bias;
  bias_params.len = 6;
  std::vector<int> bias_shape{6};
  bias_params.shape = &bias_shape;

  float correct[] = {-19.170732, -7.5019627, -13.015462, -27.760283, 4.1447954,  20.660276,  4.0412164,  -33.750015,
                     -4.560128,  7.1035166,  27.976341,  9.75216,    14.383608,  -12.87587,  -24.688887, -12.185722,
                     3.7933283,  -19.266382, 17.193876,  -49.99205,  -15.480089, -3.1659412, 19.470417,  13.758459,
                     4.0713396,  4.614437,   11.296907,  -7.244551,  -11.143417, -21.233654};
  TensorInfo out_params;
  out_params.data = correct;
  out_params.len = 30;
  std::vector<int> out_shape{5, 6};
  out_params.shape = &out_shape;
  out_params.min = -50;
  out_params.max = 50;

  auto fc_param = new MatMulParameter();
  fc_param->a_transpose_ = false;
  fc_param->b_transpose_ = true;
  fc_param->has_bias_ = true;
  fc_param->act_type_ = ActType_No;
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  FcInt8TestInit(&inputs, &outputs, &in_params, &weight_params, &bias_params, &out_params);
  auto ctx = new lite::InnerContext;
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::FullconnectionInt8CPUKernel *fc =
    new kernel::FullconnectionInt8CPUKernel(reinterpret_cast<OpParameter *>(fc_param), inputs, outputs, ctx);

  fc->Init();
  fc->Run();
  float out_scale;
  int out_zp;
  QuantProcess(correct, out_params.len, out_params.min, out_params.max, &out_scale, &out_zp, nullptr);
  float *out = new float[out_params.len];
  Dequantize(reinterpret_cast<int8_t *>(outputs[0]->MutableData()), outputs[0]->ElementsNum(), out_scale, out_zp, out);
  ASSERT_EQ(0, CompareOutputData(out, correct, 6, 0.3));
  delete fc;
  for (auto t : inputs) delete t;
  for (auto t : outputs) delete t;
  delete[] out;
}
}  // namespace mindspore
