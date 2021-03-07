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
#include "mindspore/lite/src/runtime/kernel/arm/int8/matmul_int8.h"
#include "mindspore/lite/nnacl/int8/quantize.h"
#include "nnacl/common_func.h"
#include "nnacl/int8/matmul_int8.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/src/lite_kernel.h"

namespace mindspore {
class TestMatmulInt8 : public mindspore::CommonTest {
 public:
  TestMatmulInt8() {}
};

struct TensorInfo {
  float *data;
  float min;
  float max;
  int len;
  std::vector<int> *shape;
};

void QuantProcess(float *input, int len, float min, float max, float *scale, int *zero_point, int8_t *output) {
  *scale = (max - min) / (std::numeric_limits<int8_t>::max() - std::numeric_limits<int8_t>::min());
  *zero_point = std::numeric_limits<int8_t>::max() - max / (*scale);
  if (output) {
    Quantize(input, len, *scale, *zero_point, output);
  }
}

lite::Tensor *MakeQuantTensor(int8_t *data, int len, std::vector<int> *shape, float scale, int zp) {
  auto tensor = new lite::Tensor(kNumberTypeInt8, *shape, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  tensor->MallocData();
  if (data) {
    auto tensor_ptr = reinterpret_cast<int8_t *>(tensor->MutableData());
    memcpy(tensor_ptr, data, len * sizeof(int8_t));
  }
  auto quant_arg = new mindspore::lite::QuantArg();
  quant_arg->zeroPoint = zp;
  quant_arg->scale = scale;
  tensor->AddQuantParam(*quant_arg);
  return tensor;
}

void MMInt8TestInit(std::vector<lite::Tensor *> *inputs, std::vector<lite::Tensor *> *outputs, TensorInfo *in,
                    TensorInfo *weight, TensorInfo *out) {
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
  QuantProcess(out->data, out->len, out->min, out->max, &out_scale, &out_zp, nullptr);
  auto out_tensor = MakeQuantTensor(nullptr, out->len, out->shape, out_scale, out_zp);
  outputs->push_back(out_tensor);
  delete[] in_data;
  delete[] weight_data;
}

TEST_F(TestMatmulInt8, mmtest1) {
  float in[] = {6.583835634764597,   11.337275140963907,  -4.125256949459629, 10.994337291530833,
                19.086065139532636,  3.620842999158455,   13.167624585590346, -18.326739299407755,
                14.877693740734841,  -17.092677920571653, 19.24147072807235,  -15.14805323833401,
                -18.075654829688737, -0.9164404591894204, -3.836646280336332, -10.870298671273918};
  TensorInfo in_params;
  in_params.data = in;
  in_params.len = 16;
  std::vector<int> in_shape{1, 2, 8};
  in_params.shape = &in_shape;
  in_params.min = -20;
  in_params.max = 20;

  float weight[] = {0.3651070698591563,    -0.5856943921727129,  -0.7472032663840145,  0.9489992871641959,
                    -0.8179490270358738,   -0.873058811259344,   0.39876672713807215,  -0.1816769383004213,
                    -0.13584645926733696,  -0.7614673836659709,  -0.2535825872616164,  -0.05265760030895916,
                    0.28558728305658754,   0.15404213943520118,  -0.1634824450738006,  -0.5068199082730189,
                    -0.026961256849111326, -0.1508441942453307,  0.9375335677537737,   0.3304690744194263,
                    -0.5091563780251127,   0.029887336278646925, -0.39540496207319276, 0.46094065001445084};
  TensorInfo weight_params;
  weight_params.data = weight;
  weight_params.len = 24;
  std::vector<int> weight_shape{1, 3, 8};
  weight_params.shape = &weight_shape;
  weight_params.min = -1;
  weight_params.max = 1;

  float correct[] = {-0.912632942, 4.08398056, -25.385608673, 2.720281124, 7.745952606, 20.893184662};
  TensorInfo out_params;
  out_params.data = correct;
  out_params.len = 6;
  std::vector<int> out_shape{1, 2, 3};
  out_params.shape = &out_shape;
  out_params.min = -30;
  out_params.max = 30;

  auto matmul_param = new MatMulParameter();
  matmul_param->a_transpose_ = false;
  matmul_param->b_transpose_ = true;
  matmul_param->has_bias_ = false;
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  MMInt8TestInit(&inputs, &outputs, &in_params, &weight_params, &out_params);
  auto ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::MatmulInt8CPUKernel *mm =
    new kernel::MatmulInt8CPUKernel(reinterpret_cast<OpParameter *>(matmul_param), inputs, outputs, ctx);

  mm->Init();
  mm->Run();
  float out_scale;
  int out_zp;
  QuantProcess(correct, out_params.len, out_params.min, out_params.max, &out_scale, &out_zp, nullptr);
  float *out = new float[out_params.len];
  Dequantize(reinterpret_cast<int8_t *>(outputs[0]->MutableData()), outputs[0]->ElementsNum(), out_scale, out_zp, out);
  ASSERT_EQ(0, CompareOutputData(out, correct, 6, 0.3));
  delete mm;
  for (auto t : inputs) delete t;
  for (auto t : outputs) delete t;
  delete[] out;
}

TEST_F(TestMatmulInt8, mmtest2) {
  float in[] = {
    -9.302902352910598,  16.65876088354537,    -7.2801759810348265, -6.3246021711950995, 8.467234093555248,
    -4.729482636552028,  -3.747183865378627,   -8.690477390174504,  -2.7419930714530523, -3.9478573566319,
    7.399137633080947,   -1.604450983941291,   0.3115665358682982,  -16.864318496334278, 2.5447052588244112,
    -13.428639671203255, 13.417832391771974,   10.37917002467671,   14.709787234172168,  -16.347969268427146,
    4.652834783979106,   6.03601450738973,     2.5788179666401874,  -9.236801653471375,  -0.18997468903009462,
    19.977363387313744,  15.163337058447325,   -12.602897730843484, -6.178507797555191,  13.457928661476004,
    -10.65587824516124,  -18.715557779424188,  -9.758039647923935,  8.102044210643097,   19.66309736072973,
    -13.368041407254193, 9.928467253978024,    4.9981961360698755,  -4.2838547685981645, 1.5021547181513526,
    -7.043468062239523,  11.964494917194845,   -4.783071964346499,  -17.646518743891008, -7.77810768119101,
    14.869414292570454,  8.333036603520906,    11.053769742928765,  -1.768128637419725,  -14.971400302494597,
    -0.8653626097283293, -6.21101640878031,    14.83875267850518,   7.224097292538833,   -16.747116419664213,
    15.310978507353724,  -0.05593751363976551, 2.066880260042687,   -3.7053788331264137, 9.788933831258937,
    -13.614523856950811, -5.656865231633642,   4.720270075138178,   -8.366650073458409,  7.197187069893303,
    -18.78518907850054,  15.691652153539678,   7.914926057095165,   10.073559408864384,  10.437631177498353,
    -3.0580194164595085, 17.36998905922836,    0.09998119223460122, 19.519199178417452,  -11.121833210377702,
    19.655990774915622,  -17.25682638091008,   11.701013896880006,  -12.746728025401781, -9.370055221833699,
    18.720474512055908,  7.634198897927405,    -15.521885320500694, -9.119267877304358,  -1.5853789671841945,
    4.783147823043613,   14.6732610092525,     -9.294170215010427,  9.835421489234331,   13.051159704193232,
    -1.422599906517025,  -1.5530696181467398,  19.51404609713284,   -12.297429715833763, 6.8811248552401985,
    13.052476234003755,  18.66085390709462,    -8.097735292301103,  -6.868239274661935,  -8.067142805841826,
    3.2707808734101533,  1.8239332220210827};
  TensorInfo in_params;
  in_params.data = in;
  in_params.len = 6 * 17;
  std::vector<int> in_shape{1, 6, 17};
  in_params.shape = &in_shape;
  in_params.min = -20;
  in_params.max = 20;

  float weight[] = {
    -0.42740096214251677,  0.8557068789482212,    0.4560574664172552,   -0.1317821769705021,  0.2845963675712846,
    0.8414603241768241,    0.24513271080109011,   0.16403708196683398,  -0.09111601416189297, -0.714027790956111,
    0.12253431683185845,   -0.4542459426686125,   0.7123202105555202,   -0.3708573394849488,  -0.4571735646072892,
    -0.595627630450934,    -0.5022671357384993,   0.2781065609468565,   -0.07586181451887586, -0.2667701710291306,
    0.03141663091360791,   -0.013304592900917456, -0.7507975439396768,  0.5886778622432618,   -0.9056075431439199,
    0.9393767525356569,    -0.2791312477047512,   0.7134531940450286,   0.3977932134993216,   -0.027832574334469395,
    0.7222024948455503,    -0.2084178952731608,   -0.4869535410639745,  -0.8255185994321805,  0.975443145421772,
    0.541914384763855,     -0.8831162309708303,   -0.3339354888475805,  0.3699271440691516,   -0.26923635397292944,
    -0.4975347179262828,   0.2440013185603882,    0.5553443771246633,   0.6111909921005778,   -0.5968624036034165,
    0.8367593317557596,    -0.843079440282104,    -0.5651924211153698,  0.7169318662247579,   0.5116755837443465,
    -0.9079299375502927,   0.025240632113315176,  -0.5819662075810048,  -0.37278414060319176, -0.172154755034845,
    -0.7372352723583462,   0.2462103743741677,    0.11785417820789856,  0.6712183976911841,   -0.7042964391243491,
    -0.8215958062965967,   -0.7304378130182314,   0.3991295415760667,   -0.07226694075875573, 0.9329628273800614,
    0.7866596674858193,    0.9410341281569592,    0.39672750454198225,  -0.5217505454791054,  0.9538253510722774,
    -0.6286845762774464,   -0.773460418882959,    0.002296000778892804, 0.9763898918063998,   0.9648708739062339,
    0.9400037814137154,    -0.6011085333221611,   -0.5890262409238565,  -0.8078857772627164,  0.233661306598278,
    -0.6726381934018617,   -0.08533323149874539,  0.19055766469859425,  -0.7956482347958518,  -0.17012651641579035,
    0.7181052528631318,    0.1285045774388125,    -0.6997527417326721,  -0.8436484573035989,  0.342855467305474,
    0.4085157503460306,    -0.6199324510955382,   -0.6883822276097309,  0.4186437018431113,   0.3030114883148305,
    0.0948227655828271,    -0.002521771948760465, -0.34878560791422397, 0.08513437045281003,  0.3116035319055901,
    -0.7177514192203747,   0.050531673446029046,  -0.7399803440665007,  -0.9353609485885221,  -0.3899340891814298,
    0.40867084031625356,   -0.17462484099335662,  -0.6313167634279941,  -0.8135597146296727,  -0.9762553414099975,
    -0.1040485487920626,   -0.6517520252975368,   0.5877412140956126,   0.9433584450325512,   0.24701546283170672,
    -0.3236849444311023,   -0.12043548611719657,  0.5300129281052712,   -0.1380138229226111,  -0.8787455295545508,
    -0.4361728423289617,   0.7331994894985936,    0.45492774136929826,  -0.17836517403432972, 0.10896668585054625,
    0.6176507847785211,    0.21617962964770676,   -0.6821928873814629,  0.021775035324277825, 0.15089571088539566,
    -0.9923383126255942,   -0.6034706970202426,   0.17729888871670285,  0.1278810065499425,   -0.6575545415840387,
    -0.022704865415375197, -0.7366071817901978,   -0.9300211224192332,  -0.153494127035938,   0.4836121912045357,
    -0.3318483587414114,   -0.9658468087620375,   0.8388464445207262,   0.45745949405796127,  -0.3671803281863002,
    -0.1543498074773253,   0.18955899788963748,   -0.4452120359256351,  -0.5338599486040962,  -0.06979561022721281,
    -0.45964195574917355,  -0.4343754114042866,   -0.4318308749403197,  0.748107130947133,    -0.4703901010752156,
    0.6655596561650823,    0.9075215202451821,    0.2708741258104177,   -0.6540233471632313,  0.7250124906689572,
    0.6674821078610087,    0.8464696566759315,    -0.6106156844283976,  0.8675828337337224,   0.8517737949695063,
    -0.8126381016475459,   -0.6140987457462099,   -0.2984524227549874,  0.2816320572339577,   -0.8131479383469931};
  TensorInfo weight_params;
  weight_params.data = weight;
  weight_params.len = 170;
  std::vector<int> weight_shape{1, 17, 10};
  weight_params.shape = &weight_shape;
  weight_params.min = -1;
  weight_params.max = 1;

  float correct[] = {35.815605,  26.532362,  14.777507,  -12.651591, -2.0373726, -47.020798,  -18.53121,  2.7848654,
                     16.19751,   -30.754261, 25.830605,  47.635204,  10.247462,  -33.260662,  34.145412,  -6.1611304,
                     -18.56802,  -24.669813, 20.314533,  -5.887198,  -14.757037, 24.78901,    20.512205,  17.985718,
                     17.62954,   20.365099,  -26.223736, 0.99702793, 12.752281,  -35.30419,   -22.09603,  8.2218,
                     8.120908,   27.685753,  -44.010464, -1.879332,  -4.531702,  21.434296,   4.2146144,  22.721859,
                     7.485317,   20.148363,  -15.49375,  -4.5062046, 37.77292,   -0.23385821, -45.532917, -21.055403,
                     46.854183,  -13.595161, 2.8823144,  -23.905682, 2.3569264,  26.975227,   32.806625,  9.185071,
                     -39.330578, -1.0041192, -6.8353715, -33.2658};
  TensorInfo out_params;
  out_params.data = correct;
  out_params.len = 60;
  std::vector<int> out_shape{1, 6, 10};
  out_params.shape = &out_shape;
  out_params.min = -50;
  out_params.max = 50;

  auto matmul_param = new MatMulParameter();
  matmul_param->a_transpose_ = false;
  matmul_param->b_transpose_ = false;
  matmul_param->has_bias_ = false;
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  MMInt8TestInit(&inputs, &outputs, &in_params, &weight_params, &out_params);
  auto ctx = new lite::InnerContext;
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::MatmulInt8CPUKernel *mm =
    new kernel::MatmulInt8CPUKernel(reinterpret_cast<OpParameter *>(matmul_param), inputs, outputs, ctx);

  mm->Init();
  mm->Run();
  float out_scale;
  int out_zp;
  QuantProcess(correct, out_params.len, out_params.min, out_params.max, &out_scale, &out_zp, nullptr);
  float *out = new float[out_params.len];
  Dequantize(reinterpret_cast<int8_t *>(outputs[0]->MutableData()), outputs[0]->ElementsNum(), out_scale, out_zp, out);
  ASSERT_EQ(0, CompareOutputData(out, correct, 6, 0.6));
  delete mm;
  for (auto t : inputs) delete t;
  for (auto t : outputs) delete t;
  delete[] out;
}

}  // namespace mindspore
