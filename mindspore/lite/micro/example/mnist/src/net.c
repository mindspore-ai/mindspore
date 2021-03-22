
/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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


#include "weight.h"
#include "net.h"

static const unsigned char *g_Input0 = 0;
int SetInputs(const void **inputs, int num) {
  if (inputs == NULL) {
    return RET_ERROR;
  }
  if (num !=1) {
    return RET_ERROR;
  }
	g_Input0 = inputs[0];
  return RET_OK;
}
int CopyOutputsData(void **outputs, int num) {
  if (outputs == NULL) {
    return RET_ERROR;
  }
  if (num != 1) {
    return RET_ERROR;
  }
  memcpy(outputs[0], g_Buffer+56, 40);
  return RET_OK;
}

int GetBufferSize() {
  return 40032;
}
int SetBuffer( void *buffer) {
  if (buffer == NULL) {
    return RET_ERROR;
  }
  g_Buffer = buffer;
  return RET_OK;
}
void FreeResource() {
  g_Buffer= NULL;
  g_Input0 = NULL;
  void *allocated[] = {g_Weight14, g_Weight15, g_Weight16, g_Weight17, g_Weight18, g_Weight19,   };
  for (int i = 0; i < 6; ++i) {
    free(allocated[i]);
    allocated[i] = NULL;
  }
}
void Inference() {
  const int g_thread_num = 1;
  {
DoQuantizeFp32ToInt8((float *)(g_Input0), (int8_t *)(g_Buffer+0), 0.007874015718698501587, 0, 784, false);
  }
  {
memset((int16_t *)(g_Buffer+10928), 0, 2048);
memset((int16_t *)(g_Buffer+12976), 0, 256);
memset((int *)(g_Buffer+13232), 0, 6144);
memset((int8_t *)(g_Buffer+19376), 0, 8112);
memset((int16_t *)(g_Buffer+27488), 0, 12544);
static QuantArg conv_param__quant_arg_in[1] = {{0.007874015718698501587, 0}};
static QuantArg conv_param__quant_arg_w[12] = {{0.003238174133002758026, -6}, {0.003890725085511803627, -8}, {0.003394871251657605171, -7}, {0.001685356837697327137, -127}, {0.004322394262999296188, 1}, {0.002274985425174236298, -56}, {0.003617759561166167259, 17}, {0.004447745624929666519, 23}, {0.004683905746787786484, 26}, {0.004021023400127887726, 24}, {0.005650237202644348145, 11}, {0.001966834301128983498, -84}};
static QuantArg conv_param__quant_arg_out[1] = {{0.01778890006244182587, 0}};
static double conv_param__real_multiplier[12] = {0.001433333970799530351, 0.001722176774828924938, 0.00150269379968211614, 0.0007460003866156953226, 0.001913249346122961134, 0.001006991503636309139, 0.001601352314486244018, 0.001968734305210294733, 0.002073267527210802957, 0.00177985160945266568, 0.002501001060249878095, 0.0008705926067589928779};
static int conv_param__left_shift[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int conv_param__right_shift[12] = {-9, -9, -9, -10, -9, -9, -9, -8, -8, -9, -8, -10};
static int conv_param__quant_multiplier[12] = {1575967367, 1893553389, 1652229306, 1640472199, 2103639903, 1107198867, 1760705490, 1082323130, 1139790877, 1956967540, 1374939873, 1914453388};
static int conv_param__out_act_min[1] = {0};
static int conv_param__out_act_max[1] = {127};
ConvQuantArg conv_param__conv_quant_arg = {(RoundingMode)(1), 2, conv_param__quant_arg_in, conv_param__quant_arg_w, conv_param__quant_arg_out, conv_param__real_multiplier, conv_param__left_shift, conv_param__right_shift, conv_param__quant_multiplier, conv_param__out_act_min, conv_param__out_act_max, 1, 12, 1, 2};
int thread_num = MSMIN(g_thread_num, 26);
ConvParameter conv_param_ = {{ "", 35, g_thread_num}, conv_param__conv_quant_arg, 3, 3, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 28, 28, 1, 1, 26, 26, 12, thread_num, 0, 0, (PadMode)(2), (ActType)(1), 0, 0, 0};
PackInputToC8Int8((int8_t *)(g_Buffer+0), (int16_t *)(g_Buffer+27488), &conv_param_);
Conv3x3Int8((int16_t *)(g_Buffer+27488), g_Weight10, g_Weight11, (int8_t *)(g_Buffer+784), (int16_t *)(g_Buffer+10928), (int16_t *)(g_Buffer+12976), (int *)(g_Buffer+13232), (int8_t *)(g_Buffer+19376), 0, &conv_param_);
PackNC4HW4ToNHWCInt8((int8_t *)(g_Buffer+19376), (int8_t *)(g_Buffer+784), 1, 676, 12);
  }
  {
static QuantArg pooling_parameter_quant_in = {0.01778890006244182587, 0};
static QuantArg pooling_parameter_quant_out = {0.01778890006244182587, 0};
static QuantArg *pooling_parameter_quant[2] = { &pooling_parameter_quant_in,  &pooling_parameter_quant_out};
const PoolingParameter pooling_parameter = {{ "", 92, g_thread_num}, (PoolMode)(1), (RoundMode)(2), (PadMode)(2), (ActType)(0), 0, false, 2, 2, 2, 2, 26, 26, 1, 12, 13, 13, 1, 12, 0, 0, 0, 0, 0, pooling_parameter_quant, false};
MaxPoolingInt8((int8_t *)(g_Buffer+784), (int8_t *)(g_Buffer+8896), (PoolingParameter *)&pooling_parameter, 0);
  }
  {
memset((int16_t *)(g_Buffer+10928), 0, 4096);
memset((int16_t *)(g_Buffer+15024), 0, 256);
memset((int *)(g_Buffer+15280), 0, 6144);
memset((int8_t *)(g_Buffer+21424), 0, 1452);
memset((int16_t *)(g_Buffer+22876), 0, 5408);
static QuantArg conv_param__quant_arg_in[1] = {{0.01778890006244182587, 0}};
static QuantArg conv_param__quant_arg_w[12] = {{0.005374609492719173431, 33}, {0.005837683100253343582, 22}, {0.004709810949862003326, -15}, {0.003726204857230186462, 27}, {0.00318551529198884964, -8}, {0.003453079145401716232, 50}, {0.004045850131660699844, -9}, {0.003903790842741727829, 30}, {0.004003710579127073288, -10}, {0.00560879148542881012, 27}, {0.005486610345542430878, -23}, {0.003554018214344978333, 4}};
static QuantArg conv_param__quant_arg_out[1] = {{0.07183934003114700317, 0}};
static double conv_param__real_multiplier[12] = {0.001330863973520378732, 0.001445530533608141606, 0.001166246148374064893, 0.0009226850783705293785, 0.0007887991893445710223, 0.0008550534992628172192, 0.001001835847923064193, 0.0009666590447744700769, 0.0009914011740411567478, 0.001388852288199173826, 0.00135859773990280961, 0.0008800481219728497088};
static int conv_param__left_shift[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int conv_param__right_shift[12] = {-9, -9, -9, -10, -10, -10, -9, -10, -9, -9, -9, -10};
static int conv_param__quant_multiplier[12] = {1463300414, 1589377630, 1282301201, 2029005945, 1734587761, 1880282530, 1101530164, 2125705720, 1090057119, 1527059240, 1493794012, 1935246286};
static int conv_param__out_act_min[1] = {0};
static int conv_param__out_act_max[1] = {127};
ConvQuantArg conv_param__conv_quant_arg = {(RoundingMode)(1), 2, conv_param__quant_arg_in, conv_param__quant_arg_w, conv_param__quant_arg_out, conv_param__real_multiplier, conv_param__left_shift, conv_param__right_shift, conv_param__quant_multiplier, conv_param__out_act_min, conv_param__out_act_max, 1, 12, 1, 2};
int thread_num = MSMIN(g_thread_num, 11);
ConvParameter conv_param_ = {{ "", 35, g_thread_num}, conv_param__conv_quant_arg, 3, 3, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 13, 13, 12, 1, 11, 11, 12, thread_num, 0, 0, (PadMode)(2), (ActType)(1), 0, 0, 0};
PackInputToC8Int8((int8_t *)(g_Buffer+8896), (int16_t *)(g_Buffer+22876), &conv_param_);
Conv3x3Int8((int16_t *)(g_Buffer+22876), g_Weight12, g_Weight13, (int8_t *)(g_Buffer+0), (int16_t *)(g_Buffer+10928), (int16_t *)(g_Buffer+15024), (int *)(g_Buffer+15280), (int8_t *)(g_Buffer+21424), 0, &conv_param_);
PackNC4HW4ToNHWCInt8((int8_t *)(g_Buffer+21424), (int8_t *)(g_Buffer+0), 1, 121, 12);
  }
  {
static QuantArg pooling_parameter_quant_in = {0.07136065512895584106, 0};
static QuantArg pooling_parameter_quant_out = {0.07136065512895584106, 0};
static QuantArg *pooling_parameter_quant[2] = { &pooling_parameter_quant_in,  &pooling_parameter_quant_out};
const PoolingParameter pooling_parameter = {{ "", 92, g_thread_num}, (PoolMode)(1), (RoundMode)(2), (PadMode)(2), (ActType)(0), 0, false, 2, 2, 2, 2, 11, 11, 1, 12, 5, 5, 1, 12, 0, 0, 0, 0, 0, pooling_parameter_quant, false};
MaxPoolingInt8((int8_t *)(g_Buffer+0), (int8_t *)(g_Buffer+1456), (PoolingParameter *)&pooling_parameter, 0);
  }
  {
const ReshapeQuantArg reshape_quant_arg = {{0.07136065512895584106, 0}, {0.07136065512895584106, 0}, -128, 127};
Int8Reshape((int8_t *)(g_Buffer+1456), (int8_t *)(g_Buffer+0), 300, reshape_quant_arg);
  }
  {
int32_t tmp_weight_zp = 1;
RowMajor2Row16x4MajorInt8((int8_t *)(g_Buffer+0)+0, (int8_t *)(g_Buffer+10928), 1, 300);
CalcInputSums((int8_t *)(g_Buffer+0)+0, 1, 300, tmp_weight_zp, (int *)(g_Buffer+12144), RowMajor);
static float filter_scale[20] = {0.003479549195617437363, 0.004490676335990428925, 0.004529818892478942871, 0.002983231563121080399, 0.003455155529081821442, 0.003223794745281338692, 0.003272445406764745712, 0.003801185870543122292, 0.003679843153804540634, 0.003040234791114926338, 0.003704284550622105598, 0.003355232765898108482, 0.002904496388509869576, 0.003024494973942637444, 0.002794801956042647362, 0.004355110693722963333, 0.003499472280964255333, 0.004184196703135967255, 0.003057289868593215942, 0.003264668164774775505};
static int filter_zp[20] = {1, 12, 3, 2, -10, -5, -11, 5, 12, 22, 16, 1, -5, 15, 13, 5, -10, -5, -6, 0};
static int left_shift[20] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int right_shift[20] = {-10, -9, -9, -10, -10, -10, -10, -9, -9, -10, -9, -10, -10, -10, -10, -9, -10, -9, -10, -10};
static int multiplier[20] = {2108215049, 1360422072, 1372280070, 1807502393, 2093435146, 1953256619, 1982733521, 1151545365, 1114785262, 1842040025, 1122189669, 2032893316, 1759797843, 1832503464, 1693335354, 1319353429, 2120286176, 1267576078, 1852373503, 1978021333};
const MatmulQuantParameter matmul_quant_parameter = {{0.07136065512895584106, 0}, {0, 0}, {0.258998185396194458, 0}, -128, 127, filter_scale, filter_zp, left_shift, right_shift, multiplier};
int32_t *cur_left = matmul_quant_parameter.left_shift_ + 0;
int32_t *cur_right = matmul_quant_parameter.right_shift_ + 0;
int32_t *cur_mul = matmul_quant_parameter.quant_multiplier_  + 0;
int32_t *cur_zp = matmul_quant_parameter.filter_zp_  + 0;
MatmulInt8Opt((int8_t *)(g_Buffer+10928), g_Weight15+0 + 0, (int8_t *)(g_Buffer+304)+0+0, 1, 20, 304, (int *)(g_Buffer+12144), g_Weight16+0, -128, 127, 0, cur_mul, cur_left, cur_right, 20, true, cur_zp);
  }
  {
int32_t tmp_weight_zp = 1;
RowMajor2Row16x4MajorInt8((int8_t *)(g_Buffer+304)+0, (int8_t *)(g_Buffer+10928), 1, 20);
CalcInputSums((int8_t *)(g_Buffer+304)+0, 1, 20, tmp_weight_zp, (int *)(g_Buffer+11056), RowMajor);
static float filter_scale[10] = {0.004678330849856138229, 0.005127115640789270401, 0.00471437256783246994, 0.004531511571258306503, 0.005476122256368398666, 0.004348111804574728012, 0.004803542047739028931, 0.006081215571612119675, 0.004532597027719020844, 0.004762654658406972885};
static int filter_zp[10] = {7, -2, 9, 2, -6, 21, 16, 10, -19, 8};
static int left_shift[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int right_shift[10] = {-8, -8, -8, -8, -8, -8, -8, -8, -8, -8};
static int multiplier[10] = {1242805482, 1362025788, 1252380041, 1203802750, 1454739904, 1155082292, 1276068015, 1615483838, 1204091115, 1265206260};
const MatmulQuantParameter matmul_quant_parameter = {{0.258998185396194458, 0}, {0, 0}, {0.5359870791435241699, 0}, -128, 127, filter_scale, filter_zp, left_shift, right_shift, multiplier};
int32_t *cur_left = matmul_quant_parameter.left_shift_ + 0;
int32_t *cur_right = matmul_quant_parameter.right_shift_ + 0;
int32_t *cur_mul = matmul_quant_parameter.quant_multiplier_  + 0;
int32_t *cur_zp = matmul_quant_parameter.filter_zp_  + 0;
MatmulInt8Opt((int8_t *)(g_Buffer+10928), g_Weight18+0 + 0, (int8_t *)(g_Buffer+0)+0+0, 1, 10, 32, (int *)(g_Buffer+11056), g_Weight19+0, -128, 127, 0, cur_mul, cur_left, cur_right, 10, true, cur_zp);
  }
  {
DoDequantizeInt8ToFp32((int8_t *)(g_Buffer+0), (float *)(g_Buffer+16), 0.5359870791435241699, 0, 10);
  }
  {
const SoftmaxParameter softmax_parameter = {{ "", 138, g_thread_num}, 1, {1, 10}, 10, 2};
memset((float *)(g_Buffer+10928), 0, 4);
Softmax((float *)(g_Buffer+16), (float *)(g_Buffer+56), (float *)(g_Buffer+10928), &softmax_parameter);
  }
}
