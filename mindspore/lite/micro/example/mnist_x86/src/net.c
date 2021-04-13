
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
  memcpy(outputs[0], g_Buffer+32, 40);
  return RET_OK;
}

int GetBufferSize() {
  return 39248;
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
  {
memset((int16_t *)(g_Buffer+10144), 0, 2048);
memset((int16_t *)(g_Buffer+12192), 0, 256);
memset((int *)(g_Buffer+12448), 0, 6144);
memset((int8_t *)(g_Buffer+18592), 0, 8112);
memset((int16_t *)(g_Buffer+26704), 0, 12544);
QuantArg conv_param__quant_arg_in[1] = {{0.003921568859368562698, -128}};
QuantArg conv_param__quant_arg_w[12] = {{0.005689438898116350174, 0}, {0.006241692230105400085, 0}, {0.007301395758986473083, 0}, {0.005148916970938444138, 0}, {0.005132303573191165924, 0}, {0.004976313561201095581, 0}, {0.00564815988764166832, 0}, {0.002269793068990111351, 0}, {0.0030086529441177845, 0}, {0.005234404932707548141, 0}, {0.007580270525068044662, 0}, {0.004589735530316829681, 0}};
QuantArg conv_param__quant_arg_out[1] = {{0.01811622083187103271, 17}};
double conv_param__real_multiplier[12] = {0.001231577267748737653, 0.001351122051282624588, 0.00158051323770531417, 0.001114571969708069233, 0.001110975704014940469, 0.001077209041359399825, 0.001222641776980984765, 0.0004913359221160916793, 0.0006512749113606706042, 0.001133077320583530554, 0.001640880438584302065, 0.0009935275121536731122};
int conv_param__left_shift[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
int conv_param__right_shift[12] = {-9, -9, -9, -9, -9, -9, -9, -10, -10, -9, -9, -9};
int conv_param__quant_multiplier[12] = {1354133526, 1485574406, 1737792683, 1225484841, 1221530705, 1184403867, 1344308850, 1080459119, 1432168676, 1245831689, 1804167122, 1092395052};
int conv_param__out_act_min[1] = {-128};
int conv_param__out_act_max[1] = {127};
ConvQuantArg conv_param__conv_quant_arg = {(RoundingMode)(2), 2, conv_param__quant_arg_in, conv_param__quant_arg_w, conv_param__quant_arg_out, conv_param__real_multiplier, conv_param__left_shift, conv_param__right_shift, conv_param__quant_multiplier, conv_param__out_act_min, conv_param__out_act_max, 1, 12, 1, 2};
int thread_num = MSMIN(g_thread_num, 26);
ConvParameter conv_param_ = {{ "", false, 35, g_thread_num, 0}, conv_param__conv_quant_arg, 3, 3, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 28, 28, 1, 1, 26, 26, 12, thread_num, 0, 0, (PadMode)(2), (ActType)(0), 0, 0, 0};
PackInputToC8Int8((int8_t *)(g_Input0), (int16_t *)(g_Buffer+26704), &conv_param_);
Conv3x3Int8((int16_t *)(g_Buffer+26704), g_Weight10, g_Weight11, (int8_t *)(g_Buffer+0), (int16_t *)(g_Buffer+10144), (int16_t *)(g_Buffer+12192), (int *)(g_Buffer+12448), (int8_t *)(g_Buffer+18592), 0, &conv_param_);
PackNC4HW4ToNHWCInt8((int8_t *)(g_Buffer+18592), (int8_t *)(g_Buffer+0), 1, 676, 12);
  }
  {
static QuantArg pooling_parameter_quant_in = {0.01811622083187103271, 17};
static QuantArg pooling_parameter_quant_out = {0.01811622083187103271, 17};
static QuantArg *pooling_parameter_quant[2] = { &pooling_parameter_quant_in,  &pooling_parameter_quant_out};
const PoolingParameter pooling_parameter = {{ "", false, 92, g_thread_num, 0}, (PoolMode)(1), (RoundMode)(2), (PadMode)(2), (ActType)(0), 0, false, 2, 2, 2, 2, 26, 26, 1, 12, 13, 13, 1, 12, 0, 0, 0, 0, 0, pooling_parameter_quant, false};
MaxPoolingInt8((int8_t *)(g_Buffer+0), (int8_t *)(g_Buffer+8112), (PoolingParameter *)&pooling_parameter, 0);
  }
  {
memset((int16_t *)(g_Buffer+10144), 0, 4096);
memset((int16_t *)(g_Buffer+14240), 0, 256);
memset((int *)(g_Buffer+14496), 0, 6144);
memset((int8_t *)(g_Buffer+20640), 0, 1452);
memset((int16_t *)(g_Buffer+22092), 0, 5408);
QuantArg conv_param__quant_arg_in[1] = {{0.01811622083187103271, 17}};
QuantArg conv_param__quant_arg_w[12] = {{0.006381968967616558075, 0}, {0.005092236679047346115, 0}, {0.004954888485372066498, 0}, {0.007594361435621976852, 0}, {0.006317862775176763535, 0}, {0.004739056341350078583, 0}, {0.004733041394501924515, 0}, {0.005125139374285936356, 0}, {0.005773660261183977127, 0}, {0.007067613303661346436, 0}, {0.00728381425142288208, 0}, {0.004714466165751218796, 0}};
QuantArg conv_param__quant_arg_out[1] = {{0.118615470826625824, 31}};
double conv_param__real_multiplier[12] = {0.0009747224012760375951, 0.0007777407468524931162, 0.0007567634496453238277, 0.001159891919861241348, 0.0009649314419479496259, 0.0007237992569070154231, 0.0007228806183814449719, 0.0007827659621256170689, 0.0008818150205007141765, 0.001079441365823280083, 0.001112461807995879974, 0.0007200436103814696152};
int conv_param__left_shift[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
int conv_param__right_shift[12] = {-10, -10, -10, -9, -10, -10, -10, -10, -10, -9, -9, -10};
int conv_param__quant_multiplier[12] = {2143437228, 1710269989, 1664140425, 1275314653, 2121906681, 1591651398, 1589631291, 1721320554, 1939131737, 1186858333, 1223164693, 1583392644};
int conv_param__out_act_min[1] = {-128};
int conv_param__out_act_max[1] = {127};
ConvQuantArg conv_param__conv_quant_arg = {(RoundingMode)(1), 2, conv_param__quant_arg_in, conv_param__quant_arg_w, conv_param__quant_arg_out, conv_param__real_multiplier, conv_param__left_shift, conv_param__right_shift, conv_param__quant_multiplier, conv_param__out_act_min, conv_param__out_act_max, 1, 12, 1, 2};
int thread_num = MSMIN(g_thread_num, 11);
ConvParameter conv_param_ = {{ "", false, 35, g_thread_num, 0}, conv_param__conv_quant_arg, 3, 3, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 13, 13, 12, 1, 11, 11, 12, thread_num, 0, 0, (PadMode)(2), (ActType)(0), 0, 0, 0};
PackInputToC8Int8((int8_t *)(g_Buffer+8112), (int16_t *)(g_Buffer+22092), &conv_param_);
Conv3x3Int8((int16_t *)(g_Buffer+22092), g_Weight12, g_Weight13, (int8_t *)(g_Buffer+0), (int16_t *)(g_Buffer+10144), (int16_t *)(g_Buffer+14240), (int *)(g_Buffer+14496), (int8_t *)(g_Buffer+20640), 0, &conv_param_);
PackNC4HW4ToNHWCInt8((int8_t *)(g_Buffer+20640), (int8_t *)(g_Buffer+0), 1, 121, 12);
  }
  {
static QuantArg pooling_parameter_quant_in = {0.118615470826625824, 31};
static QuantArg pooling_parameter_quant_out = {0.118615470826625824, 31};
static QuantArg *pooling_parameter_quant[2] = { &pooling_parameter_quant_in,  &pooling_parameter_quant_out};
const PoolingParameter pooling_parameter = {{ "", false, 92, g_thread_num, 0}, (PoolMode)(1), (RoundMode)(2), (PadMode)(2), (ActType)(0), 0, false, 2, 2, 2, 2, 11, 11, 1, 12, 5, 5, 1, 12, 0, 0, 0, 0, 0, pooling_parameter_quant, false};
MaxPoolingInt8((int8_t *)(g_Buffer+0), (int8_t *)(g_Buffer+1456), (PoolingParameter *)&pooling_parameter, 0);
  }
  {
const ReshapeQuantArg reshape_quant_arg = {{0.118615470826625824, 31}, {0.118615470826625824, 31}, -128, 127};
Int8Reshape((int8_t *)(g_Buffer+1456), (int8_t *)(g_Buffer+0), 300, reshape_quant_arg);
  }
  {
int32_t tmp_weight_zp = 0;
RowMajor2Row16x4MajorInt8((int8_t *)(g_Buffer+0)+0, (int8_t *)(g_Buffer+10144), 1, 300);
CalcInputSums((int8_t *)(g_Buffer+0)+0, 1, 300, tmp_weight_zp, (int *)(g_Buffer+11360), RowMajor);
float filter_scale[1] = {0.007667620200663805008};
int filter_zp[1] = {0};
int left_shift[1] = {0};
int right_shift[1] = {-8};
int multiplier[1] = {1379728867};
const MatmulQuantParameter matmul_quant_parameter = {{0.118615470826625824, 31}, {0, 0}, {0.3623915016651153564, 11}, -128, 127, filter_scale, filter_zp, left_shift, right_shift, multiplier};
int32_t *cur_left = matmul_quant_parameter.left_shift_;
int32_t *cur_right = matmul_quant_parameter.right_shift_;
int32_t *cur_mul = matmul_quant_parameter.quant_multiplier_ ;
int32_t *cur_zp = matmul_quant_parameter.filter_zp_ ;
MatmulInt8Opt((int8_t *)(g_Buffer+10144), g_Weight15+0 + 0, (int8_t *)(g_Buffer+304)+0+0, 1, 20, 304, (int *)(g_Buffer+11360), g_Weight16+0, -128, 127, 11, cur_mul, cur_left, cur_right, 20, false, cur_zp);
  }
  {
int32_t tmp_weight_zp = 0;
RowMajor2Row16x4MajorInt8((int8_t *)(g_Buffer+304)+0, (int8_t *)(g_Buffer+10144), 1, 20);
CalcInputSums((int8_t *)(g_Buffer+304)+0, 1, 20, tmp_weight_zp, (int *)(g_Buffer+10272), RowMajor);
float filter_scale[1] = {0.006908571347594261169};
int filter_zp[1] = {0};
int left_shift[1] = {0};
int right_shift[1] = {-8};
int multiplier[1] = {1282256865};
const MatmulQuantParameter matmul_quant_parameter = {{0.3623915016651153564, 11}, {0, 0}, {1.073398709297180176, -20}, -128, 127, filter_scale, filter_zp, left_shift, right_shift, multiplier};
int32_t *cur_left = matmul_quant_parameter.left_shift_;
int32_t *cur_right = matmul_quant_parameter.right_shift_;
int32_t *cur_mul = matmul_quant_parameter.quant_multiplier_ ;
int32_t *cur_zp = matmul_quant_parameter.filter_zp_ ;
MatmulInt8Opt((int8_t *)(g_Buffer+10144), g_Weight18+0 + 0, (int8_t *)(g_Buffer+0)+0+0, 1, 10, 32, (int *)(g_Buffer+10272), g_Weight19+0, -128, 127, -20, cur_mul, cur_left, cur_right, 10, false, cur_zp);
  }
  {
const SoftmaxQuantArg quant_args = {{1.073398709297180176, 20}, {0.00390625, -128}, -128, 127, 1152553088, 27, 27};
const SoftmaxParameter softmax_parameter = {{ "", false, 138, g_thread_num, 0}, 1, {1, 10}, 10, 2};
memset((int *)(g_Buffer+10144), 0, 40);
memset((int *)(g_Buffer+10184), 0, 40);
SoftmaxInt8((int8_t *)(g_Buffer+0), (int8_t *)(g_Buffer+16), 1, (int *)(g_Buffer+10144), (int *)(g_Buffer+10184), quant_args, (SoftmaxParameter *)&softmax_parameter);
  }
  {
DoDequantizeInt8ToFp32((int8_t *)(g_Buffer+16), (float *)(g_Buffer+32), 0.00390625, -128, 10);
  }
}
