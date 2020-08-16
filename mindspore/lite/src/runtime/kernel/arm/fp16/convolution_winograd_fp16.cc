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

#include "src/runtime/kernel/arm/fp16/convolution_winograd_fp16.h"
#include "src/runtime/kernel/arm/fp16/matrix_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp16/conv_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp16/common_func.h"
#include "src/runtime/kernel/arm/nnacl/fp16/cast_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp16/pack_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp16/winograd_transform_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp16/winograd_utils_fp16.h"
#include "src/runtime/kernel/arm/fp16/layout_transform_fp16.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2D;

namespace mindspore::kernel {
void WinogradFilterTransformFp16(const float16_t *weight_data, Matrix *trans_weight, int kernel_unit, int input_unit,
                                 ConvParameter *conv_param, int oc_block) {
  // original weight format : ohwi
  auto channel_in = conv_param->input_channel_;
  auto channel_out = conv_param->output_channel_;
  int input_unit_square = input_unit * input_unit;

  // generate matrix_G && matrix_GT
  auto matrix_g = TransformMatrixGenerator(input_unit, kernel_unit);
  auto matrix_gt = TransformMatrixGenerator(kernel_unit, input_unit);
  ChooseMatrixG(matrix_g, matrix_gt);
  auto matrix_g_data = reinterpret_cast<float *>(matrix_g->GetData());
  auto matrix_gt_data = reinterpret_cast<float *>(matrix_gt->GetData());
  auto matrix_g_data_fp16 = reinterpret_cast<float16_t *>(malloc(input_unit * kernel_unit * sizeof(float16_t)));
  auto matrix_gt_data_fp16 = reinterpret_cast<float16_t *>(malloc(input_unit * kernel_unit * sizeof(float16_t)));
  Float32ToFloat16(matrix_g_data, matrix_g_data_fp16, input_unit * kernel_unit);
  Float32ToFloat16(matrix_gt_data, matrix_gt_data_fp16, input_unit * kernel_unit);

  // trans_filter = G*g*GT (g represents weight_data)
  // separate into two steps ===> tmp = G*g ===> out = tmp * GT
  auto tmp_weight_data = reinterpret_cast<float16_t *>(malloc(kernel_unit * kernel_unit * sizeof(float16_t)));
  auto tmp_data = reinterpret_cast<float16_t *>(malloc(input_unit * kernel_unit * sizeof(float16_t)));
  auto trans_out_data = reinterpret_cast<float16_t *>(malloc(input_unit * input_unit * sizeof(float16_t)));
  bool row = true;
  auto trans_weight_data = reinterpret_cast<float16_t *>(trans_weight->GetData());
  std::vector<int> strides = trans_weight->GetStride();

  int kernel_plane_stride = channel_in;
  if (oc_block == 0) {
    MS_LOG(ERROR) << "Divide by zero";
    return;
  }
  for (int i = 0; i < channel_out; i++) {
    int out_c_block = i / oc_block;
    int out_c_res = i % oc_block;
    int input_oz_offset = i * kernel_unit * kernel_unit * channel_in;
    int output_oz_offset = out_c_block * strides[1] * input_unit * input_unit + out_c_res;
    for (int j = 0; j < channel_in; j++) {
      int ic4_block = j / C4NUM;
      int ic4_res = j % C4NUM;
      int input_iz_offset = input_oz_offset + j;
      int output_iz_offset = output_oz_offset + ic4_block * strides[2] + ic4_res * strides[3];
      for (int k = 0; k < kernel_unit * kernel_unit; k++) {
        int input_xy_offset = input_iz_offset + k * kernel_plane_stride;
        tmp_weight_data[k] = *(weight_data + input_xy_offset);
      }
      // now we only support row-major matrix-multiply
      // tmp = G * g
      MatrixMultiplyFp16(matrix_g_data_fp16, tmp_weight_data, tmp_data, input_unit, kernel_unit, kernel_unit, row);
      // out = tmp * GT
      MatrixMultiplyFp16(tmp_data, matrix_gt_data_fp16, trans_out_data, input_unit, kernel_unit, input_unit, row);

      for (int z = 0; z < input_unit_square; z++) {
        int output_xy_offset = output_iz_offset + z * strides[1];
        *(trans_weight_data + output_xy_offset) = trans_out_data[z];
      }
    }
  }
  free(tmp_weight_data);
  free(tmp_data);
  free(trans_out_data);
  free(matrix_g_data_fp16);
  free(matrix_gt_data_fp16);
  delete matrix_g;
  delete matrix_gt;
}

int ConvolutionWinogradFP16CPUKernel::InitWeightBias() {
  int output_channel = conv_param_->output_channel_;
  int oc_block, oc_block_num;
  oc_block = C8NUM;
  oc_block_num = UP_DIV(output_channel, C8NUM);

  // init weight
  auto ret = MallocFilterMatrix(oc_block, oc_block_num);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc filter matrix failed.";
    return RET_ERROR;
  }

  ret = ConvolutionBaseFP16CPUKernel::GetExecuteFilter();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get Execute filter failed.";
    return ret;
  }
  WinogradFilterTransformFp16(execute_weight_, trans_weight_, kernel_unit_, input_unit_, conv_param_, oc_block);

  // init bias
  bias_data_ = malloc(oc_block_num * oc_block * sizeof(float16_t));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc bias_data_ failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, oc_block_num * oc_block * sizeof(float16_t));
  auto fp16_bias_data = reinterpret_cast<float16_t *>(bias_data_);
  if (in_tensors_.size() == kInputSize2) {
    auto ori_bias = reinterpret_cast<float *>(in_tensors_.at(kBiasIndex)->Data());
    for (int i = 0; i < output_channel; ++i) {
      fp16_bias_data[i] = (float16_t)ori_bias[i];
    }
  } else {
    MS_ASSERT(inputs_.size() == kInputSize1);
  }
  return RET_OK;
}

int ConvolutionWinogradFP16CPUKernel::MallocFilterMatrix(int oc_block, int oc_block_num) {
  int channel_in = conv_param_->input_channel_;
  int ic4 = UP_DIV(channel_in, BLOCK);

  // set data
  auto trans_matrix_data_size = input_unit_ * input_unit_ * ic4 * C4NUM * oc_block_num * oc_block * sizeof(float);
  auto matrix_buffer = malloc(trans_matrix_data_size);
  if (matrix_buffer == nullptr) {
    MS_LOG(ERROR) << "malloc matrix_buffer failed.";
    return RET_ERROR;
  }
  memset(matrix_buffer, 0, trans_matrix_data_size);
  trans_weight_ = new Matrix();
  trans_weight_->SetData(matrix_buffer);
  trans_weight_->SetNDim(5);

  std::vector<int> shapes;
  std::vector<int> strides;
  // set shape
  shapes.push_back(input_unit_ * input_unit_);
  shapes.push_back(oc_block_num);
  shapes.push_back(ic4);
  shapes.push_back(C4NUM);
  shapes.push_back(oc_block);
  // set stride
  for (int i = 0; i < 4; i++) {
    int stride = 1;
    for (int j = i + 1; j < 5; j++) {
      stride *= shapes[j];
    }
    strides.push_back(stride);
  }
  trans_weight_->SetShape(shapes);
  trans_weight_->SetStride(strides);
  return RET_OK;
}

int ConvolutionWinogradFP16CPUKernel::InitTmpBuffer() {
  int cal_num = 16;
  int channel_in = conv_param_->input_channel_;
  int channel_out = conv_param_->output_channel_;
  int output_h = conv_param_->output_h_;
  int output_w = conv_param_->output_w_;
  int ic4 = UP_DIV(channel_in, C4NUM);
  int oc8 = UP_DIV(channel_out, C8NUM);

  /*=============================fp16_input_============================*/
  size_t fp16_input_size = conv_param_->input_channel_ * conv_param_->input_batch_ * conv_param_->input_h_ *
                           conv_param_->input_w_ * sizeof(float16_t);
  fp16_input_ = reinterpret_cast<float16_t *>(malloc(fp16_input_size));
  if (fp16_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc fp16_input_ failed.";
    return RET_ERROR;
  }

  /*=============================trans_input_============================*/
  size_t tile_buffer_size = thread_count_ * cal_num * input_unit_ * input_unit_ * ic4 * C4NUM * sizeof(float16_t);
  trans_input_ = reinterpret_cast<float16_t *>(malloc(tile_buffer_size));
  if (trans_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc trans_input_ failed.";
    return RET_ERROR;
  }
  memset(trans_input_, 0, tile_buffer_size);

  /*=============================gemm_out_============================*/
  gemm_out_ = reinterpret_cast<float16_t *>(
    malloc(thread_count_ * cal_num * input_unit_ * input_unit_ * oc8 * C8NUM * sizeof(float16_t)));
  if (gemm_out_ == nullptr) {
    MS_LOG(ERROR) << "malloc gemm_out_ failed.";
    return RET_ERROR;
  }

  /*=============================tmp_out_data_============================*/
  int out_w_block = UP_DIV(output_w, output_unit_);
  int out_h_block = UP_DIV(output_h, output_unit_);
  tmp_out_data_ = reinterpret_cast<float16_t *>(malloc(conv_param_->output_batch_ * out_w_block * out_h_block *
                                                       output_unit_ * output_unit_ * oc8 * C8NUM * sizeof(float16_t)));
  if (tmp_out_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc tmp_out_data_ failed.";
    return RET_ERROR;
  }
  /*=============================fp16_out_============================*/
  size_t fp16_output_size = conv_param_->output_channel_ * conv_param_->output_batch_ * conv_param_->output_h_ *
                            conv_param_->output_w_ * sizeof(float16_t);
  fp16_out_ = reinterpret_cast<float16_t *>(malloc(fp16_output_size));
  if (fp16_out_ == nullptr) {
    MS_LOG(ERROR) << "malloc fp16_out_ failed.";
    return RET_ERROR;
  }

  /*=============================tmp_data_============================*/
  tmp_data_ =
    reinterpret_cast<float16_t *>(malloc(thread_count_ * C4NUM * input_unit_ * input_unit_ * sizeof(float16_t)));
  if (tmp_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc tmp_data_ failed.";
    return RET_ERROR;
  }
  memset(tmp_data_, 0, C4NUM * input_unit_ * input_unit_ * sizeof(float16_t));

  tmp_buffer_address_list_[0] = trans_input_;
  tmp_buffer_address_list_[1] = gemm_out_;
  tmp_buffer_address_list_[2] = tmp_out_data_;
  tmp_buffer_address_list_[3] = tmp_data_;

  /*=============================nhwc4_input_============================*/
  size_t nhwc4_input_size =
    ic4 * C4NUM * conv_param_->input_batch_ * conv_param_->input_h_ * conv_param_->input_w_ * sizeof(float16_t);
  nhwc4_input_ = malloc(nhwc4_input_size);
  if (nhwc4_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc nhwc4_input_ failed.";
    return RET_ERROR;
  }
  memset(nhwc4_input_, 0, nhwc4_input_size);
  return RET_OK;
}

int ConvolutionWinogradFP16CPUKernel::ConfigInputOutput() {
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto ret = CheckLayout(input_tensor);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Check layout failed.";
    return RET_ERROR;
  }
  auto output_tensor = out_tensors_.at(kOutputIndex);
  output_tensor->SetFormat(schema::Format_NHWC);

  // choose input transformer function (4x4 unit or 8x8 unit)
  input_trans_func_ = GetInputTransFuncFp16(input_unit_);
  if (input_trans_func_ == nullptr) {
    MS_LOG(ERROR) << "Get input_trans_func failed.";
    return RET_ERROR;
  }
  output_trans_func_ = GetOutputTransFuncFp16(input_unit_, output_unit_);
  if (output_trans_func_ == nullptr) {
    MS_LOG(ERROR) << "Get output_trans_func_ failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionWinogradFP16CPUKernel::Init() {
  auto ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase init failed.";
    return RET_ERROR;
  }
  kernel_unit_ = conv_param_->kernel_h_;
  input_unit_ = output_unit_ + kernel_unit_ - 1;
  conv_param_->input_unit_ = input_unit_;
  conv_param_->output_unit_ = output_unit_;

  ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init weight bias failed.";
    return RET_ERROR;
  }
  // malloc tmp buffer
  ret = InitTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init tmp buffer failed.";
    return RET_ERROR;
  }
  ret = ConfigInputOutput();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConfigInputOutput failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionWinogradFP16CPUKernel::ReSize() {
  if (tmp_data_ != nullptr) {
    free(tmp_data_);
  }
  if (trans_input_ != nullptr) {
    free(trans_input_);
  }
  if (gemm_out_ != nullptr) {
    free(gemm_out_);
  }
  if (tmp_out_data_ != nullptr) {
    free(tmp_out_data_);
  }
  if (nhwc4_input_ != nullptr) {
    free(nhwc4_input_);
  }
  if (fp16_input_ != nullptr) {
    free(fp16_input_);
  }
  if (fp16_out_ != nullptr) {
    free(fp16_out_);
  }

  auto ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase init failed.";
    return RET_ERROR;
  }
  kernel_unit_ = conv_param_->kernel_h_;
  input_unit_ = output_unit_ + kernel_unit_ - 1;
  conv_param_->input_unit_ = input_unit_;
  conv_param_->output_unit_ = output_unit_;

  ret = InitTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init tmp buffer failed.";
    return RET_ERROR;
  }
  ret = ConfigInputOutput();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConfigInputOutput failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionWinogradFP16CPUKernel::RunImpl(int task_id) {
  ConvWinogardFp16(reinterpret_cast<float16_t *>(nhwc4_input_), reinterpret_cast<float16_t *>(trans_weight_->GetData()),
                   reinterpret_cast<const float16_t *>(bias_data_), tmp_buffer_address_list_, task_id, conv_param_,
                   input_trans_func_, output_trans_func_);
  return RET_OK;
}

int ConvolutionWinogradFp16Impl(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto conv = reinterpret_cast<ConvolutionWinogradFP16CPUKernel *>(cdata);
  auto error_code = conv->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionWinograd Fp16 Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionWinogradFP16CPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }

  auto ret = ConvolutionBaseFP16CPUKernel::GetExecuteTensor();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get Execute tensor failed.";
    return ret;
  }

  int in_batch = conv_param_->input_batch_;
  int in_h = conv_param_->input_h_;
  int in_w = conv_param_->input_w_;
  int in_channel = conv_param_->input_channel_;
  convert_func_(execute_input_, nhwc4_input_, in_batch, in_h * in_w, in_channel);

  int error_code = LiteBackendParallelLaunch(ConvolutionWinogradFp16Impl, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv winograd error error_code[" << error_code << "]";
    return RET_ERROR;
  }

  // get real output
  UnPackWinogradOutputFp16(tmp_out_data_, execute_output_, conv_param_->output_batch_, conv_param_->output_h_,
                           conv_param_->output_w_, conv_param_->output_channel_, output_unit_);
  int output_num =
    conv_param_->output_channel_ * conv_param_->output_h_ * conv_param_->output_w_ * conv_param_->output_batch_;
  if (conv_param_->is_relu_) {
    ReluFp16(execute_output_, execute_output_, output_num);
  } else if (conv_param_->is_relu6_) {
    Relu6Fp16(execute_output_, execute_output_, output_num);
  } else {
    // do nothing
  }
  ConvolutionBaseFP16CPUKernel::IfCastOutput();
  return RET_OK;
}
}  // namespace mindspore::kernel
