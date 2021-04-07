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

#include "minddata/dataset/kernels/image/soft_dvpp/utils/soft_vpc.h"
#include <securec.h>
#include "minddata/dataset/kernels/image/soft_dvpp/utils/soft_dp_check.h"
#include "minddata/dataset/kernels/image/soft_dvpp/utils/soft_dp_tools.h"
#include "minddata/dataset/kernels/image/soft_dvpp/utils/yuv_scaler_para_set.h"

constexpr int32_t dpSucc = 0;
constexpr int32_t dpFail = -1;
constexpr uint32_t yuvCoeffiNum4 = 4;
constexpr uint32_t yuvCoeffiNum5 = 5;
constexpr uint32_t uvReductCoeffNum = 5;
constexpr int32_t uvReductCoeff[uvReductCoeffNum] = {13, 65, 100, 65, 13};  // yuv444 dimension reduction filter.
constexpr uint32_t scalerTap4 = 4;
constexpr uint32_t scalerTap6 = 6;
constexpr uint32_t scalerCoeff = 16;  // yuv conversion coefficient
constexpr uint32_t low3BitVal = 0x7;
constexpr int32_t low16BitVal = 0xffff;
constexpr uint32_t bit8Offset = 8;
constexpr uint32_t bit13Offset = 13;
constexpr uint32_t bit16Offset = 16;
constexpr uint32_t maxCoeff = 65536;
constexpr uint32_t num2 = 2;
constexpr int32_t scalerTap2 = 2;

// yuv convert rgb coefficient table
constexpr int32_t rtAippYuv2RgbCscMatrixR0c0 = (256);
constexpr int32_t rtAippYuv2RgbCscMatrixR0c1 = (0);
constexpr int32_t rtAippYuv2RgbCscMatrixR0c2 = (359);
constexpr int32_t rtAippYuv2RgbCscMatrixR1c0 = (256);
constexpr int32_t rtAippYuv2RgbCscMatrixR1c1 = (-88);
constexpr int32_t rtAippYuv2RgbCscMatrixR1c2 = (-183);
constexpr int32_t rtAippYuv2RgbCscMatrixR2c0 = (256);
constexpr int32_t rtAippYuv2RgbCscMatrixR2c1 = (454);
constexpr int32_t rtAippYuv2RgbCscMatrixR2c2 = (0);
constexpr int32_t rtAippYuv2RgbCscInputBias0 = (0);
constexpr int32_t rtAippYuv2RgbCscInputBias1 = (128);
constexpr int32_t rtAippYuv2RgbCscInputBias2 = (128);
constexpr int32_t rtAippConverCoeffi = (256);

SoftVpc::SoftVpc()
    : in_format_(INPUT_VPC_UNKNOWN),
      in_width_(0),
      in_height_(0),
      in_data_(nullptr),
      in_y_data_(nullptr),
      in_u_data_(nullptr),
      in_v_data_(nullptr),
      left_(0),
      right_(0),
      up_(0),
      down_(0),
      out_width_(0),
      out_height_(0),
      out_data_(nullptr),
      out_y_data_(nullptr),
      out_u_data_(nullptr),
      out_v_data_(nullptr),
      pre_scaler_num_(0),
      half_line_mode_(false),
      horizon_coeff_(0),
      vertical_coeff_(0),
      horizon_bypass_(false),
      vertical_bypass_(false),
      y_horizon_tap_(nullptr),
      uv_horizon_tap_(nullptr),
      vertical_tap_(nullptr) {}

void SoftVpc::SetYuv422OutBuffer() {
  out_y_data_ = out_data_;
  out_u_data_ = out_y_data_ + out_width_ * out_height_;
  out_v_data_ = out_u_data_ + out_width_ * out_height_ / yuvCoeffiNum2;
}

int32_t SoftVpc::CheckParamter() {
  VPC_CHECK_COND_FAIL_PRINT_RETURN((left_ < right_), dpFail, "left(%u) should be < right(%u).", left_, right_);
  VPC_CHECK_COND_FAIL_PRINT_RETURN((right_ < in_width_), dpFail, "right(%u) should be < inWidth(%u).", right_,
                                   in_width_);
  VPC_CHECK_COND_FAIL_PRINT_RETURN((up_ < down_), dpFail, "up(%u) should be < down(%u).", up_, down_);
  VPC_CHECK_COND_FAIL_PRINT_RETURN((down_ < in_height_), dpFail, "down_(%u) should be < in_height(%u).", down_,
                                   in_height_);

  uint32_t crop_width = right_ - left_ + 1;
  uint32_t crop_height = down_ - up_ + 1;
  VPC_CHECK_COND_FAIL_PRINT_RETURN((crop_width >= 10), dpFail,  // mini width is 10
                                   "right(%u) - left(%u) + 1 = crop_width(%u) should be >= 10.", right_, left_,
                                   crop_width);
  VPC_CHECK_COND_FAIL_PRINT_RETURN((in_width_ <= 8192), dpFail,  // max width is 8192
                                   "inWidth(%u) should be <= 8192.", in_width_);
  VPC_CHECK_COND_FAIL_PRINT_RETURN((crop_height >= 6), dpFail,  // mini height is 6
                                   "down(%u) - up(%u) + 1 = crop_height(%u) should be >= 6.", down_, up_, crop_height);
  VPC_CHECK_COND_FAIL_PRINT_RETURN((in_height_ <= 8192), dpFail,  // max height is 8192
                                   "inHeight(%u) should be <= 8192.", in_height_);

  uint32_t out_width = out_width_;
  uint32_t out_height = out_height_;
  bool flag = (out_width * 32 >= crop_width) ? true : false;  // A maximum of 32x zoom-out
  VPC_CHECK_COND_FAIL_PRINT_RETURN(flag, dpFail,
                                   "Max reduction multiple is 32. Please check left(%u), right(%u), out_width(%u).",
                                   left_, right_, out_width);  // Up to 16x magnification
  flag = (crop_width * 16 >= out_width) ? true : false;
  VPC_CHECK_COND_FAIL_PRINT_RETURN(flag, dpFail,
                                   "Max magnification is 16. Please check left(%u), right(%u), out_width(%u).", left_,
                                   right_, out_width);
  flag = (out_height * 32 >= crop_height) ? true : false;  // A maximum of 32x zoom-out
  VPC_CHECK_COND_FAIL_PRINT_RETURN(flag, dpFail,
                                   "Max reduction multiple is 32. Please check up(%u), down(%u), out_height(%u).", up_,
                                   down_, out_height);
  flag = (crop_height * 16 >= out_height) ? true : false;  // Up to 16x magnification
  VPC_CHECK_COND_FAIL_PRINT_RETURN(
    flag, dpFail, "Max magnification is 16. Please check up(%u), down(%u), out_height(%u).", up_, down_, out_height);
  return dpSucc;
}

void SoftVpc::Init(VpcInfo input, SoftDpCropInfo crop, VpcInfo output) {
  in_info_ = input;
  out_info_ = output;
  left_ = (crop.left & 0x1) ? (crop.left + 1) : crop.left;      // Round up the value to an even number.
  right_ = (crop.right & 0x1) ? crop.right : (crop.right - 1);  // Take an odd number downwards.
  up_ = (crop.up & 0x1) ? (crop.up + 1) : crop.up;              // Round up the value to an even number.
  down_ = (crop.down & 0x1) ? crop.down : (crop.down - 1);      // Take an odd number downwards.

  in_format_ = input.format;
  in_width_ = input.width;
  in_height_ = input.height;

  in_data_ = input.addr;
  // Offset the start address of each channel to the cropped address.
  in_y_data_ = in_data_ + up_ * in_width_ + left_;
  in_u_data_ = in_data_ + in_width_ * in_height_ + up_ * in_width_ / yuvCoeffiNum4 + left_ / yuvCoeffiNum2;
  in_v_data_ = in_data_ + in_width_ * in_height_ * yuvCoeffiNum5 / yuvCoeffiNum4 + up_ * in_width_ / yuvCoeffiNum4 +
               left_ / yuvCoeffiNum2;

  if (in_format_ == INPUT_YUV422_PLANNER) {
    in_u_data_ = in_data_ + in_width_ * in_height_ + up_ * in_width_ / yuvCoeffiNum2 + left_ / yuvCoeffiNum2;
    in_v_data_ = in_data_ + in_width_ * in_height_ * yuvCoeffiNum3 / yuvCoeffiNum2 + up_ * in_width_ / yuvCoeffiNum2 +
                 left_ / yuvCoeffiNum2;
  }

  if (in_format_ == INPUT_YUV444_PLANNER) {
    in_u_data_ = in_data_ + in_width_ * in_height_ + up_ * in_width_ + left_;
    in_v_data_ = in_data_ + in_width_ * in_height_ * yuvCoeffiNum2 + up_ * in_width_ + left_;
  }

  out_width_ = output.width;
  out_height_ = output.height;
}

// Converts the input result of the chip sub-module to the input of the next level and releases the input memory.
void SoftVpc::OutputChangeToInput() {
  in_width_ = out_width_;
  in_height_ = out_height_;
  left_ = 0;
  right_ = in_width_ - 1;
  up_ = 0;
  down_ = in_height_ - 1;
  delete[] in_data_;
  in_data_ = out_data_;
  in_y_data_ = out_y_data_;
  in_u_data_ = out_u_data_;
  in_v_data_ = out_v_data_;
}

// For the tasks that cannot be processed by the chip at a time, split the tasks whose scaling coefficients in the
// horizontal direction are greater than those in the vertical direction.
void SoftVpc::HorizonSplit(ResizeUnit *pre_unit, ResizeUnit *can_process_unit) {
  uint32_t in_width = pre_unit->in_width;
  uint32_t out_width = pre_unit->out_width;
  uint32_t in_height = pre_unit->in_height;
  uint32_t out_height = pre_unit->out_height;

  if (out_width > 4 * in_width) {  // The horizontal scaling ratio is greater than 4x.
    // Ensure that the output is less than four times of the input and the input is an even number.
    can_process_unit->in_width = AlignUp(out_width, 8) / 4;
    if (out_height > 4 * in_height) {  // The vertical scaling ratio is greater than 4x.
      // Ensure that the output is less than four times of the input and the input is an even number.
      can_process_unit->in_height = AlignUp(out_height, 8) / 4;
    } else if (out_height >= in_height) {  // The vertical scaling range is [1, 4].
      can_process_unit->in_height = in_height;
    } else if (out_height * 4 >= in_height) {  // The vertical scaling range is [1/4, 1)
      can_process_unit->in_height = out_height;
    } else {
      can_process_unit->in_height = out_height * 4;  // vertical scaling range is smaller than 1/4x
    }
  } else {  // The horizontal scaling ratio is less than or equal to 4x.
    can_process_unit->in_width = in_width;
    can_process_unit->in_height = out_height * 4;  // The vertical scaling ratio is less than 1/4.
  }

  can_process_unit->out_width = out_width;
  can_process_unit->out_height = out_height;
  pre_unit->out_width = can_process_unit->in_width;
  pre_unit->out_height = can_process_unit->in_height;
}

// For the tasks that cannot be processed by the chip at a time, split the tasks whose vertical scaling coefficients
// are greater than the horizontal scaling coefficients.
void SoftVpc::VerticalSplit(ResizeUnit *pre_unit, ResizeUnit *can_process_unit) {
  uint32_t in_width = pre_unit->in_width;
  uint32_t out_width = pre_unit->out_width;
  uint32_t in_height = pre_unit->in_height;
  uint32_t out_height = pre_unit->out_height;

  if (out_height > 4 * in_height) {  // The vertical scaling ratio is greater than 4x.
    // // Ensure that the output is less than four times of the input and the input is an even number.
    can_process_unit->in_height = AlignUp(out_height, 8) / 4;
    if (out_width > 4 * in_width) {
      can_process_unit->in_width = AlignUp(out_width, 8) / 4;
    } else if (out_width >= in_width) {
      can_process_unit->in_width = in_width;
    } else if (out_width * 4 >= in_width) {
      can_process_unit->in_width = out_width;
    } else {
      can_process_unit->in_width = out_width * 4;
    }
  } else {
    // If the vertical scaling ratio is less than or equal to 4x, the horizontal scaling
    // ratio must be less than 1/4.
    can_process_unit->in_height = in_height;
    can_process_unit->in_width = out_width * 4;  // The horizontal scaling ratio is less than 1/4.
  }

  can_process_unit->out_width = out_width;
  can_process_unit->out_height = out_height;
  pre_unit->out_width = can_process_unit->in_width;
  pre_unit->out_height = can_process_unit->in_height;
}

// Check whether the VPC chip can complete the processing at a time based on the input and output sizes.
bool SoftVpc::CanVpcChipProcess(const ResizeUnit &pre_unit) {
  uint32_t input_width = pre_unit.in_width;
  uint32_t output_width = pre_unit.out_width;
  uint32_t input_height = pre_unit.in_height;
  uint32_t output_height = pre_unit.out_height;
  uint32_t pre_scaler_num = 0;

  // 4 and 16 inorder to check whether the aspect ratio ranges from 1/4 to 4.
  while (!(IsInTheScope(4 * output_width, input_width, 16 * input_width)) ||
         !(IsInTheScope(4 * output_height, input_height, 16 * input_height))) {
    // The number of used prescalers increases by 1.
    ++pre_scaler_num;
    // Each time the prescaler is used, the input size is reduced to 1/2 of the original size divided by 2,
    // and the size must be 2-pixel aligned.
    input_width = AlignDown(input_width / 2, 2);
    // The value divided by 2 indicates that the input size is reduced to half of
    // the original size and must be 2-pixel aligned.
    input_height = AlignDown(input_height / 2, 2);
    // If the scaling coefficient is still greater than 4 after prescaler, false is returned. If the
    // scaling coefficient is greater than 4 or the number of prescalers is greater than 3, false is returned.
    if ((output_width > (4 * input_width)) || (output_height > (4 * input_height)) || (pre_scaler_num > 3)) {
      return false;
    }
  }
  return true;
}

// Creates a scaling parameter stack based on the user input and output information. The elements in the stack are
// the input and output information, and the input and output information stores the scaling information.
void SoftVpc::BuildResizeStack() {
  uint32_t in_width = right_ - left_ + 1;
  uint32_t in_height_ = down_ - up_ + 1;
  ResizeUnit pre_unit = {in_width, in_height_, out_width_, out_height_};  // Scaling information to be split.

  while (!CanVpcChipProcess(pre_unit)) {
    uint32_t input_width = pre_unit.in_width;
    uint32_t output_width = pre_unit.out_width;
    uint32_t input_height = pre_unit.in_height;
    uint32_t output_height = pre_unit.out_height;
    ResizeUnit can_process_unit = {0, 0, 0, 0};  // Scaling information that can be processed by the chip.

    // Split the input and output, the horizontal scaling coefficient is greater than
    // the vertical scaling coefficient.
    if (output_width * input_height > output_height * input_width) {
      HorizonSplit(&pre_unit, &can_process_unit);
    } else {  // The horizontal scaling coefficient is less than the vertical scaling coefficient.
      VerticalSplit(&pre_unit, &can_process_unit);
    }

    can_process_unit.out_width = output_width;
    can_process_unit.out_height = output_height;
    pre_unit.out_width = can_process_unit.in_width;
    pre_unit.out_height = can_process_unit.in_height;

    // Pushes a set of scaled information that can be processed into a stack.
    resize_stack_.push(can_process_unit);
  }

  // Push the information that can be processed by the chip for one time into the stack.
  resize_stack_.push(pre_unit);
}

int32_t SoftVpc::Yuv422pToYuv420p() {
  in_format_ = INPUT_YUV420_PLANNER;
  out_width_ = in_width_;
  out_height_ = in_height_;
  uint32_t buffer_size = out_width_ * out_height_ * yuvCoeffiNum3 / yuvCoeffiNum2;
  out_data_ = new (std::nothrow) uint8_t[buffer_size];
  VPC_CHECK_COND_FAIL_PRINT_RETURN((out_data_ != nullptr), dpFail, "alloc buffer fail.");
  out_y_data_ = out_data_;
  out_u_data_ = out_y_data_ + out_width_ * out_height_;
  out_v_data_ = out_u_data_ + out_width_ * out_height_ / yuvCoeffiNum4;

  for (uint32_t i = 0; i < out_height_; i++) {  // Y data remains unchanged.
    for (uint32_t j = 0; j < out_width_; j++) {
      out_y_data_[i * out_width_ + j] = in_y_data_[i * out_width_ + j];
    }
  }

  uint32_t yuv420_uv_w = out_width_ / yuvCoeffiNum2;
  uint32_t yuv420_uv_h = out_height_ / yuvCoeffiNum2;
  //  The UV data is reduced by half. Only the UV data of 422 odd rows is obtained.
  for (uint32_t i = 0; i < yuv420_uv_h; i++) {
    for (uint32_t j = 0; j < yuv420_uv_w; j++) {
      out_u_data_[i * yuv420_uv_w + j] = in_u_data_[i * out_width_ + j];
      out_v_data_[i * yuv420_uv_w + j] = in_v_data_[i * out_width_ + j];
    }
  }
  OutputChangeToInput();
  return dpSucc;
}

void SoftVpc::ChipPreProcess() {
  pre_scaler_num_ = 0;
  uint32_t crop_width = (right_ - left_ + 1);
  uint32_t crop_height = (down_ - up_ + 1);
  // The minimum scaling ratio of the scaler module is 1/4. If the scaling ratio is less than 1/4, the prescaler is
  // used for scaling. One prescaler is scaled by 1/2.
  while ((out_width_ * scalerTap4 < crop_width) || (out_height_ * scalerTap4 < crop_height)) {
    pre_scaler_num_++;
    crop_width /= yuvCoeffiNum2;
    crop_width = AlignDown(crop_width, yuvCoeffiNum2);
    crop_height /= yuvCoeffiNum2;
    crop_height = AlignDown(crop_height, yuvCoeffiNum2);
  }
  // Each time a prescaler is used, the alignment value needs to be doubled.
  uint32_t align_size = (yuvCoeffiNum2 << pre_scaler_num_);
  crop_width = (right_ - left_ + 1);
  uint32_t gap = crop_width % align_size;
  left_ += AlignDown(gap / yuvCoeffiNum2, yuvCoeffiNum2);
  right_ -= AlignUp(gap / yuvCoeffiNum2, yuvCoeffiNum2);
  crop_width -= gap;

  crop_height = (down_ - up_ + 1);
  gap = crop_height % align_size;
  up_ += AlignDown(gap / yuvCoeffiNum2, yuvCoeffiNum2);
  down_ -= AlignUp(gap / yuvCoeffiNum2, yuvCoeffiNum2);
  crop_height -= gap;

  uint32_t move_step = scalerCoeff - pre_scaler_num_;
  horizon_coeff_ = (crop_width << move_step) / out_width_;
  horizon_bypass_ = (horizon_coeff_ == maxCoeff) ? true : false;
  vertical_coeff_ = (crop_height << move_step) / out_height_;
  vertical_bypass_ = (vertical_coeff_ == maxCoeff) ? true : false;

  half_line_mode_ = false;
  // If the width is less than 2048, the half mode is used.
  if ((vertical_coeff_ >= 0x2aab) && (vertical_coeff_ <= 0x8000) && (out_width_ <= 2048)) {
    half_line_mode_ = true;
  }

  YuvWPara *yuv_scaler_paraset = YuvScalerParaSet::GetInstance();
  YuvScalerPara *scale = yuv_scaler_paraset->scale;
  int32_t index = GetScalerParameterIndex(horizon_coeff_, yuv_scaler_paraset);
  y_horizon_tap_ = scale[index].taps_6;
  uv_horizon_tap_ = scale[index].taps_4;

  index = GetScalerParameterIndex(vertical_coeff_, yuv_scaler_paraset);
  vertical_tap_ = (half_line_mode_) ? scale[index].taps_6 : scale[index].taps_4;
}

void SoftVpc::SetUvValue(int32_t *u_value, int32_t *v_value, int32_t y, int32_t pos) {
  int32_t crop_width = right_ - left_ + 1;
  int32_t in_w_stride = in_width_;
  // 5-order filtering dimension reduction algorithm.
  for (uint32_t i = 0; i < uvReductCoeffNum; i++) {
    int32_t index = pos + i - uvReductCoeffNum / yuvCoeffiNum2;
    if ((index + static_cast<int32_t>(left_) % 0x80) < 0) {
      index = -index;
    }
    if (index > (crop_width - 1)) {
      index = yuvCoeffiNum2 * (crop_width - 1) - index;
    }
    *u_value += in_u_data_[y * in_w_stride + index] * uvReductCoeff[i];
    *v_value += in_v_data_[y * in_w_stride + index] * uvReductCoeff[i];
  }
}

int32_t SoftVpc::Yuv444PackedToYuv422Packed() {
  int32_t in_w_stride = in_width_;
  int32_t crop_width = right_ - left_ + 1;
  int32_t crop_height = down_ - up_ + 1;
  out_width_ = crop_width;
  out_height_ = crop_height;

  out_data_ = new (std::nothrow) uint8_t[out_width_ * out_height_ * yuvCoeffiNum2];
  VPC_CHECK_COND_FAIL_PRINT_RETURN((out_data_ != nullptr), dpFail, "alloc buffer fail.");
  SetYuv422OutBuffer();

  for (int32_t i = 0; i < crop_height; i++) {  // 拷贝y数据
    int32_t ret = memcpy_s(out_y_data_ + i * crop_width, crop_width, in_y_data_ + i * in_w_stride, crop_width);
    VPC_CHECK_COND_FAIL_PRINT_RETURN((ret == dpSucc), dpFail, "memcpy fail.");
  }

  int32_t uv_width = crop_width / yuvCoeffiNum2;
  // Reduces the dimension of the UV data. The 5-order filtering algorithm is used for dimension reduction.
  for (int32_t y = 0; y < crop_height; y++) {
    for (int32_t x = 0; x < uv_width; x++) {
      int32_t pos = static_cast<uint32_t>(x) << 1;
      int32_t u_value = 0;
      int32_t v_value = 0;

      SetUvValue(&u_value, &v_value, y, pos);
      // The most significant eight bits of the dimension reduction result are used.
      u_value = static_cast<uint32_t>(u_value + 0x80) >> 8;
      v_value = static_cast<uint32_t>(v_value + 0x80) >> 8;
      if (u_value > 0xff) u_value = 0xff;
      if (v_value > 0xff) v_value = 0xff;
      out_u_data_[y * uv_width + x] = u_value;
      out_v_data_[y * uv_width + x] = v_value;
    }
  }

  in_format_ = INPUT_YUV422_PLANNER;
  OutputChangeToInput();
  return dpSucc;
}

// For the YUV420 input, the output width and height are reduced by 1/2, the output format is YUV422,
// and the amount of output UV data is reduced by only half.
void SoftVpc::Yuv420PlannerUvPrescaler(uint8_t *(&in_uv_data)[yuvCoeffiNum2], uint8_t *(&out_uv_data)[yuvCoeffiNum2],
                                       uint32_t in_w_stride) {
  for (uint32_t k = 0; k < yuvCoeffiNum2; k++) {
    for (uint32_t i = 0; i < out_height_; i++) {
      for (uint32_t j = 0; j < out_width_ / yuvCoeffiNum2; j++) {  // Zoom out by 1/2
        uint8_t a = in_uv_data[k][i * in_w_stride / yuvCoeffiNum2 + yuvCoeffiNum2 * j];
        uint8_t b = in_uv_data[k][i * in_w_stride / yuvCoeffiNum2 + yuvCoeffiNum2 * j + 1];
        out_uv_data[k][i * out_width_ / yuvCoeffiNum2 + j] = (a + b + 1) / yuvCoeffiNum2;
      }
    }
  }
}

// For the YUV420 input, the output width and height are reduced by 1/2, the output format is YUV422, and the
// amount of output UV data is reduced by 3/4. The prescaler scaling algorithm is a bilinear interpolation
// algorithm. The scaling ratio is 1/2 horizontally and vertically. That is, two horizontal points are combined
// into one point, and two vertical points are combined into one point.
void SoftVpc::Yuv422PackedUvPrescaler(uint8_t *(&in_uv_data)[yuvCoeffiNum2], uint8_t *(&out_uv_data)[yuvCoeffiNum2],
                                      uint32_t in_w_stride) {
  for (uint32_t k = 0; k < yuvCoeffiNum2; k++) {
    for (uint32_t i = 0; i < out_height_; i++) {
      for (uint32_t j = 0; j < out_width_ / yuvCoeffiNum2; j++) {
        uint8_t a = in_uv_data[k][i * in_w_stride + yuvCoeffiNum2 * j];
        uint8_t b = in_uv_data[k][i * in_w_stride + yuvCoeffiNum2 * j + 1];
        uint8_t aa = (a + b + 1) / yuvCoeffiNum2;
        uint8_t c = in_uv_data[k][(yuvCoeffiNum2 * i + 1) * in_w_stride / yuvCoeffiNum2 + yuvCoeffiNum2 * j];
        uint8_t d = in_uv_data[k][(yuvCoeffiNum2 * i + 1) * in_w_stride / yuvCoeffiNum2 + yuvCoeffiNum2 * j + 1];
        uint8_t bb = (c + d + 1) / yuvCoeffiNum2;
        out_uv_data[k][i * out_width_ / yuvCoeffiNum2 + j] = (aa + bb + 1) / yuvCoeffiNum2;
      }
    }
  }
}

void SoftVpc::UvPrescaler() {
  uint32_t in_w_stride = in_width_;
  uint8_t *in_uv_data[yuvCoeffiNum2] = {in_u_data_, in_v_data_};
  uint8_t *out_uv_data[yuvCoeffiNum2] = {out_u_data_, out_v_data_};
  if (in_format_ == INPUT_YUV420_PLANNER) {
    Yuv420PlannerUvPrescaler(in_uv_data, out_uv_data, in_w_stride);
  } else {
    Yuv422PackedUvPrescaler(in_uv_data, out_uv_data, in_w_stride);
  }
}

int32_t SoftVpc::PreScaler() {
  uint32_t in_w_stride = in_width_;
  uint32_t crop_width = right_ - left_ + 1;
  uint32_t crop_height = down_ - up_ + 1;
  out_width_ = crop_width / yuvCoeffiNum2;
  out_height_ = crop_height / yuvCoeffiNum2;
  out_data_ = new (std::nothrow) uint8_t[out_width_ * out_height_ * yuvCoeffiNum2];
  VPC_CHECK_COND_FAIL_PRINT_RETURN((out_data_ != nullptr), dpFail, "alloc buffer fail.");
  SetYuv422OutBuffer();

  // The scaling algorithm of the rescaler is a bilinear interpolation algorithm. The scaling ratio is 1/2
  // horizontally and vertically. That is, two horizontal points are combined into one point,
  // and two vertical points are combined into one point.
  for (uint32_t i = 0; i < out_height_; i++) {
    for (uint32_t j = 0; j < out_width_; j++) {
      uint8_t a = in_y_data_[yuvCoeffiNum2 * i * in_w_stride + yuvCoeffiNum2 * j];
      uint8_t b = in_y_data_[yuvCoeffiNum2 * i * in_w_stride + yuvCoeffiNum2 * j + 1];
      uint8_t aa = (a + b + 1) / yuvCoeffiNum2;
      uint8_t c = in_y_data_[(yuvCoeffiNum2 * i + 1) * in_w_stride + yuvCoeffiNum2 * j];
      uint8_t d = in_y_data_[(yuvCoeffiNum2 * i + 1) * in_w_stride + yuvCoeffiNum2 * j + 1];
      uint8_t bb = (c + d + 1) / yuvCoeffiNum2;
      out_y_data_[i * out_width_ + j] = (aa + bb + 1) / yuvCoeffiNum2;
    }
  }
  UvPrescaler();

  in_format_ = INPUT_YUV422_PLANNER;
  OutputChangeToInput();
  return dpSucc;
}

int32_t SoftVpc::BypassHorizonScaler() {
  uint32_t in_w_stride = in_width_;
  uint32_t crop_width = right_ - left_ + 1;
  uint32_t crop_height = down_ - up_ + 1;
  for (uint32_t i = 0; i < crop_height; i++) {
    int32_t ret = memcpy_s(out_y_data_ + i * crop_width, crop_width, in_y_data_ + i * in_w_stride, crop_width);
    VPC_CHECK_COND_FAIL_PRINT_RETURN((ret == dpSucc), dpFail, "memcpy fail.");
  }

  uint32_t uv_w_stride = in_w_stride / yuvCoeffiNum2;
  uint32_t uv_width = crop_width / yuvCoeffiNum2;

  // The input format is 420. After the format is converted to 422, the UV data is doubled.
  // Therefore, the data needs to be copied twice.
  if (in_format_ == INPUT_YUV420_PLANNER) {
    uint32_t uv_height = crop_height / yuvCoeffiNum2;
    for (uint32_t i = 0; i < uv_height; i++) {
      int32_t ret =
        memcpy_s(out_u_data_ + uv_width * i * yuvCoeffiNum2, uv_width, in_u_data_ + uv_w_stride * i, uv_width);
      VPC_CHECK_COND_FAIL_PRINT_RETURN((ret == dpSucc), dpFail, "memcpy fail.");
      ret =
        memcpy_s(out_u_data_ + uv_width * (i * yuvCoeffiNum2 + 1), uv_width, in_u_data_ + uv_w_stride * i, uv_width);
      VPC_CHECK_COND_FAIL_PRINT_RETURN((ret == dpSucc), dpFail, "memcpy fail.");

      ret = memcpy_s(out_v_data_ + uv_width * i * yuvCoeffiNum2, uv_width, in_v_data_ + uv_w_stride * i, uv_width);
      VPC_CHECK_COND_FAIL_PRINT_RETURN((ret == dpSucc), dpFail, "memcpy fail.");
      ret =
        memcpy_s(out_v_data_ + uv_width * (i * yuvCoeffiNum2 + 1), uv_width, in_v_data_ + uv_w_stride * i, uv_width);
      VPC_CHECK_COND_FAIL_PRINT_RETURN((ret == dpSucc), dpFail, "memcpy fail.");
    }
  } else {
    uint32_t uv_height = crop_height;
    for (uint32_t i = 0; i < uv_height; i++) {
      int32_t ret = memcpy_s(out_u_data_ + uv_width * i, uv_width, in_u_data_ + uv_w_stride * i, uv_width);
      VPC_CHECK_COND_FAIL_PRINT_RETURN((ret == dpSucc), dpFail, "memcpy fail.");
      ret = memcpy_s(out_v_data_ + uv_width * i, uv_width, in_v_data_ + uv_w_stride * i, uv_width);
      VPC_CHECK_COND_FAIL_PRINT_RETURN((ret == dpSucc), dpFail, "memcpy fail.");
    }
  }
  return dpSucc;
}

void SoftVpc::StartHorizonScalerEx(uint32_t width_index, uint32_t tmp_offset, uint8_t *(&in_data)[yuvCoeffiNum3],
                                   uint8_t *(&out_data)[yuvCoeffiNum3]) {
  int16_t *taps[yuvCoeffiNum3] = {y_horizon_tap_, uv_horizon_tap_, uv_horizon_tap_};
  int32_t crop_w = right_ - left_;
  int32_t in_w[yuvCoeffiNum3] = {crop_w, crop_w / scalerTap2, crop_w / scalerTap2};
  uint32_t taps_num[yuvCoeffiNum3] = {scalerTap6, scalerTap4, scalerTap4};
  uint32_t out_w[yuvCoeffiNum3] = {out_width_, out_width_ / yuvCoeffiNum2, out_width_ / yuvCoeffiNum2};
  uint32_t mid_num = (taps_num[width_index] >> 1) - 1;
  uint32_t acc = 0;

  // higher order filter algorithm
  // Map the output position to the input position, calculate the phase based on the input position, and find the
  // corresponding filter (6-order or 4-order filter window) based on the phase.
  // The input data and the filter perform convolution operation to obtain the output data.
  for (uint32_t j = 0; j < out_w[width_index]; j++) {
    uint32_t pos = acc >> bit16Offset;
    uint32_t phase = (acc >> bit13Offset) & low3BitVal;
    int16_t *coeffs = taps[width_index] + taps_num[width_index] * phase;

    int32_t value = 0;
    for (uint32_t k = 0; k < taps_num[width_index]; k++) {  // convolution operation
      int32_t index = pos + k - mid_num;
      index = TruncatedFunc(index, 0, in_w[width_index]);
      int32_t v1 = static_cast<int32_t>(in_data[width_index][tmp_offset + index]);
      int32_t v2 = static_cast<int32_t>(coeffs[k]);
      value += v1 * v2;
    }

    value = TruncatedFunc((value + 0x80), 0, low16BitVal);
    value = static_cast<uint32_t>(value) >> bit8Offset;

    *out_data[width_index]++ = static_cast<uint8_t>(value);
    acc += horizon_coeff_;
  }
  return;
}

void SoftVpc::HorizonScalerEx() {
  uint8_t *in_data[yuvCoeffiNum3] = {in_y_data_, in_u_data_, in_v_data_};
  uint8_t *out_data[yuvCoeffiNum3] = {out_y_data_, out_u_data_, out_v_data_};
  uint32_t in_w_stride[yuvCoeffiNum3] = {in_width_, in_width_ / yuvCoeffiNum2, in_width_ / yuvCoeffiNum2};

  for (uint32_t m = 0; m < yuvCoeffiNum3; m++) {
    for (uint32_t i = 0; i < out_height_; i++) {
      auto tmp_offset = i * in_w_stride[m];  // Offset of each row of data relative to the start position.
      if ((m > 0) && (in_format_ == INPUT_YUV420_PLANNER)) {
        // The width of the UV channel is half of that of the Y channel.
        tmp_offset = i / yuvCoeffiNum2 * in_w_stride[m];
      }
      StartHorizonScalerEx(m, tmp_offset, in_data, out_data);
    }
  }
}

int32_t SoftVpc::HorizonScaler() {
  uint32_t crop_width = right_ - left_ + 1;
  uint32_t crop_height = down_ - up_ + 1;
  out_width_ = (crop_width << scalerCoeff) / horizon_coeff_;
  out_height_ = crop_height;
  out_data_ = new (std::nothrow) uint8_t[out_width_ * out_height_ * yuvCoeffiNum2];
  VPC_CHECK_COND_FAIL_PRINT_RETURN((out_data_ != nullptr), dpFail, "alloc buffer fail.");
  SetYuv422OutBuffer();

  // in bypass mode, the input and output sizes are the same.
  // To be compatible with the YUV420 output, the YUV422 format is used.
  if (horizon_bypass_) {
    int32_t ret = BypassHorizonScaler();
    VPC_CHECK_COND_FAIL_PRINT_RETURN((ret == dpSucc), dpFail, "BypassHorizonScaler fail.");
  } else {
    HorizonScalerEx();
  }

  in_format_ = INPUT_YUV422_PLANNER;
  OutputChangeToInput();
  return dpSucc;
}

void SoftVpc::StartVerticalScaler(uint32_t yuv_index, uint32_t out_w[], uint8_t *(&in_data)[yuvCoeffiNum3],
                                  uint8_t *(&out_data)[yuvCoeffiNum3]) {
  uint32_t num_taps = half_line_mode_ ? scalerTap6 : scalerTap4;
  uint32_t mid_num = (num_taps >> 1) - 1;
  int32_t max_offset = in_height_ - 1;

  // higher order filter algorithm
  // Map the output position to the input position, calculate the phase based on the input position, and find the
  // corresponding filter (6-order or 4-order filter window) based on the phase. The input data and the filter
  // perform convolution operation to obtain the output data.
  for (uint32_t i = 0; i < out_height_; i++) {
    uint32_t acc = i * vertical_coeff_;
    uint32_t pos = acc >> bit16Offset;
    uint32_t phase = (acc >> bit13Offset) & low3BitVal;
    int16_t *coeffs = vertical_tap_ + num_taps * phase;
    for (uint32_t j = 0; j < out_w[yuv_index]; j++) {
      int32_t value = 0;
      for (uint32_t k = 0; k < num_taps; k++) {  // convolution operation
        int32_t index = pos + k - mid_num;
        index = TruncatedFunc(index, 0, max_offset);
        int32_t v1 = in_data[yuv_index][index * out_w[yuv_index] + j];
        int32_t v2 = coeffs[k];
        value += v1 * v2;
      }
      value = TruncatedFunc((value + 0x80), 0, low16BitVal);
      value = static_cast<uint32_t>(value) >> bit8Offset;
      *out_data[yuv_index]++ = static_cast<uint8_t>(value);
    }
  }
  return;
}

int32_t SoftVpc::VerticalScaler() {
  out_width_ = in_width_;
  out_height_ = (in_height_ << scalerCoeff) / vertical_coeff_;
  out_data_ = new (std::nothrow) uint8_t[out_width_ * out_height_ * yuvCoeffiNum2];
  VPC_CHECK_COND_FAIL_PRINT_RETURN((out_data_ != nullptr), dpFail, "alloc buffer fail.");
  SetYuv422OutBuffer();

  uint8_t *in_data[yuvCoeffiNum3] = {in_y_data_, in_u_data_, in_v_data_};
  uint8_t *out_data[yuvCoeffiNum3] = {out_y_data_, out_u_data_, out_v_data_};
  uint32_t out_w[yuvCoeffiNum3] = {out_width_, out_width_ / yuvCoeffiNum2, out_width_ / yuvCoeffiNum2};
  for (uint32_t m = 0; m < yuvCoeffiNum3; m++) {
    StartVerticalScaler(m, out_w, in_data, out_data);
  }

  OutputChangeToInput();
  return dpSucc;
}

// yuv scalser is core scaler, The high-order filtering and scaling algorithm is used.
int32_t SoftVpc::YuvScaler() {
  int32_t ret = HorizonScaler();
  VPC_CHECK_COND_FAIL_PRINT_RETURN((ret == dpSucc), dpFail, "HorizonScaler fail.");
  if (!vertical_bypass_) {
    ret = VerticalScaler();
    VPC_CHECK_COND_FAIL_PRINT_RETURN((ret == dpSucc), dpFail, "VerticalScaler fail.");
  }
  return ret;
}

int32_t SoftVpc::ChipProcess() {
  ChipPreProcess();
  // Determine whether dimension reduction is required.
  if (in_format_ == INPUT_YUV444_PLANNER) {
    VPC_CHECK_COND_FAIL_PRINT_RETURN((Yuv444PackedToYuv422Packed() == dpSucc), dpFail,
                                     "Yuv444PackedToYuv422Packed fail.");
  }

  // Analog chip PreScaler function
  for (uint32_t i = 0; i < pre_scaler_num_; i++) {
    VPC_CHECK_COND_FAIL_PRINT_RETURN((PreScaler() == dpSucc), dpFail, "PreScaler fail.");
  }

  // Analog chip Yuv Scaler function
  VPC_CHECK_COND_FAIL_PRINT_RETURN((YuvScaler() == dpSucc), dpFail, "YuvScaler fail.");
  return dpSucc;
}

void SoftVpc::YuvToRgb() {
  uint8_t *out_data = out_info_.addr;
  int32_t yy, uu, vv;
  int32_t rr, gg, bb;
  for (uint32_t j = 0; j < in_height_; j++) {
    for (uint32_t i = 0; i < in_width_; i++) {
      yy = in_y_data_[(j * in_width_) + i];
      uu = in_u_data_[((j - (j % num2)) * (in_width_ / yuvCoeffiNum2)) + (i / yuvCoeffiNum2)];
      vv = in_v_data_[((j - (j % num2)) * (in_width_ / yuvCoeffiNum2)) + (i / yuvCoeffiNum2)];

      // yuv convert rgb formula
      rr = ((yy - rtAippYuv2RgbCscInputBias0) * rtAippYuv2RgbCscMatrixR0c0 +
            (uu - rtAippYuv2RgbCscInputBias1) * rtAippYuv2RgbCscMatrixR0c1 +
            (vv - rtAippYuv2RgbCscInputBias2) * rtAippYuv2RgbCscMatrixR0c2) /
           rtAippConverCoeffi;
      gg = ((yy - rtAippYuv2RgbCscInputBias0) * rtAippYuv2RgbCscMatrixR1c0 +
            (uu - rtAippYuv2RgbCscInputBias1) * rtAippYuv2RgbCscMatrixR1c1 +
            (vv - rtAippYuv2RgbCscInputBias2) * rtAippYuv2RgbCscMatrixR1c2) /
           rtAippConverCoeffi;
      bb = ((yy - rtAippYuv2RgbCscInputBias0) * rtAippYuv2RgbCscMatrixR2c0 +
            (uu - rtAippYuv2RgbCscInputBias1) * rtAippYuv2RgbCscMatrixR2c1 +
            (vv - rtAippYuv2RgbCscInputBias2) * rtAippYuv2RgbCscMatrixR2c2) /
           rtAippConverCoeffi;

      *out_data++ = (rr < 0) ? 0 : ((rr < 0xff) ? rr : 0xff);
      *out_data++ = (gg < 0) ? 0 : ((gg < 0xff) ? gg : 0xff);
      *out_data++ = (bb < 0) ? 0 : ((bb < 0xff) ? bb : 0xff);
    }
  }

  delete[] in_data_;
  in_data_ = nullptr;
}

int32_t SoftVpc::Process(VpcInfo input, const SoftDpCropInfo crop, const VpcInfo output) {
  Init(input, crop, output);
  int32_t ret = CheckParamter();
  if (ret != dpSucc) {
    delete[] input.addr;
    input.addr = nullptr;
    return ret;
  }

  BuildResizeStack();
  while (!resize_stack_.empty()) {
    ResizeUnit &unit = resize_stack_.top();
    resize_stack_.pop();

    out_width_ = unit.out_width;
    out_height_ = unit.out_height;

    ret = ChipProcess();
    VPC_CHECK_COND_FAIL_PRINT_RETURN((ret == dpSucc), dpFail, "ChipProcess fail.");

    if (!resize_stack_.empty()) {
      ret = Yuv422pToYuv420p();
      VPC_CHECK_COND_FAIL_PRINT_RETURN((ret == dpSucc), dpFail, "Yuv422pToYuv420p fail.");
    }
  }
  YuvToRgb();

  return dpSucc;
}
