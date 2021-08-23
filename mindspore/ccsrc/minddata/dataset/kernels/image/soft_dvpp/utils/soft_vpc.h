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

#ifndef SOFT_VPC_H
#define SOFT_VPC_H

#include "minddata/dataset/kernels/image/soft_dvpp/utils/soft_dp.h"
#include <stack>

constexpr uint32_t yuvCoeffiNum2 = 2;
constexpr uint32_t yuvCoeffiNum3 = 3;

struct ResizeUnit {
  uint32_t in_width;
  uint32_t in_height;
  uint32_t out_width;
  uint32_t out_height;
};

class SoftVpc {
 public:
  SoftVpc();

  ~SoftVpc() = default;

  /*
   * @brief : vpc Cropping and Scaling APIs.
   * @param [in] VpcInfo input : Structure input to the VPC for processing.
   * @param [in] SoftDpCropInfo crop : crop struct.
   * @param [in] VpcInfo output : vpc output struct.
   * @return : dpSucc:vpc process succ，dpFail:vpc process failed.
   */
  int32_t Process(VpcInfo input, SoftDpCropInfo crop, VpcInfo output);

 private:
  enum JpegdToVpcFormat in_format_;
  uint32_t in_width_;
  uint32_t in_height_;
  uint8_t *in_data_;
  uint8_t *in_y_data_;
  uint8_t *in_u_data_;
  uint8_t *in_v_data_;

  // crop area
  uint32_t left_;
  uint32_t right_;
  uint32_t up_;
  uint32_t down_;

  // output config
  uint32_t out_width_;
  uint32_t out_height_;
  uint8_t *out_data_;
  uint8_t *out_y_data_;
  uint8_t *out_u_data_;
  uint8_t *out_v_data_;

  // resize config
  uint32_t pre_scaler_num_;

  // If the image is amplified by 2x or more and the output width is less than 2048 pixels,
  // the half-line mode is required.
  bool half_line_mode_;
  uint32_t horizon_coeff_;   // Horizontal scaling coefficient
  uint32_t vertical_coeff_;  // Vertical scaling coefficient.
  bool horizon_bypass_;
  bool vertical_bypass_;
  int16_t *y_horizon_tap_;   // Filtering coefficients for horizontal scaling of channel y
  int16_t *uv_horizon_tap_;  // Filtering coefficients of the horizontal scaling UV channel.
  int16_t *vertical_tap_;    // Filtering coefficient table for vertical scaling. Y and UV signals share the same table.

  // Scaling unit stack, used to store the input and output information processed by the chip at a time.
  std::stack<ResizeUnit> resize_stack_;
  VpcInfo in_info_;   // Original input information.
  VpcInfo out_info_;  // Original output information.

  /*
   * @brief : set output format is YUV422
   */
  void SetYuv422OutBuffer();

  /*
   * @brief : check params
   * @return : dpSucc:check succ， dpFail:check failed.
   */
  int32_t CheckParamter();

  /*
   * @brief : init vpc output info struct
   * @param [in] VpcInfo input : Structure input to the VPC for processing.
   * @param [in] SoftDpCropInfo crop : crop struct.
   * @param [in] VpcInfo output : vpc output struct.
   */
  void Init(VpcInfo input, SoftDpCropInfo crop, VpcInfo output);

  void OutputChangeToInput();

  /*
   * @brief : For the tasks that cannot be processed by the chip at a time, split the tasks whose scaling
   *          coefficients in the horizontal direction are greater than those in the vertical direction.
   * @param [in] ResizeUnit *pre_unit : input resize unit.
   * @param [in] ResizeUnit *can_process_unit : chip can process resize unit.
   */
  void HorizonSplit(ResizeUnit *pre_unit, ResizeUnit *can_process_unit);

  /*
   * @brief :  For the tasks that cannot be processed by the chip at a time, split the tasks whose vertical scaling
   *           coefficients are greater than the horizontal scaling coefficients.
   * @param [in] ResizeUnit *pre_unit : input resize unit.
   * @param [in] ResizeUnit *can_process_unit : chip can process resize unit.
   */
  void VerticalSplit(ResizeUnit *pre_unit, ResizeUnit *can_process_unit);

  /*
   * @brief : Check whether the VPC chip can complete the processing at a time based on the input and output sizes.
   * @param [in] const ResizeUnit& pre_unit : input resize unit.
   * @return : true:vpc process succ， false:vpc process failed.
   */
  bool CanVpcChipProcess(const ResizeUnit &pre_unit);

  /*
   * @brief : Creates a scaling parameter stack based on the user input and output information. The elements
   *          in the stack are the input and output information. The input and output information stores the
   *          scaling information task.
   */
  void BuildResizeStack();

  /*
   * @brief : YUV422 planner format convert YUV420 format
   * @return : dpSucc: downsampling success, dpFail: downsampling failed
   */
  int32_t Yuv422pToYuv420p();

  /*
   * @brief : Preprocesses the chip, calculates the number of chip prescalers, and adjusts the cropping area based on
   *          the input and output information.
   */
  void ChipPreProcess();

  /*
   * @brief : when YUV444 packed format convert YUV422 packed, Calculate the conversion of UV.
   * @param [in] int32_t *u_value : u value.
   * @param [in] int32_t *v_value : v value.
   * @param [in] int32_t y :y value.
   * @param [in] int32_t pos :
   */
  void SetUvValue(int32_t *u_value, int32_t *v_value, int32_t y, int32_t pos);

  /*
   * @brief : YUV444 packed convert YUV422 packed.
   * @return : dpSucc:Downsampling succ， dpFail:Downsampling failed.
   */
  int32_t Yuv444PackedToYuv422Packed();

  /*
   * @brief : Pre-scaling the UV image.
   */
  void UvPrescaler();

  /*
   * @brief : Prescaling the UV in YUV420 format.
   * @param [in] uint8_t* (&in_uv_data)[yuvCoeffiNum2] : input uv data
   * @param [in] uint8_t* (&out_uv_data)[yuvCoeffiNum2] : output uv data
   * @param [in] uint32_t in_w_stride : input stride
   */
  void Yuv420PlannerUvPrescaler(uint8_t *(&in_uv_data)[yuvCoeffiNum2], uint8_t *(&out_uv_data)[yuvCoeffiNum2],
                                uint32_t in_w_stride);

  /*
   * @brief : Prescaling the UV in YUV422 format.
   * @param [in] uint8_t* (&in_uv_data)[yuvCoeffiNum2] : input uv data
   * @param [in] uint8_t* (&out_uv_data)[yuvCoeffiNum2]: output uv data
   * @param [in] uint32_t in_w_stride : input stride
   */
  void Yuv422PackedUvPrescaler(uint8_t *(&in_uv_data)[yuvCoeffiNum2], uint8_t *(&out_uv_data)[yuvCoeffiNum2],
                               uint32_t in_w_stride);

  /*
   * @brief : Chip prescaler processing.
   */
  int32_t PreScaler();

  /*
   * @brief : Horizontal scaling bypass.
   */
  int32_t BypassHorizonScaler();

  /*
   * @brief : Single-channel horizontal scaling of the chip.
   * @param [in] uint32_t width_index : index of output width array.
   * @param [in] uint32_t tmp_offset : Offset of each row of data relative to the start position.
   * @param [in] uint8_t* (&in_data)[yuvCoeffiNum3] : input y,u,v data array.
   * @param [in] uint8_t* (&out_data)[yuvCoeffiNum3] : output y,u,v data array.
   */
  void StartHorizonScalerEx(uint32_t width_index, uint32_t tmp_offset, uint8_t *(&in_data)[yuvCoeffiNum3],
                            uint8_t *(&out_data)[yuvCoeffiNum3]);

  /*
   * @brief : Horizontal scaling.
   */
  void HorizonScalerEx();

  /*
   * @brief : Horizontal scaling.
   * @return : dpSucc : Horizontal scaling succ， dpFail:Horizontal scaling failed.
   */
  int32_t HorizonScaler();

  /*
   * @brief : start Vertical scaling.
   * @param [in] uint32_t yuv_index : index of output width array.
   * @param [in] uint32_t out_w[] : output width array.
   * @param [in] uint8_t* (&in_data)[yuvCoeffiNum3] : input y,u,v data array.
   * @param [in] uint8_t* (&out_data)[yuvCoeffiNum3] : output y,u,v data array.
   */
  void StartVerticalScaler(uint32_t yuv_index, uint32_t out_w[], uint8_t *(&in_data)[yuvCoeffiNum3],
                           uint8_t *(&out_data)[yuvCoeffiNum3]);

  /*
   * @brief : Vertical scaling
   * @return : dpSucc : Vertical scaling succ， dpFail : Vertical scaling failed.
   */
  int32_t VerticalScaler();

  /*
   * @brief : Yuv Scaler Horizontal scaling and vertical scaling.
   * @return : dpSucc:yuv scaler succ. dpFail:yuv scaler failed.
   */
  int32_t YuvScaler();

  /*
   * @brief : Software Implementation of the Simulation Chip PreScaler and Yuv Scaler function.
   * @return : dpSucc : Analog chip scaling succ， dpFail: Analog chip scaling failed.
   */
  int32_t ChipProcess();

  /*
   * @brief : YUV planner convert RGB format.
   */
  void YuvToRgb();
};

#endif  // SOFT_VPC_H
