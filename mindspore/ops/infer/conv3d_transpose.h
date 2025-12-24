/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_CONV3D_TRANSPOSE_H_
#define MINDSPORE_CORE_OPS_CONV3D_TRANSPOSE_H_

#include <memory>
#include <vector>

#include "mindapi/base/format.h"
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameConv3DTranspose = "Conv3DTranspose";

class OPS_API Conv3DTranspose : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Conv3DTranspose);
  /// \brief Constructor.
  Conv3DTranspose() : BaseOperator(kNameConv3DTranspose) { InitIOName({"x", "w"}, {"output"}); }

  void Init(int64_t in_channel, int64_t out_channel, const std::vector<int64_t> &kernel_size, int64_t mode = 1,
            const PadMode &pad_mode = VALID, const std::vector<int64_t> &pad = {0, 0, 0, 0, 0, 0},
            const std::vector<int64_t> &stride = {1, 1, 1, 1, 1},
            const std::vector<int64_t> &dilation = {1, 1, 1, 1, 1}, int64_t group = 1,
            const std::vector<int64_t> &output_padding = {0, 0, 0, 0, 0}, const Format &format = NCDHW);
  /// \brief Set in_channel.
  void set_in_channel(int64_t in_channel);
  /// \brief Set out_channel.
  void set_out_channel(int64_t out_channel);
  /// \brief Set kernel_size.
  virtual void set_kernel_size(const std::vector<int64_t> &kernel_size);
  /// \brief Set stride.
  void set_stride(const std::vector<int64_t> &stride);
  /// \brief Set dilation.
  virtual void set_dilation(const std::vector<int64_t> &dilation);
  /// \brief Set pad_mode.
  void set_pad_mode(const PadMode &pad_mode);
  /// \brief Set pad.
  void set_pad(const std::vector<int64_t> &pad);
  /// \brief Set mode.
  void set_mode(int64_t mode);
  /// \brief Set group.
  void set_group(int64_t group);
  /// \brief Set output_padding.
  void set_output_padding(const std::vector<int64_t> &pad);
  /// \brief Set data_format.
  void set_data_format(const Format &format);

  /// \brief Get in_channel.
  ///
  /// \return in_channel.
  int64_t get_in_channel() const;
  /// \brief Get out_channel.
  ///
  /// \return out_channel.
  int64_t get_out_channel() const;
  /// \brief Get kernel_size.
  ///
  /// \return kernel_size.
  std::vector<int64_t> get_kernel_size() const;
  /// \brief Get stride.
  ///
  /// \return stride.
  std::vector<int64_t> get_stride() const;
  /// \brief Get dilation.
  ///
  /// \return dilation.
  std::vector<int64_t> get_dilation() const;
  /// \brief Get pad_mode.
  ///
  /// \return pad_mode.
  PadMode get_pad_mode() const;
  /// \brief Get pad.
  ///
  /// \return pad.
  std::vector<int64_t> get_pad() const;
  /// \brief Get mode.
  ///
  /// \return mode.
  int64_t get_mode() const;
  /// \brief Get group.
  ///
  /// \return group.
  int64_t get_group() const;
  /// \brief Get data_format.
  ///
  /// \return data_format.
  Format get_data_format() const;
  /// \brief Get output_padding.
  ///
  /// \return output_padding.
  std::vector<int64_t> get_output_padding() const;
};
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_CONV3D_TRANSPOSE_H_
