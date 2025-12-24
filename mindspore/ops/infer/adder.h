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

#ifndef MINDSPORE_CORE_OPS_ADDER_H_
#define MINDSPORE_CORE_OPS_ADDER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/format.h"
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAdder = "Adder";
/// \brief All defined All operator prototype of lite.
class OPS_API Adder : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Adder);
  /// \brief Constructor.
  explicit Adder(const std::string &k_name = kNameAdder) : BaseOperator(k_name) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] in_channel Define the input channel.
  /// \param[in] out_channel Define the output channel.
  /// \param[in] kernel_size Define the kernel size.
  /// \param[in] pad_mode Define the pad mode.
  /// \param[in] stride Define the stride.
  /// \param[in] pad_list Define the pad list.
  /// \param[in] dilation Define the dilation.
  /// \param[in] group Define the group.
  /// \param[in] format Define the format.
  void Init(const int64_t in_channel, const int64_t out_channel, const std::vector<int64_t> &kernel_size,
            const PadMode &pad_mode, const std::vector<int64_t> &stride, const std::vector<int64_t> &pad_list,
            const std::vector<int64_t> &dilation, const int64_t group, const Format &format);

  /// \brief Method to set in_channel attributes.
  ///
  /// \param[in] in_channel Define the input channel.
  void set_in_channel(const int64_t in_channel);

  /// \brief Method to set out_channel attributes.
  ///
  /// \param[in] out_channel Define the output channel.
  void set_out_channel(const int64_t out_channel);

  /// \brief Method to set kernel_size attributes.
  ///
  /// \param[in] kernel_size Define the kernel size.
  void set_kernel_size(const std::vector<int64_t> &kernel_size);

  /// \brief Method to set pad_mode attributes.
  ///
  /// \param[in] pad_mode Define the pad mode.
  void set_pad_mode(const PadMode &pad_mode);

  /// \brief Method to set stride attributes.
  ///
  /// \param[in] stride Define the stride.
  void set_stride(const std::vector<int64_t> &stride);

  /// \brief Method to set pad_list attributes.
  ///
  /// \param[in] pad_list Define the pad list.
  void set_pad_list(const std::vector<int64_t> &pad_list);

  /// \brief Method to set dilation attributes.
  ///
  /// \param[in] dilation Define the dilation.
  void set_dilation(const std::vector<int64_t> &dilation);

  /// \brief Method to set group attributes.
  ///
  /// \param[in] group Define the group.
  void set_group(const int64_t group);

  /// \brief Method to set format attributes.
  ///
  /// \param[in] format Define the format.
  void set_format(const Format &format);

  /// \brief Method to get in_channel attributes.
  ///
  /// \return in_channel attributes.
  int64_t get_in_channel() const;

  /// \brief Method to get out_channel attributes.
  ///
  /// \return out_channel attributes.
  int64_t get_out_channel() const;

  /// \brief Method to get kernel_size attributes.
  ///
  /// \return kernel_size attributes.
  std::vector<int64_t> get_kernel_size() const;

  /// \brief Method to get pad_mode attributes.
  ///
  /// \return pad_mode attributes.
  PadMode get_pad_mode() const;

  /// \brief Method to get stride attributes.
  ///
  /// \return stride attributes.
  std::vector<int64_t> get_stride() const;

  /// \brief Method to get pad_list attributes.
  ///
  /// \return pad_list attributes.
  std::vector<int64_t> get_pad_list() const;

  /// \brief Method to get dilation attributes.
  ///
  /// \return dilation attributes.
  std::vector<int64_t> get_dilation() const;

  /// \brief Method to get group attributes.
  ///
  /// \return group attributes.
  int64_t get_group() const;

  /// \brief Method to get format attributes.
  ///
  /// \return format attributes.
  Format get_format() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ADDER_H_
