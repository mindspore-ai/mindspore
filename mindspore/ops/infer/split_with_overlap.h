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

#ifndef MINDSPORE_CORE_OPS_SPLIT_WITH_OVERLAP_H_
#define MINDSPORE_CORE_OPS_SPLIT_WITH_OVERLAP_H_
#include <memory>
#include <vector>
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSplitWithOverlap = "SplitWithOverlap";
/// \brief All defined All operator prototype of lite.
class OPS_API SplitWithOverlap : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SplitWithOverlap);
  /// \brief Constructor.
  SplitWithOverlap() : BaseOperator(kNameSplitWithOverlap) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] number_split Define the number split.
  /// \param[in] ratio Define the ratio.
  /// \param[in] extend_top Define the extend top.
  /// \param[in] extend_bottom Define the extend bottom.
  /// \param[in] split_dim Define the split dim.
  /// \param[in] stride Define the pad stride.
  /// \param[in] pad_top Define the pad top.
  /// \param[in] trans_format Define the trans format.
  void Init(int64_t number_split, const std::vector<int64_t> &ratio, const std::vector<int64_t> &extend_top,
            const std::vector<int64_t> &extend_bottom, int64_t split_dim, int64_t stride, int64_t pad_top,
            bool trans_format);

  /// \brief Method to set ratio attributes.
  ///
  /// \param[in] ratio Define the ratio.
  void set_ratio(const std::vector<int64_t> &ratio);

  /// \brief Method to set extend_top attributes.
  ///
  /// \param[in] extend_top Define the extend top.
  void set_extend_top(const std::vector<int64_t> &extend_top);

  /// \brief Method to set extend_bottom attributes.
  ///
  /// \param[in] extend_bottom Define the extend bottom.
  void set_extend_bottom(const std::vector<int64_t> &extend_bottom);

  /// \brief Method to set number_split attributes.
  ///
  /// \param[in] number_split Define the number split.
  void set_number_split(int64_t number_split);

  /// \brief Method to set split_dim attributes.
  ///
  /// \param[in] split_dim Define the split dim.
  void set_split_dim(int64_t split_dim);

  /// \brief Method to set stride attributes.
  ///
  /// \param[in] stride Define the stride.
  void set_split_stride(int64_t stride);

  /// \brief Method to set pad_top attributes.
  ///
  /// \param[in] pad_top Define the pad top.
  void set_pad_top(int64_t pad_top);

  /// \brief Method to set trans_format attributes.
  ///
  /// \param[in] trans_format Define the trans format.
  void set_trans_format(bool trans_format);

  /// \brief Method to get ratio attributes.
  ///
  /// \return ratio attributes.
  std::vector<int64_t> get_ratio() const;

  /// \brief Method to get extend_top attributes.
  ///
  /// \return extend_top attributes.
  std::vector<int64_t> get_extend_top() const;

  /// \brief Method to get extend_bottom attributes.
  ///
  /// \return extend_bottom attributes.
  std::vector<int64_t> get_extend_bottom() const;

  /// \brief Method to get number_split attributes.
  ///
  /// \return number_split attributes.
  int64_t get_number_split() const;

  /// \brief Method to get split_dim attributes.
  ///
  /// \return split_dim attributes.
  int64_t get_split_dim() const;

  /// \brief Method to get split_stride attributes.
  ///
  /// \return split_stride attributes.
  int64_t get_split_stride() const;

  /// \brief Method to get pad_top attributes.
  ///
  /// \return pad_top attributes.
  int64_t get_pad_top() const;

  /// \brief Method to get trans_format attributes.
  ///
  /// \return trans_format attributes.
  bool get_trans_format() const;
};
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_SPLIT_WITH_OVERLAP_H_
