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

#ifndef MINDSPORE_CORE_OPS_FSE_DECODER_H_
#define MINDSPORE_CORE_OPS_FSE_DECODER_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameFSEDecode = "FSEDecode";
/// \brief FSEDecode FSEDecode the FSEDecode operator prototype.
class OPS_API FSEDecode : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(FSEDecode);
  /// \brief Constructor.
  FSEDecode() : BaseOperator(kNameFSEDecode) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] dst_t Define the data type of output.
  void Init(const int64_t dst_t, const int64_t curr_chunk, const int64_t curr_chunk_index, const int64_t curr_bit_count,
            const int64_t table_log);

  /// \brief Method to set dst_t attribute.
  ///
  /// \param[in] dst_t Define the data type of output.
  void set_dst_t(const int64_t dst_t);

  /// \brief Method to get dst_t attribute.
  ///
  /// \return the data type of output.
  int64_t get_dst_t() const;

  /// \brief Method to set curr_chunk attribute.
  ///
  /// \param[in] curr_chunk Define the curr_chunk attribute.
  void set_curr_chunk(const int64_t curr_chunk);

  /// \brief Method to get curr_chunk attribute.
  ///
  /// \return the curr_chunk attribute.
  int64_t get_curr_chunk() const;

  /// \brief Method to set curr_chunk_index attribute.
  ///
  /// \param[in] curr_chunk_index Define the curr_chunk_index attribute.
  void set_curr_chunk_index(const int64_t curr_chunk_index);

  /// \brief Method to get curr_chunk_index attribute.
  ///
  /// \return the curr_chunk_index attribute.
  int64_t get_curr_chunk_index() const;

  /// \brief Method to set curr_bit_count attribute.
  ///
  /// \param[in] curr_bit_count Define the curr_bit_count attribute..
  void set_curr_bit_count(const int64_t curr_bit_count);

  /// \brief Method to get curr_bit_count attribute.
  ///
  /// \return the curr_bit_count attribute.
  int64_t get_curr_bit_count() const;

  /// \brief Method to set table_log attribute.
  ///
  /// \param[in] table_log Define the table_log attribute.
  void set_table_log(const int64_t table_log);

  /// \brief Method to get table_log attribute.
  ///
  /// \return the table_log attribute.
  int64_t get_table_log() const;
};
abstract::AbstractBasePtr FSEDecodeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_FSE_DECODER_H_
