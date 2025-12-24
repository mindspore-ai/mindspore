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

#ifndef MINDSPORE_CORE_OPS_DATA_FORMAT_VEC_PERMUTE_H_
#define MINDSPORE_CORE_OPS_DATA_FORMAT_VEC_PERMUTE_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameDataFormatVecPermute = "DataFormatVecPermute";
/// \brief Permute input tensor from src_format to dst_format.
/// Refer to Python API @ref mindspore.ops.DataFormatVecPermute for more details.
class OPS_API DataFormatVecPermute : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(DataFormatVecPermute);
  /// \brief Constructor.
  DataFormatVecPermute() : BaseOperator(kNameDataFormatVecPermute) { InitIOName({"x"}, {"y"}); }
  /// \brief Init.
  void Init(const std::string &src_format = "NHWC", const std::string &dst_format = "NCHW");
  /// \brief Set src_format.
  void set_src_format(const std::string &src_format);
  /// \brief Set dst_format.
  void set_dst_format(const std::string &dst_format);
  /// \brief Get src_format.
  std::string get_src_format() const;
  /// \brief Get dst_format.
  std::string get_dst_format() const;
};

OPS_API abstract::AbstractBasePtr DataFormatVecPermuteInfer(const abstract::AnalysisEnginePtr &,
                                                            const PrimitivePtr &primitive,
                                                            const std::vector<abstract::AbstractBasePtr> &input_args);

using PrimDataFormatVecPermutePtr = std::shared_ptr<DataFormatVecPermute>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_DATA_FORMAT_VEC_PERMUTE_H_
