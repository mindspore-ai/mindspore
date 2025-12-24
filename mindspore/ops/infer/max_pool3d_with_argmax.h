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

#ifndef MINDSPORE_CORE_OPS_MAX_POOL3D_WITH_ARGMAX_H_
#define MINDSPORE_CORE_OPS_MAX_POOL3D_WITH_ARGMAX_H_
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "ops/base_operator.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMaxPool3DWithArgmax = "MaxPool3DWithArgmax";
/// \brief 3D Max pooling with argmax operation.
/// Refer to Python API @ref mindspore.ops.MaxPool3DWithArgmax for more details.
class OPS_API MaxPool3DWithArgmax : public BaseOperator {
 public:
  /// \brief Constructor.
  MIND_API_BASE_MEMBER(MaxPool3DWithArgmax);
  /// \brief Destructor.
  MaxPool3DWithArgmax() : BaseOperator(kNameMaxPool3DWithArgmax) {
    InitIOName({"input"}, {"output_y", "output_argmax"});
  }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.MaxPool3DWithArgmax for the inputs.
  void Init(const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &strides,
            const std::vector<int64_t> &pads, const std::vector<int64_t> &dialtion = {1, 1, 1}, bool ceil_mode = false,
            const Format &format = NCDHW, const TypeId &argmax_type = kNumberTypeInt64);

  void set_kernel_size(const std::vector<int64_t> &kernel_size);
  void set_strides(const std::vector<int64_t> &strides);
  void set_pads(const std::vector<int64_t> &pads);
  void set_dilation(const std::vector<int64_t> &dialtion);
  void set_ceil_mode(bool ceil_mode);
  void set_format(const Format &format);
  void set_argmax_type(const TypeId &argmax_type);

  std::vector<int64_t> get_kernel_size() const;
  std::vector<int64_t> get_strides() const;
  std::vector<int64_t> get_pads() const;
  std::vector<int64_t> get_dilation() const;
  bool get_ceil_mode() const;
  Format get_format() const;
  TypeId get_argmax_type() const;
};

AbstractBasePtr MaxPool3DWithArgmaxInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args);
using PrimMaxPool3DWithArgmaxPtr = std::shared_ptr<MaxPool3DWithArgmax>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MAX_POOL3D_WITH_ARGMAX_H_
