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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_TENSOR_HELPERS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_TENSOR_HELPERS_H_

#include <memory>
#include <vector>

#include "minddata/dataset/include/constants.h"

namespace mindspore {
namespace dataset {
class Slice {
 public:
  Slice() : start_(0), stop_(0), step_(0) {}
  Slice(dsize_t start, dsize_t stop, dsize_t step) : start_(start), stop_(stop), step_(step) {}
  Slice(dsize_t start, dsize_t stop) : start_(start), stop_(stop), step_(1) {}
  explicit Slice(dsize_t stop) : start_(0), stop_(stop), step_(1) {}
  Slice(Slice const &slice) = default;

  ~Slice() = default;

  bool valid() const { return step_ != 0; }
  dsize_t start_;
  dsize_t stop_;
  dsize_t step_;
};

class SliceOption {
 public:
  explicit SliceOption(bool all) : all_(all) {}
  explicit SliceOption(std::vector<dsize_t> indices) : indices_(indices) {}
  explicit SliceOption(Slice slice) : slice_(slice) {}
  SliceOption(SliceOption const &slice) = default;

  ~SliceOption() = default;

  // only one of the following will be valid
  // given indices to slice the Tensor.
  std::vector<dsize_t> indices_ = {};
  // Slice object. All start, stop and step are 0 if invalid.
  Slice slice_;
  bool all_ = false;
};

/// Recursive helper function to generate indices based on vector of SliceOptions. It recursively iterates through each
/// range represented by slice_options to generate a list of indices to be sliced.
/// \param[out] matrix Generated nested vector of indices
///       Example: For a 4 x 2 tensor, and with slice_list = {SliceOption({0})} (the first row), matrix will become
///       {{0}}. For slice_list = {SliceOption(all), SliceOption({0})} (the first column), matrix will become
///       {{0, 0}, {1, 0}, {2, 0}, {3, 0}}.
///       For slice_list = {SliceOption({0, 2})}, matrix will become {{0}, {2}}. The size of each nested array is always
///       equal to (slice_list).size().
/// \param[in] depth used to keep track of recursion level
/// \param[in] numbers vector used to represent current index
/// \param[in] matrix 2D vector to be populated with desired indices
/// \param[in] slice_options vector of SliceOption objects
void IndexGeneratorHelper(int8_t depth, std::vector<dsize_t> *numbers, const std::vector<SliceOption> &slice_list,
                          std::vector<std::vector<dsize_t>> *matrix);

/// Generate indices based on vector of SliceOptions
/// Calls the recursive helper function IndexGeneratorHelper
/// \param[in] slice_list vector of SliceOption objects. Note: If the user passes
///       {SliceOption(true), SliceOption(true)}, it will return a M x 2 vector, instead of reducing it to
///       {SliceOption(true)} first to only generate a M x 1 vector.
/// \return std::vector<std::vector<dsize_t>> 2D vector of generated indices, M x (slice_list).size()
std::vector<std::vector<dsize_t>> IndexGenerator(const std::vector<SliceOption> &slice_list);
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_TENSOR_HELPERS_H_
