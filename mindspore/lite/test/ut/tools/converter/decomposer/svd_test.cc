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

#include <vector>
#include "common/common_test.h"
#include "tools/converter/decomposer/svd_matrix.h"
#include "include/errorcode.h"

using mindspore::lite::decomposer::SVDMatrix;

namespace mindspore {
namespace lite {
class SVDTest : public mindspore::CommonTest {
 public:
  SVDTest() = default;
  float Reconsruct_err(std::vector<float> *const origin_ptr, std::vector<float> *const mat_a_ptr,
                       std::vector<float> *const mat_b_ptr, const std::vector<int> &origin_shape,
                       const std::vector<int> &mat_a_shape, const std::vector<int> &mat_b_shape) {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> origin_matrix =
      Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(origin_ptr->data(), origin_ptr->size());
    origin_matrix.resize(origin_shape[0], origin_shape[1]);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a_m =
      Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(mat_a_ptr->data(), mat_a_ptr->size());
    a_m.resize(mat_a_shape[0], mat_a_shape[1]);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> b_m =
      Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(mat_b_ptr->data(), mat_b_ptr->size());
    b_m.resize(mat_b_shape[0], mat_b_shape[1]);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> reconsruct = a_m * b_m;
    float err = (reconsruct - origin_matrix).norm();
    return err;
  }
};

TEST_F(SVDTest, TestSVD) {
  std::vector<int> shape = {7, 5};
  std::vector<float> data;
  for (int i = 0; i < shape[0] * shape[1]; i++) {
    data.push_back(i + 0.2);
  }
  SVDMatrix svd(data.data(), data.size(), shape);

  int status = svd.CompressWithRank(3);
  ASSERT_EQ(status, RET_OK);
  std::vector<float> mat_a_rank = svd.GetMatA();
  std::vector<float> mat_b_rank = svd.GetMatB();
  std::vector<int> mat_a_rank_shape = svd.GetMatShapeA();
  std::vector<int> mat_b_rank_shape = svd.GetMatShapeB();
  float err = Reconsruct_err(&data, &mat_a_rank, &mat_b_rank, shape, mat_a_rank_shape, mat_b_rank_shape);
  ASSERT_LE(err, 0.001);
  ASSERT_EQ(mat_a_rank_shape[1], mat_b_rank_shape[0]);
  ASSERT_EQ(mat_a_rank.size(), 21);
  ASSERT_EQ(mat_b_rank.size(), 15);

  status = svd.CompressWithFNorm(5);
  ASSERT_EQ(status, RET_OK);
  std::vector<float> mat_a_err = svd.GetMatA();
  std::vector<float> mat_b_err = svd.GetMatB();
  std::vector<int> mat_a_err_shape = svd.GetMatShapeA();
  std::vector<int> mat_b_err_shape = svd.GetMatShapeB();
  err = Reconsruct_err(&data, &mat_a_err, &mat_b_err, shape, mat_a_err_shape, mat_b_err_shape);
  ASSERT_LE(err, 5);
  ASSERT_EQ(mat_a_err_shape[1], mat_b_err_shape[0]);
  ASSERT_EQ(mat_a_err.size(), 7);
  ASSERT_EQ(mat_b_err.size(), 5);
}
}  // namespace lite
}  // namespace mindspore
