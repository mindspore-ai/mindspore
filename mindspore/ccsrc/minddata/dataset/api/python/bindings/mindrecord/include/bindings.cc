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
#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "minddata/dataset/api/python/pybind_register.h"

#include "minddata/dataset/util/random.h"
#include "minddata/mindrecord/include/shard_distributed_sample.h"
#include "minddata/mindrecord/include/shard_operator.h"
#include "minddata/mindrecord/include/shard_pk_sample.h"
#include "minddata/mindrecord/include/shard_sample.h"
#include "minddata/mindrecord/include/shard_sequential_sample.h"
#include "minddata/mindrecord/include/shard_shuffle.h"

namespace mindspore {
namespace dataset {

PYBIND_REGISTER(ShardOperator, 0, ([](const py::module *m) {
                  (void)py::class_<mindrecord::ShardOperator, std::shared_ptr<mindrecord::ShardOperator>>(
                    *m, "ShardOperator")
                    .def("add_child",
                         [](std::shared_ptr<mindrecord::ShardOperator> self,
                            std::shared_ptr<mindrecord::ShardOperator> child) { self->SetChildOp(child); });
                }));

PYBIND_REGISTER(ShardDistributedSample, 1, ([](const py::module *m) {
                  (void)py::class_<mindrecord::ShardDistributedSample, mindrecord::ShardSample,
                                   std::shared_ptr<mindrecord::ShardDistributedSample>>(*m,
                                                                                        "MindrecordDistributedSampler")
                    .def(py::init<int64_t, int64_t, bool, uint32_t, int64_t, int64_t>());
                }));

PYBIND_REGISTER(
  ShardPkSample, 1, ([](const py::module *m) {
    (void)py::class_<mindrecord::ShardPkSample, mindrecord::ShardOperator, std::shared_ptr<mindrecord::ShardPkSample>>(
      *m, "MindrecordPkSampler")
      .def(py::init([](int64_t kVal, std::string kColumn, bool shuffle, int64_t num_samples) {
        if (shuffle == true) {
          return std::make_shared<mindrecord::ShardPkSample>(kColumn, kVal, std::numeric_limits<int64_t>::max(),
                                                             GetSeed(), num_samples);
        } else {
          return std::make_shared<mindrecord::ShardPkSample>(kColumn, kVal, num_samples);
        }
      }));
  }));

PYBIND_REGISTER(
  ShardSample, 0, ([](const py::module *m) {
    (void)py::class_<mindrecord::ShardSample, mindrecord::ShardOperator, std::shared_ptr<mindrecord::ShardSample>>(
      *m, "MindrecordSubsetSampler")
      .def(py::init<std::vector<int64_t>, uint32_t>())
      .def(py::init<std::vector<int64_t>>());
  }));

PYBIND_REGISTER(ShardSequentialSample, 0, ([](const py::module *m) {
                  (void)py::class_<mindrecord::ShardSequentialSample, mindrecord::ShardSample,
                                   std::shared_ptr<mindrecord::ShardSequentialSample>>(*m,
                                                                                       "MindrecordSequentialSampler")
                    .def(py::init([](int64_t num_samples, int64_t start_index) {
                      return std::make_shared<mindrecord::ShardSequentialSample>(num_samples, start_index);
                    }));
                }));

PYBIND_REGISTER(
  ShardShuffle, 1, ([](const py::module *m) {
    (void)py::class_<mindrecord::ShardShuffle, mindrecord::ShardOperator, std::shared_ptr<mindrecord::ShardShuffle>>(
      *m, "MindrecordRandomSampler")
      .def(py::init([](int64_t num_samples, bool replacement, bool reshuffle_each_epoch) {
        return std::make_shared<mindrecord::ShardShuffle>(GetSeed(), num_samples, replacement, reshuffle_each_epoch);
      }));
  }));

}  // namespace dataset
}  // namespace mindspore
