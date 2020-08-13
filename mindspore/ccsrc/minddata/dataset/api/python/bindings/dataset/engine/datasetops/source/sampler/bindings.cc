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

#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/distributed_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/pk_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/python_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/random_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/subset_random_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/weighted_random_sampler.h"

namespace mindspore {
namespace dataset {

PYBIND_REGISTER(Sampler, 0, ([](const py::module *m) {
                  (void)py::class_<Sampler, std::shared_ptr<Sampler>>(*m, "Sampler")
                    .def("set_num_rows",
                         [](Sampler &self, int64_t rows) { THROW_IF_ERROR(self.SetNumRowsInDataset(rows)); })
                    .def("set_num_samples",
                         [](Sampler &self, int64_t samples) { THROW_IF_ERROR(self.SetNumSamples(samples)); })
                    .def("initialize", [](Sampler &self) { THROW_IF_ERROR(self.InitSampler()); })
                    .def("get_indices",
                         [](Sampler &self) {
                           py::array ret;
                           THROW_IF_ERROR(self.GetAllIdsThenReset(&ret));
                           return ret;
                         })
                    .def("add_child", [](std::shared_ptr<Sampler> self, std::shared_ptr<Sampler> child) {
                      THROW_IF_ERROR(self->AddChild(child));
                    });
                }));

PYBIND_REGISTER(DistributedSampler, 1, ([](const py::module *m) {
                  (void)py::class_<DistributedSampler, Sampler, std::shared_ptr<DistributedSampler>>(
                    *m, "DistributedSampler")
                    .def(py::init<int64_t, int64_t, int64_t, bool, uint32_t, int64_t>());
                }));

PYBIND_REGISTER(PKSampler, 1, ([](const py::module *m) {
                  (void)py::class_<PKSampler, Sampler, std::shared_ptr<PKSampler>>(*m, "PKSampler")
                    .def(py::init<int64_t, int64_t, bool>());
                }));

PYBIND_REGISTER(PythonSampler, 1, ([](const py::module *m) {
                  (void)py::class_<PythonSampler, Sampler, std::shared_ptr<PythonSampler>>(*m, "PythonSampler")
                    .def(py::init<int64_t, py::object>());
                }));

PYBIND_REGISTER(RandomSampler, 1, ([](const py::module *m) {
                  (void)py::class_<RandomSampler, Sampler, std::shared_ptr<RandomSampler>>(*m, "RandomSampler")
                    .def(py::init<int64_t, bool, bool>());
                }));

PYBIND_REGISTER(SequentialSampler, 1, ([](const py::module *m) {
                  (void)py::class_<SequentialSampler, Sampler, std::shared_ptr<SequentialSampler>>(*m,
                                                                                                   "SequentialSampler")
                    .def(py::init<int64_t, int64_t>());
                }));

PYBIND_REGISTER(SubsetRandomSampler, 1, ([](const py::module *m) {
                  (void)py::class_<SubsetRandomSampler, Sampler, std::shared_ptr<SubsetRandomSampler>>(
                    *m, "SubsetRandomSampler")
                    .def(py::init<int64_t, std::vector<int64_t>>());
                }));

PYBIND_REGISTER(WeightedRandomSampler, 1, ([](const py::module *m) {
                  (void)py::class_<WeightedRandomSampler, Sampler, std::shared_ptr<WeightedRandomSampler>>(
                    *m, "WeightedRandomSampler")
                    .def(py::init<int64_t, std::vector<double>, bool>());
                }));

}  // namespace dataset
}  // namespace mindspore
