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

PYBIND_REGISTER(SamplerRT, 0, ([](const py::module *m) {
                  (void)py::class_<SamplerRT, std::shared_ptr<SamplerRT>>(*m, "Sampler")
                    .def("set_num_rows",
                         [](SamplerRT &self, int64_t rows) { THROW_IF_ERROR(self.SetNumRowsInDataset(rows)); })
                    .def("set_num_samples",
                         [](SamplerRT &self, int64_t samples) { THROW_IF_ERROR(self.SetNumSamples(samples)); })
                    .def("initialize", [](SamplerRT &self) { THROW_IF_ERROR(self.InitSampler()); })
                    .def("get_indices",
                         [](SamplerRT &self) {
                           py::array ret;
                           THROW_IF_ERROR(self.GetAllIdsThenReset(&ret));
                           return ret;
                         })
                    .def("add_child", [](std::shared_ptr<SamplerRT> self, std::shared_ptr<SamplerRT> child) {
                      THROW_IF_ERROR(self->AddChild(child));
                    });
                }));

PYBIND_REGISTER(DistributedSamplerRT, 1, ([](const py::module *m) {
                  (void)py::class_<DistributedSamplerRT, SamplerRT, std::shared_ptr<DistributedSamplerRT>>(
                    *m, "DistributedSampler")
                    .def(py::init<int64_t, int64_t, int64_t, bool, uint32_t, int64_t>());
                }));

PYBIND_REGISTER(PKSamplerRT, 1, ([](const py::module *m) {
                  (void)py::class_<PKSamplerRT, SamplerRT, std::shared_ptr<PKSamplerRT>>(*m, "PKSampler")
                    .def(py::init<int64_t, int64_t, bool>());
                }));

PYBIND_REGISTER(PythonSamplerRT, 1, ([](const py::module *m) {
                  (void)py::class_<PythonSamplerRT, SamplerRT, std::shared_ptr<PythonSamplerRT>>(*m, "PythonSampler")
                    .def(py::init<int64_t, py::object>());
                }));

PYBIND_REGISTER(RandomSamplerRT, 1, ([](const py::module *m) {
                  (void)py::class_<RandomSamplerRT, SamplerRT, std::shared_ptr<RandomSamplerRT>>(*m, "RandomSampler")
                    .def(py::init<int64_t, bool, bool>());
                }));

PYBIND_REGISTER(SequentialSamplerRT, 1, ([](const py::module *m) {
                  (void)py::class_<SequentialSamplerRT, SamplerRT, std::shared_ptr<SequentialSamplerRT>>(
                    *m, "SequentialSampler")
                    .def(py::init<int64_t, int64_t>());
                }));

PYBIND_REGISTER(SubsetRandomSamplerRT, 2, ([](const py::module *m) {
                  (void)py::class_<SubsetRandomSamplerRT, SubsetSamplerRT, std::shared_ptr<SubsetRandomSamplerRT>>(
                    *m, "SubsetRandomSampler")
                    .def(py::init<int64_t, std::vector<int64_t>>());
                }));

PYBIND_REGISTER(SubsetSamplerRT, 1, ([](const py::module *m) {
                  (void)py::class_<SubsetSamplerRT, SamplerRT, std::shared_ptr<SubsetSamplerRT>>(*m, "SubsetSampler")
                    .def(py::init<int64_t, std::vector<int64_t>>());
                }));

PYBIND_REGISTER(WeightedRandomSamplerRT, 1, ([](const py::module *m) {
                  (void)py::class_<WeightedRandomSamplerRT, SamplerRT, std::shared_ptr<WeightedRandomSamplerRT>>(
                    *m, "WeightedRandomSampler")
                    .def(py::init<int64_t, std::vector<double>, bool>());
                }));

}  // namespace dataset
}  // namespace mindspore
