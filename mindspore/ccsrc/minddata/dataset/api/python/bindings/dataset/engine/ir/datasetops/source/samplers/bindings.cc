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
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

#include "minddata/dataset/engine/datasetops/source/sampler/python_sampler.h"
#include "minddata/dataset/api/python/pybind_conversion.h"
#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/callback/py_ds_callback.h"
#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/samplers_ir.h"

namespace mindspore {
namespace dataset {

PYBIND_REGISTER(SamplerObj, 1, ([](const py::module *m) {
                  (void)py::class_<SamplerObj, std::shared_ptr<SamplerObj>>(*m, "SamplerObj", "to create a SamplerObj")
                    .def("add_child", [](std::shared_ptr<SamplerObj> self, std::shared_ptr<SamplerObj> child) {
                      THROW_IF_ERROR(self->AddChildSampler(child));
                    });
                }));

PYBIND_REGISTER(DistributedSamplerObj, 2, ([](const py::module *m) {
                  (void)py::class_<DistributedSamplerObj, SamplerObj, std::shared_ptr<DistributedSamplerObj>>(
                    *m, "DistributedSamplerObj", "to create a DistributedSamplerObj")
                    .def(py::init([](int64_t num_shards, int64_t shard_id, bool shuffle, int64_t num_samples,
                                     uint32_t seed, int64_t offset, bool even_dist) {
                      std::shared_ptr<DistributedSamplerObj> sampler = std::make_shared<DistributedSamplerObj>(
                        num_shards, shard_id, shuffle, num_samples, seed, offset, even_dist);
                      THROW_IF_ERROR(sampler->ValidateParams());
                      return sampler;
                    }));
                }));

PYBIND_REGISTER(PreBuiltSamplerObj, 2, ([](const py::module *m) {
                  (void)py::class_<PreBuiltSamplerObj, SamplerObj, std::shared_ptr<PreBuiltSamplerObj>>(
                    *m, "PreBuiltSamplerObj", "to create a PreBuiltSamplerObj")
                    .def(py::init([](int64_t num_samples, py::object sampler) {
                      auto sampler_rt = std::make_shared<PythonSamplerRT>(num_samples, sampler);
                      auto sampler_obj = std::make_shared<PreBuiltSamplerObj>(std::move(sampler_rt));
                      THROW_IF_ERROR(sampler_obj->ValidateParams());
                      return sampler_obj;
                    }));
                }));

PYBIND_REGISTER(PKSamplerObj, 2, ([](const py::module *m) {
                  (void)py::class_<PKSamplerObj, SamplerObj, std::shared_ptr<PKSamplerObj>>(*m, "PKSamplerObj",
                                                                                            "to create a PKSamplerObj")
                    .def(py::init([](int64_t num_val, bool shuffle, int64_t num_samples) {
                      std::shared_ptr<PKSamplerObj> sampler =
                        std::make_shared<PKSamplerObj>(num_val, shuffle, num_samples);
                      THROW_IF_ERROR(sampler->ValidateParams());
                      return sampler;
                    }));
                }));

PYBIND_REGISTER(RandomSamplerObj, 2, ([](const py::module *m) {
                  (void)py::class_<RandomSamplerObj, SamplerObj, std::shared_ptr<RandomSamplerObj>>(
                    *m, "RandomSamplerObj", "to create a RandomSamplerObj")
                    .def(py::init([](bool replacement, int64_t num_samples, bool reshuffle_each_epoch) {
                      std::shared_ptr<RandomSamplerObj> sampler =
                        std::make_shared<RandomSamplerObj>(replacement, num_samples, reshuffle_each_epoch);
                      THROW_IF_ERROR(sampler->ValidateParams());
                      return sampler;
                    }));
                }));

PYBIND_REGISTER(SequentialSamplerObj, 2, ([](const py::module *m) {
                  (void)py::class_<SequentialSamplerObj, SamplerObj, std::shared_ptr<SequentialSamplerObj>>(
                    *m, "SequentialSamplerObj", "to create a SequentialSamplerObj")
                    .def(py::init([](int64_t start_index, int64_t num_samples) {
                      std::shared_ptr<SequentialSamplerObj> sampler =
                        std::make_shared<SequentialSamplerObj>(start_index, num_samples);
                      THROW_IF_ERROR(sampler->ValidateParams());
                      return sampler;
                    }));
                }));

PYBIND_REGISTER(SubsetSamplerObj, 2, ([](const py::module *m) {
                  (void)py::class_<SubsetSamplerObj, SamplerObj, std::shared_ptr<SubsetSamplerObj>>(
                    *m, "SubsetSamplerObj", "to create a SubsetSamplerObj")
                    .def(py::init([](std::vector<int64_t> indices, int64_t num_samples) {
                      std::shared_ptr<SubsetSamplerObj> sampler =
                        std::make_shared<SubsetSamplerObj>(indices, num_samples);
                      THROW_IF_ERROR(sampler->ValidateParams());
                      return sampler;
                    }));
                }));

PYBIND_REGISTER(SubsetRandomSamplerObj, 3, ([](const py::module *m) {
                  (void)py::class_<SubsetRandomSamplerObj, SubsetSamplerObj, std::shared_ptr<SubsetRandomSamplerObj>>(
                    *m, "SubsetRandomSamplerObj", "to create a SubsetRandomSamplerObj")
                    .def(py::init([](std::vector<int64_t> indices, int64_t num_samples) {
                      std::shared_ptr<SubsetRandomSamplerObj> sampler =
                        std::make_shared<SubsetRandomSamplerObj>(indices, num_samples);
                      THROW_IF_ERROR(sampler->ValidateParams());
                      return sampler;
                    }));
                }));

PYBIND_REGISTER(WeightedRandomSamplerObj, 2, ([](const py::module *m) {
                  (void)py::class_<WeightedRandomSamplerObj, SamplerObj, std::shared_ptr<WeightedRandomSamplerObj>>(
                    *m, "WeightedRandomSamplerObj", "to create a WeightedRandomSamplerObj")
                    .def(py::init([](std::vector<double> weights, int64_t num_samples, bool replacement) {
                      std::shared_ptr<WeightedRandomSamplerObj> sampler =
                        std::make_shared<WeightedRandomSamplerObj>(weights, num_samples, replacement);
                      THROW_IF_ERROR(sampler->ValidateParams());
                      return sampler;
                    }));
                }));
}  // namespace dataset
}  // namespace mindspore
