/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/api/python/pybind_conversion.h"

#include "minddata/dataset/engine/python_runtime_context.h"
#include "minddata/dataset/engine/consumers/python_tree_consumer.h"

namespace mindspore {
namespace dataset {
PYBIND_REGISTER(TreeConsumer, 0, ([](const py::module *m) {
                  (void)py::class_<TreeConsumer, std::shared_ptr<TreeConsumer>>(*m, "TreeConsumer")
                    .def("Reset", [](TreeConsumer &self, int64_t step, int64_t epoch) {
                      THROW_IF_ERROR(self.Reset(step, epoch));
                    });
                }));
PYBIND_REGISTER(PythonIteratorConsumer, 1, ([](const py::module *m) {
                  (void)py::class_<PythonIteratorConsumer, TreeConsumer, std::shared_ptr<PythonIteratorConsumer>>(
                    *m, "PythonIteratorConsumer")
                    .def(py::init<int32_t>())
                    .def("Init", [](PythonIteratorConsumer &self,
                                    std::shared_ptr<DatasetNode> d) { THROW_IF_ERROR(self.Init(d)); })
                    .def("GetNextAsMap",
                         [](PythonIteratorConsumer &self) {
                           py::dict output;
                           THROW_IF_ERROR(self.GetNextAsDict(&output));
                           return output;
                         })
                    .def("GetOffload", [](PythonIteratorConsumer &self) { return self.GetOffload(); })
                    .def("GetNextAsList", [](PythonIteratorConsumer &self) {
                      py::list output;
                      THROW_IF_ERROR(self.GetNextAsList(&output));
                      return output;
                    });
                }));

PYBIND_REGISTER(
  PythonPullBasedIteratorConsumer, 1, ([](const py::module *m) {
    (void)py::class_<PythonPullBasedIteratorConsumer, TreeConsumer, std::shared_ptr<PythonPullBasedIteratorConsumer>>(
      *m, "PythonPullBasedIteratorConsumer")
      .def(py::init<int32_t>())
      .def("Init",
           [](PythonPullBasedIteratorConsumer &self, std::shared_ptr<DatasetNode> d) { THROW_IF_ERROR(self.Init(d)); })
      .def("GetNextAsMap",
           [](PythonPullBasedIteratorConsumer &self) {
             py::dict output;
             THROW_IF_ERROR(self.GetNextAsDict(&output));
             return output;
           })
      .def("GetOffload", [](PythonPullBasedIteratorConsumer &self) { return self.GetOffload(); })
      .def("GetNextAsList", [](PythonPullBasedIteratorConsumer &self) {
        py::list output;
        THROW_IF_ERROR(self.GetNextAsList(&output));
        return output;
      });
  }));

PYBIND_REGISTER(TreeGetters, 1, ([](const py::module *m) {
                  (void)py::class_<PythonTreeGetters, TreeConsumer, std::shared_ptr<PythonTreeGetters>>(*m,
                                                                                                        "TreeGetters")
                    .def(py::init<>())
                    .def("Init",
                         [](PythonTreeGetters &self, std::shared_ptr<DatasetNode> d) { THROW_IF_ERROR(self.Init(d)); })
                    .def("GetOutputShapes",
                         [](PythonTreeGetters &self, bool estimate) {
                           std::vector<TensorShape> shapes = {};
                           THROW_IF_ERROR(self.GetOutputShapes(&shapes, estimate));
                           return shapesToListOfShape(shapes);
                         })
                    .def("GetOutputTypes",
                         [](PythonTreeGetters &self) {
                           std::vector<DataType> types = {};
                           THROW_IF_ERROR(self.GetOutputTypes(&types));
                           return typesToListOfType(types);
                         })
                    .def("GetNumClasses",
                         [](PythonTreeGetters &self) {
                           int64_t num_classes = -1;
                           THROW_IF_ERROR(self.GetNumClasses(&num_classes));
                           return num_classes;
                         })
                    .def("GetRepeatCount",
                         [](PythonTreeGetters &self) {
                           int64_t repeat_count = -1;
                           THROW_IF_ERROR(self.GetRepeatCount(&repeat_count));
                           return repeat_count;
                         })
                    .def("GetBatchSize",
                         [](PythonTreeGetters &self) {
                           int64_t batch_size = -1;
                           THROW_IF_ERROR(self.GetBatchSize(&batch_size));
                           return batch_size;
                         })
                    .def("GetColumnNames",
                         [](PythonTreeGetters &self) {
                           std::vector<std::string> col_names = {};
                           THROW_IF_ERROR(self.GetColumnNames(&col_names));
                           return col_names;
                         })
                    .def("GetClassIndexing",
                         [](PythonTreeGetters &self) {
                           std::vector<std::pair<std::string, std::vector<int32_t>>> output_class_indexing;
                           THROW_IF_ERROR(self.GetClassIndexing(&output_class_indexing));
                           return output_class_indexing;
                         })
                    .def("__deepcopy__", [](py::object &tree_getter, py::dict memo) { return tree_getter; });
                }));

PYBIND_REGISTER(PythonRuntimeContext, 2, ([](const py::module *m) {
                  (void)py::class_<PythonRuntimeContext, std::shared_ptr<PythonRuntimeContext>>(*m,
                                                                                                "PythonRuntimeContext")
                    .def(py::init<>())
                    .def("Init", [](PythonRuntimeContext &self) { THROW_IF_ERROR(self.Init()); })
                    .def("AssignConsumer", &PythonRuntimeContext::AssignConsumer)
                    .def("Terminate", [](PythonRuntimeContext &self) { THROW_IF_ERROR(self.Terminate()); })
                    .def("GetConsumer", &PythonRuntimeContext::GetPythonConsumer, py::return_value_policy::reference)
                    .def("__deepcopy__", [](py::object &runtime_context, py::dict memo) { return runtime_context; });
                }));

PYBIND_REGISTER(PythonBuildVocabConsumer, 1, ([](const py::module *m) {
                  (void)py::class_<PythonBuildVocabConsumer, TreeConsumer, std::shared_ptr<PythonBuildVocabConsumer>>(
                    *m, "PythonBuildVocabConsumer")
                    .def(py::init<>())
                    .def("Init", [](PythonBuildVocabConsumer &self,
                                    std::shared_ptr<DatasetNode> d) { THROW_IF_ERROR(self.Init(d)); })
                    .def("Start", [](PythonBuildVocabConsumer &self) { THROW_IF_ERROR(self.Start()); });
                }));

PYBIND_REGISTER(ToDevice, 1, ([](const py::module *m) {
                  (void)py::class_<ToDevice, TreeConsumer, std::shared_ptr<ToDevice>>(*m, "ToDevice")
                    .def(py::init<int32_t>())
                    .def("Init", [](ToDevice &self, std::shared_ptr<DatasetNode> d) { THROW_IF_ERROR(self.Init(d)); })
                    .def("Send", [](ToDevice &self) { THROW_IF_ERROR(self.Send()); })
                    .def("ContinueSend", [](ToDevice &self) { THROW_IF_ERROR(self.Continue()); })
                    .def("StopSend", [](ToDevice &self) { THROW_IF_ERROR(self.Stop()); })
                    .def("GetOffload", [](ToDevice &self) { return self.GetOffload(); })
                    .def("GetDataInfo",
                         [](ToDevice &self) {
                           std::vector<DataType> types_c;
                           std::vector<TensorShape> shapes_c;
                           {
                             py::gil_scoped_release rel;
                             THROW_IF_ERROR(self.GetDataInfo(&types_c, &shapes_c));
                           }
                           py::list types, shapes;
                           for (auto el : types_c) {
                             types.append(el.AsNumpyType());
                             py::list shape;
                           }
                           for (auto el : shapes_c) {
                             py::list shape = el.AsPyList();
                             shapes.append(shape);
                           }
                           return py::make_tuple(types, shapes);
                         })
                    .def("__deepcopy__", [](py::object &to_device, py::dict memo) { return to_device; });
                }));

PYBIND_REGISTER(PythonSaveToDisk, 1, ([](const py::module *m) {
                  (void)py::class_<PythonSaveToDisk, TreeConsumer, std::shared_ptr<PythonSaveToDisk>>(
                    *m, "PythonSaveToDisk")
                    .def(py::init([](std::string &dataset_path, int32_t numFiles, std::string &datasetType) {
                      auto save = std::make_shared<PythonSaveToDisk>(dataset_path, numFiles, datasetType);
                      THROW_IF_ERROR(save->ValidateParams());
                      return save;
                    }))
                    .def("Init",
                         [](PythonSaveToDisk &self, std::shared_ptr<DatasetNode> d) { THROW_IF_ERROR(self.Init(d)); })
                    .def("Save", [](PythonSaveToDisk &self) { THROW_IF_ERROR(self.Save()); });
                }));

PYBIND_REGISTER(PythonDatasetSizeGetter, 1, ([](const py::module *m) {
                  (void)py::class_<PythonDatasetSizeGetter, TreeConsumer, std::shared_ptr<PythonDatasetSizeGetter>>(
                    *m, "DatasetSizeGetters")
                    .def(py::init<>())
                    .def("Init", [](PythonDatasetSizeGetter &self,
                                    std::shared_ptr<DatasetNode> d) { THROW_IF_ERROR(self.Init(d)); })
                    .def("GetDatasetSize", [](PythonDatasetSizeGetter &self, bool estimate) {
                      int64_t size;
                      THROW_IF_ERROR(self.GetDatasetSize(&size, estimate));
                      return size;
                    });
                }));
}  // namespace dataset
}  // namespace mindspore
