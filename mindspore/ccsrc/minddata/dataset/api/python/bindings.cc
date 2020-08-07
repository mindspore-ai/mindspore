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
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/api/python/de_pipeline.h"

namespace mindspore {
namespace dataset {

PYBIND_REGISTER(
  DEPipeline, 0, ([](const py::module *m) {
    (void)py::class_<DEPipeline>(*m, "DEPipeline")
      .def(py::init<>())
      .def(
        "AddNodeToTree",
        [](DEPipeline &de, const OpName &op_name, const py::dict &args) {
          py::dict out;
          THROW_IF_ERROR(de.AddNodeToTree(op_name, args, &out));
          return out;
        },
        py::return_value_policy::reference)
      .def_static("AddChildToParentNode",
                  [](const DsOpPtr &child_op, const DsOpPtr &parent_op) {
                    THROW_IF_ERROR(DEPipeline::AddChildToParentNode(child_op, parent_op));
                  })
      .def("AssignRootNode",
           [](DEPipeline &de, const DsOpPtr &dataset_op) { THROW_IF_ERROR(de.AssignRootNode(dataset_op)); })
      .def("SetBatchParameters",
           [](DEPipeline &de, const py::dict &args) { THROW_IF_ERROR(de.SetBatchParameters(args)); })
      .def("LaunchTreeExec", [](DEPipeline &de, int32_t num_epochs) { THROW_IF_ERROR(de.LaunchTreeExec(num_epochs)); })
      .def("GetNextAsMap",
           [](DEPipeline &de) {
             py::dict out;
             THROW_IF_ERROR(de.GetNextAsMap(&out));
             return out;
           })
      .def("GetNextAsList",
           [](DEPipeline &de) {
             py::list out;
             THROW_IF_ERROR(de.GetNextAsList(&out));
             return out;
           })
      .def("GetOutputShapes",
           [](DEPipeline &de) {
             py::list out;
             THROW_IF_ERROR(de.GetOutputShapes(&out));
             return out;
           })
      .def("GetOutputTypes",
           [](DEPipeline &de) {
             py::list out;
             THROW_IF_ERROR(de.GetOutputTypes(&out));
             return out;
           })
      .def("GetDatasetSize", &DEPipeline::GetDatasetSize)
      .def("GetBatchSize", &DEPipeline::GetBatchSize)
      .def("GetNumClasses", &DEPipeline::GetNumClasses)
      .def("GetRepeatCount", &DEPipeline::GetRepeatCount)
      .def("StopSend", [](DEPipeline &de) { THROW_IF_ERROR(de.StopSend()); })
      .def("SaveDataset", [](DEPipeline &de, const std::vector<std::string> &file_names, const std::string &file_type) {
        THROW_IF_ERROR(de.SaveDataset(file_names, file_type));
        return true;
      });
  }));

PYBIND_REGISTER(OpName, 0, ([](const py::module *m) {
                  (void)py::enum_<OpName>(*m, "OpName", py::arithmetic())
                    .value("SHUFFLE", OpName::kShuffle)
                    .value("BATCH", OpName::kBatch)
                    .value("BUCKETBATCH", OpName::kBucketBatch)
                    .value("BARRIER", OpName::kBarrier)
                    .value("MINDRECORD", OpName::kMindrecord)
                    .value("CACHE", OpName::kCache)
                    .value("REPEAT", OpName::kRepeat)
                    .value("SKIP", OpName::kSkip)
                    .value("TAKE", OpName::kTake)
                    .value("ZIP", OpName::kZip)
                    .value("CONCAT", OpName::kConcat)
                    .value("MAP", OpName::kMap)
                    .value("FILTER", OpName::kFilter)
                    .value("DEVICEQUEUE", OpName::kDeviceQueue)
                    .value("GENERATOR", OpName::kGenerator)
                    .export_values()
                    .value("RENAME", OpName::kRename)
                    .value("TFREADER", OpName::kTfReader)
                    .value("PROJECT", OpName::kProject)
                    .value("IMAGEFOLDER", OpName::kImageFolder)
                    .value("MNIST", OpName::kMnist)
                    .value("MANIFEST", OpName::kManifest)
                    .value("VOC", OpName::kVoc)
                    .value("COCO", OpName::kCoco)
                    .value("CIFAR10", OpName::kCifar10)
                    .value("CIFAR100", OpName::kCifar100)
                    .value("RANDOMDATA", OpName::kRandomData)
                    .value("BUILDVOCAB", OpName::kBuildVocab)
                    .value("SENTENCEPIECEVOCAB", OpName::kSentencePieceVocab)
                    .value("CELEBA", OpName::kCelebA)
                    .value("TEXTFILE", OpName::kTextFile)
                    .value("EPOCHCTRL", OpName::kEpochCtrl)
                    .value("CSV", OpName::kCsv)
                    .value("CLUE", OpName::kClue);
                }));

}  // namespace dataset
}  // namespace mindspore
