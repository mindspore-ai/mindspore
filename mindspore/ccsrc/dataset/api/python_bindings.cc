/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include <exception>

#include "dataset/api/de_pipeline.h"
#include "dataset/kernels/no_op.h"
#include "dataset/kernels/data/one_hot_op.h"
#include "dataset/kernels/image/center_crop_op.h"
#include "dataset/kernels/image/cut_out_op.h"
#include "dataset/kernels/image/decode_op.h"
#include "dataset/kernels/image/hwc_to_chw_op.h"
#include "dataset/kernels/image/image_utils.h"
#include "dataset/kernels/image/normalize_op.h"
#include "dataset/kernels/image/pad_op.h"
#include "dataset/kernels/image/random_color_adjust_op.h"
#include "dataset/kernels/image/random_crop_decode_resize_op.h"
#include "dataset/kernels/image/random_crop_and_resize_op.h"
#include "dataset/kernels/image/random_crop_op.h"
#include "dataset/kernels/image/random_horizontal_flip_op.h"
#include "dataset/kernels/image/random_resize_op.h"
#include "dataset/kernels/image/random_rotation_op.h"
#include "dataset/kernels/image/random_vertical_flip_op.h"
#include "dataset/kernels/image/rescale_op.h"
#include "dataset/kernels/image/resize_bilinear_op.h"
#include "dataset/kernels/image/resize_op.h"
#include "dataset/kernels/image/uniform_aug_op.h"
#include "dataset/kernels/data/type_cast_op.h"
#include "dataset/engine/datasetops/source/cifar_op.h"
#include "dataset/engine/datasetops/source/image_folder_op.h"
#include "dataset/engine/datasetops/source/io_block.h"
#include "dataset/engine/datasetops/source/mnist_op.h"
#include "dataset/engine/datasetops/source/manifest_op.h"
#include "dataset/engine/datasetops/source/mindrecord_op.h"
#include "dataset/engine/datasetops/source/random_data_op.h"
#include "dataset/engine/datasetops/source/sampler/distributed_sampler.h"
#include "dataset/engine/datasetops/source/sampler/pk_sampler.h"
#include "dataset/engine/datasetops/source/sampler/random_sampler.h"
#include "dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "dataset/engine/datasetops/source/sampler/subset_random_sampler.h"
#include "dataset/engine/datasetops/source/sampler/weighted_random_sampler.h"
#include "dataset/engine/datasetops/source/sampler/python_sampler.h"
#include "dataset/engine/datasetops/source/tf_reader_op.h"
#include "dataset/engine/jagged_connector.h"
#include "dataset/engine/datasetops/source/text_file_op.h"
#include "dataset/engine/datasetops/source/voc_op.h"
#include "dataset/kernels/data/to_float16_op.h"
#include "dataset/util/random.h"
#include "mindrecord/include/shard_operator.h"
#include "mindrecord/include/shard_pk_sample.h"
#include "mindrecord/include/shard_sample.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

namespace py = pybind11;

namespace mindspore {
namespace dataset {
#define THROW_IF_ERROR(s)                                      \
  do {                                                         \
    Status rc = std::move(s);                                  \
    if (rc.IsError()) throw std::runtime_error(rc.ToString()); \
  } while (false)

void bindDEPipeline(py::module *m) {
  (void)py::class_<DEPipeline>(*m, "DEPipeline")
    .def(py::init<>())
    .def(
      "AddNodeToTree",
      [](DEPipeline &de, const OpName &op_name, const py::dict &args) {
        DsOpPtr op;
        THROW_IF_ERROR(de.AddNodeToTree(op_name, args, &op));
        return op;
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
    .def("LaunchTreeExec", [](DEPipeline &de) { THROW_IF_ERROR(de.LaunchTreeExec()); })
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
    .def("GetRepeatCount", &DEPipeline::GetRepeatCount);
}
void bindDatasetOps(py::module *m) {
  (void)py::class_<TFReaderOp, DatasetOp, std::shared_ptr<TFReaderOp>>(*m, "TFReaderOp")
    .def_static("get_num_rows", [](const py::list &files, int64_t numParallelWorkers, bool estimate = false) {
      int64_t count = 0;
      std::vector<std::string> filenames;
      for (auto l : files) {
        !l.is_none() ? filenames.push_back(py::str(l)) : (void)filenames.emplace_back("");
      }
      THROW_IF_ERROR(TFReaderOp::CountTotalRows(&count, filenames, numParallelWorkers, estimate));
      return count;
    });

  (void)py::class_<CifarOp, DatasetOp, std::shared_ptr<CifarOp>>(*m, "CifarOp")
    .def_static("get_num_rows", [](const std::string &dir, int64_t numSamples, bool isCifar10) {
      int64_t count = 0;
      THROW_IF_ERROR(CifarOp::CountTotalRows(dir, numSamples, isCifar10, &count));
      return count;
    });

  (void)py::class_<ImageFolderOp, DatasetOp, std::shared_ptr<ImageFolderOp>>(*m, "ImageFolderOp")
    .def_static("get_num_rows_and_classes", [](const std::string &path, int64_t numSamples) {
      int64_t count = 0, num_classes = 0;
      THROW_IF_ERROR(
        ImageFolderOp::CountRowsAndClasses(path, numSamples, std::set<std::string>{}, &count, &num_classes));
      return py::make_tuple(count, num_classes);
    });

  (void)py::class_<MindRecordOp, DatasetOp, std::shared_ptr<MindRecordOp>>(*m, "MindRecordOp")
    .def_static("get_num_rows",
                [](const std::vector<std::string> &paths, bool load_dataset, const py::object &sampler) {
                  int64_t count = 0;
                  std::shared_ptr<mindrecord::ShardOperator> op;
                  if (py::hasattr(sampler, "_create_for_minddataset")) {
                    auto create = sampler.attr("_create_for_minddataset");
                    op = create().cast<std::shared_ptr<mindrecord::ShardOperator>>();
                  }
                  THROW_IF_ERROR(MindRecordOp::CountTotalRows(paths, load_dataset, op, &count));
                  return count;
                });

  (void)py::class_<ManifestOp, DatasetOp, std::shared_ptr<ManifestOp>>(*m, "ManifestOp")
    .def_static("get_num_rows_and_classes",
                [](const std::string &file, int64_t numSamples, const py::dict &dict, const std::string &usage) {
                  int64_t count = 0, num_classes = 0;
                  THROW_IF_ERROR(ManifestOp::CountTotalRows(file, numSamples, dict, usage, &count, &num_classes));
                  return py::make_tuple(count, num_classes);
                })
    .def_static("get_class_indexing",
                [](const std::string &file, int64_t numSamples, const py::dict &dict, const std::string &usage) {
                  std::map<std::string, int32_t> output_class_indexing;
                  THROW_IF_ERROR(ManifestOp::GetClassIndexing(file, numSamples, dict, usage, &output_class_indexing));
                  return output_class_indexing;
                });

  (void)py::class_<MnistOp, DatasetOp, std::shared_ptr<MnistOp>>(*m, "MnistOp")
    .def_static("get_num_rows", [](const std::string &dir, int64_t numSamples) {
      int64_t count = 0;
      THROW_IF_ERROR(MnistOp::CountTotalRows(dir, numSamples, &count));
      return count;
    });

  (void)py::class_<TextFileOp, DatasetOp, std::shared_ptr<TextFileOp>>(*m, "TextFileOp")
    .def_static("get_num_rows", [](const py::list &files) {
      int64_t count = 0;
      std::vector<std::string> filenames;
      for (auto file : files) {
        !file.is_none() ? filenames.push_back(py::str(file)) : (void)filenames.emplace_back("");
      }
      THROW_IF_ERROR(TextFileOp::CountAllFileRows(filenames, &count));
      return count;
    });
  (void)py::class_<VOCOp, DatasetOp, std::shared_ptr<VOCOp>>(*m, "VOCOp")
    .def_static("get_class_indexing", [](const std::string &dir, const std::string &task_type,
                                         const std::string &task_mode, const py::dict &dict, int64_t numSamples) {
      std::map<std::string, int32_t> output_class_indexing;
      THROW_IF_ERROR(VOCOp::GetClassIndexing(dir, task_type, task_mode, dict, numSamples, &output_class_indexing));
      return output_class_indexing;
    });
}
void bindTensor(py::module *m) {
  (void)py::class_<GlobalContext>(*m, "GlobalContext")
    .def_static("config_manager", &GlobalContext::config_manager, py::return_value_policy::reference);

  (void)py::class_<ConfigManager, std::shared_ptr<ConfigManager>>(*m, "ConfigManager")
    .def("__str__", &ConfigManager::ToString)
    .def("set_rows_per_buffer", &ConfigManager::set_rows_per_buffer)
    .def("set_num_parallel_workers", &ConfigManager::set_num_parallel_workers)
    .def("set_worker_connector_size", &ConfigManager::set_worker_connector_size)
    .def("set_op_connector_size", &ConfigManager::set_op_connector_size)
    .def("set_seed", &ConfigManager::set_seed)
    .def("get_rows_per_buffer", &ConfigManager::rows_per_buffer)
    .def("get_num_parallel_workers", &ConfigManager::num_parallel_workers)
    .def("get_worker_connector_size", &ConfigManager::worker_connector_size)
    .def("get_op_connector_size", &ConfigManager::op_connector_size)
    .def("get_seed", &ConfigManager::seed)
    .def("load", [](ConfigManager &c, std::string s) { (void)c.LoadFile(s); });

  (void)py::class_<Tensor, std::shared_ptr<Tensor>>(*m, "Tensor", py::buffer_protocol())
    .def(py::init([](py::array arr) {
      std::shared_ptr<Tensor> out;
      THROW_IF_ERROR(Tensor::CreateTensor(&out, arr));
      return out;
    }))
    .def_buffer([](Tensor &tensor) {
      py::buffer_info info;
      THROW_IF_ERROR(Tensor::GetBufferInfo(tensor, &info));
      return info;
    })
    .def("__str__", &Tensor::ToString)
    .def("shape", &Tensor::shape)
    .def("type", &Tensor::type)
    .def("as_array", [](py::object &t) {
      auto &tensor = py::cast<Tensor &>(t);
      if (tensor.type() == DataType::DE_STRING) {
        py::array res;
        tensor.GetDataAsNumpyStrings(&res);
        return res;
      }
      py::buffer_info info;
      THROW_IF_ERROR(Tensor::GetBufferInfo(tensor, &info));
      return py::array(pybind11::dtype(info), info.shape, info.strides, info.ptr, t);
    });

  (void)py::class_<TensorShape>(*m, "TensorShape")
    .def(py::init<py::list>())
    .def("__str__", &TensorShape::ToString)
    .def("as_list", &TensorShape::AsPyList)
    .def("is_known", &TensorShape::known);

  (void)py::class_<DataType>(*m, "DataType")
    .def(py::init<std::string>())
    .def(py::self == py::self)
    .def("__str__", &DataType::ToString)
    .def("__deepcopy__", [](py::object &t, py::dict memo) { return t; });
}

void bindTensorOps1(py::module *m) {
  (void)py::class_<TensorOp, std::shared_ptr<TensorOp>>(*m, "TensorOp")
    .def("__deepcopy__", [](py::object &t, py::dict memo) { return t; });

  (void)py::class_<NormalizeOp, TensorOp, std::shared_ptr<NormalizeOp>>(
    *m, "NormalizeOp", "Tensor operation to normalize an image. Takes mean and std.")
    .def(py::init<float, float, float, float, float, float>(), py::arg("meanR"), py::arg("meanG"), py::arg("meanB"),
         py::arg("stdR"), py::arg("stdG"), py::arg("stdB"));

  (void)py::class_<RescaleOp, TensorOp, std::shared_ptr<RescaleOp>>(
    *m, "RescaleOp", "Tensor operation to rescale an image. Takes scale and shift.")
    .def(py::init<float, float>(), py::arg("rescale"), py::arg("shift"));

  (void)py::class_<CenterCropOp, TensorOp, std::shared_ptr<CenterCropOp>>(
    *m, "CenterCropOp", "Tensor operation to crop and image in the middle. Takes height and width (optional)")
    .def(py::init<int32_t, int32_t>(), py::arg("height"), py::arg("width") = CenterCropOp::kDefWidth);

  (void)py::class_<ResizeOp, TensorOp, std::shared_ptr<ResizeOp>>(
    *m, "ResizeOp", "Tensor operation to resize an image. Takes height, width and mode")
    .def(py::init<int32_t, int32_t, InterpolationMode>(), py::arg("targetHeight"),
         py::arg("targetWidth") = ResizeOp::kDefWidth, py::arg("interpolation") = ResizeOp::kDefInterpolation);

  (void)py::class_<UniformAugOp, TensorOp, std::shared_ptr<UniformAugOp>>(
    *m, "UniformAugOp", "Tensor operation to apply random augmentation(s).")
    .def(py::init<py::list, int32_t>(), py::arg("operations"), py::arg("NumOps") = UniformAugOp::kDefNumOps);

  (void)py::class_<ResizeBilinearOp, TensorOp, std::shared_ptr<ResizeBilinearOp>>(
    *m, "ResizeBilinearOp",
    "Tensor operation to resize an image using "
    "Bilinear mode. Takes height and width.")
    .def(py::init<int32_t, int32_t>(), py::arg("targetHeight"), py::arg("targetWidth") = ResizeBilinearOp::kDefWidth);

  (void)py::class_<DecodeOp, TensorOp, std::shared_ptr<DecodeOp>>(*m, "DecodeOp",
                                                                  "Tensor operation to decode a jpg image")
    .def(py::init<>())
    .def(py::init<bool>(), py::arg("rgb_format") = DecodeOp::kDefRgbFormat);

  (void)py::class_<RandomHorizontalFlipOp, TensorOp, std::shared_ptr<RandomHorizontalFlipOp>>(
    *m, "RandomHorizontalFlipOp", "Tensor operation to randomly flip an image horizontally.")
    .def(py::init<float>(), py::arg("probability") = RandomHorizontalFlipOp::kDefProbability);
}

void bindTensorOps2(py::module *m) {
  (void)py::class_<RandomVerticalFlipOp, TensorOp, std::shared_ptr<RandomVerticalFlipOp>>(
    *m, "RandomVerticalFlipOp", "Tensor operation to randomly flip an image vertically.")
    .def(py::init<float>(), py::arg("probability") = RandomVerticalFlipOp::kDefProbability);

  (void)py::class_<RandomCropOp, TensorOp, std::shared_ptr<RandomCropOp>>(*m, "RandomCropOp",
                                                                          "Gives random crop of specified size "
                                                                          "Takes crop size")
    .def(py::init<int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, BorderType, bool, uint8_t, uint8_t, uint8_t>(),
         py::arg("cropHeight"), py::arg("cropWidth"), py::arg("padTop") = RandomCropOp::kDefPadTop,
         py::arg("padBottom") = RandomCropOp::kDefPadBottom, py::arg("padLeft") = RandomCropOp::kDefPadLeft,
         py::arg("padRight") = RandomCropOp::kDefPadRight, py::arg("borderType") = RandomCropOp::kDefBorderType,
         py::arg("padIfNeeded") = RandomCropOp::kDefPadIfNeeded, py::arg("fillR") = RandomCropOp::kDefFillR,
         py::arg("fillG") = RandomCropOp::kDefFillG, py::arg("fillB") = RandomCropOp::kDefFillB);
  (void)py::class_<HwcToChwOp, TensorOp, std::shared_ptr<HwcToChwOp>>(*m, "ChannelSwapOp").def(py::init<>());

  (void)py::class_<OneHotOp, TensorOp, std::shared_ptr<OneHotOp>>(
    *m, "OneHotOp", "Tensor operation to apply one hot encoding. Takes number of classes.")
    .def(py::init<int32_t>());

  (void)py::class_<RandomRotationOp, TensorOp, std::shared_ptr<RandomRotationOp>>(
    *m, "RandomRotationOp",
    "Tensor operation to apply RandomRotation."
    "Takes a range for degrees and "
    "optional parameters for rotation center and image expand")
    .def(py::init<float, float, float, float, InterpolationMode, bool, uint8_t, uint8_t, uint8_t>(),
         py::arg("startDegree"), py::arg("endDegree"), py::arg("centerX") = RandomRotationOp::kDefCenterX,
         py::arg("centerY") = RandomRotationOp::kDefCenterY,
         py::arg("interpolation") = RandomRotationOp::kDefInterpolation,
         py::arg("expand") = RandomRotationOp::kDefExpand, py::arg("fillR") = RandomRotationOp::kDefFillR,
         py::arg("fillG") = RandomRotationOp::kDefFillG, py::arg("fillB") = RandomRotationOp::kDefFillB);
}

void bindTensorOps3(py::module *m) {
  (void)py::class_<RandomCropAndResizeOp, TensorOp, std::shared_ptr<RandomCropAndResizeOp>>(
    *m, "RandomCropAndResizeOp",
    "Tensor operation to randomly crop an image and resize to a given size."
    "Takes output height and width and"
    "optional parameters for lower and upper bound for aspect ratio (h/w) and scale,"
    "interpolation mode, and max attempts to crop")
    .def(py::init<int32_t, int32_t, float, float, float, float, InterpolationMode, int32_t>(), py::arg("targetHeight"),
         py::arg("targetWidth"), py::arg("scaleLb") = RandomCropAndResizeOp::kDefScaleLb,
         py::arg("scaleUb") = RandomCropAndResizeOp::kDefScaleUb,
         py::arg("aspectLb") = RandomCropAndResizeOp::kDefAspectLb,
         py::arg("aspectUb") = RandomCropAndResizeOp::kDefAspectUb,
         py::arg("interpolation") = RandomCropAndResizeOp::kDefInterpolation,
         py::arg("maxIter") = RandomCropAndResizeOp::kDefMaxIter);

  (void)py::class_<RandomColorAdjustOp, TensorOp, std::shared_ptr<RandomColorAdjustOp>>(
    *m, "RandomColorAdjustOp",
    "Tensor operation to adjust an image's color randomly."
    "Takes range for brightness, contrast, saturation, hue and")
    .def(py::init<float, float, float, float, float, float, float, float>(), py::arg("bright_factor_start"),
         py::arg("bright_factor_end"), py::arg("contrast_factor_start"), py::arg("contrast_factor_end"),
         py::arg("saturation_factor_start"), py::arg("saturation_factor_end"), py::arg("hue_factor_start"),
         py::arg("hue_factor_end"));

  (void)py::class_<RandomResizeOp, TensorOp, std::shared_ptr<RandomResizeOp>>(
    *m, "RandomResizeOp",
    "Tensor operation to resize an image using a randomly selected interpolation. Takes height and width.")
    .def(py::init<int32_t, int32_t>(), py::arg("targetHeight"),
         py::arg("targetWidth") = RandomResizeOp::kDefTargetWidth);

  (void)py::class_<CutOutOp, TensorOp, std::shared_ptr<CutOutOp>>(
    *m, "CutOutOp", "Tensor operation to randomly erase a portion of the image. Takes height and width.")
    .def(py::init<int32_t, int32_t, int32_t, bool, uint8_t, uint8_t, uint8_t>(), py::arg("boxHeight"),
         py::arg("boxWidth"), py::arg("numPatches"), py::arg("randomColor") = CutOutOp::kDefRandomColor,
         py::arg("fillR") = CutOutOp::kDefFillR, py::arg("fillG") = CutOutOp::kDefFillG,
         py::arg("fillB") = CutOutOp::kDefFillB);
}

void bindTensorOps4(py::module *m) {
  (void)py::class_<TypeCastOp, TensorOp, std::shared_ptr<TypeCastOp>>(
    *m, "TypeCastOp", "Tensor operator to type cast data to a specified type.")
    .def(py::init<DataType>(), py::arg("data_type"))
    .def(py::init<std::string>(), py::arg("data_type"));

  (void)py::class_<NoOp, TensorOp, std::shared_ptr<NoOp>>(*m, "NoOp",
                                                          "TensorOp that does nothing, for testing purposes only.")
    .def(py::init<>());

  (void)py::class_<ToFloat16Op, TensorOp, std::shared_ptr<ToFloat16Op>>(
    *m, "ToFloat16Op", py::dynamic_attr(), "Tensor operator to type cast float32 data to a float16 type.")
    .def(py::init<>());

  (void)py::class_<RandomCropDecodeResizeOp, TensorOp, std::shared_ptr<RandomCropDecodeResizeOp>>(
    *m, "RandomCropDecodeResizeOp", "equivalent to RandomCropAndResize but crops before decoding")
    .def(py::init<int32_t, int32_t, float, float, float, float, InterpolationMode, int32_t>(), py::arg("targetHeight"),
         py::arg("targetWidth"), py::arg("scaleLb") = RandomCropDecodeResizeOp::kDefScaleLb,
         py::arg("scaleUb") = RandomCropDecodeResizeOp::kDefScaleUb,
         py::arg("aspectLb") = RandomCropDecodeResizeOp::kDefAspectLb,
         py::arg("aspectUb") = RandomCropDecodeResizeOp::kDefAspectUb,
         py::arg("interpolation") = RandomCropDecodeResizeOp::kDefInterpolation,
         py::arg("maxIter") = RandomCropDecodeResizeOp::kDefMaxIter);

  (void)py::class_<PadOp, TensorOp, std::shared_ptr<PadOp>>(
    *m, "PadOp",
    "Pads image with specified color, default black, "
    "Takes amount to pad for top, bottom, left, right of image, boarder type and color")
    .def(py::init<int32_t, int32_t, int32_t, int32_t, BorderType, uint8_t, uint8_t, uint8_t>(), py::arg("padTop"),
         py::arg("padBottom"), py::arg("padLeft"), py::arg("padRight"), py::arg("borderTypes") = PadOp::kDefBorderType,
         py::arg("fillR") = PadOp::kDefFillR, py::arg("fillG") = PadOp::kDefFillG, py::arg("fillB") = PadOp::kDefFillB);
}

void bindSamplerOps(py::module *m) {
  (void)py::class_<Sampler, std::shared_ptr<Sampler>>(*m, "Sampler")
    .def("set_num_rows", [](Sampler &self, int64_t rows) { THROW_IF_ERROR(self.SetNumRowsInDataset(rows)); })
    .def("set_num_samples", [](Sampler &self, int64_t samples) { THROW_IF_ERROR(self.SetNumSamples(samples)); })
    .def("initialize", [](Sampler &self) { THROW_IF_ERROR(self.InitSampler()); })
    .def("get_indices", [](Sampler &self) {
      py::array ret;
      THROW_IF_ERROR(self.GetAllIdsThenReset(&ret));
      return ret;
    });

  (void)py::class_<mindrecord::ShardOperator, std::shared_ptr<mindrecord::ShardOperator>>(*m, "ShardOperator");

  (void)py::class_<DistributedSampler, Sampler, std::shared_ptr<DistributedSampler>>(*m, "DistributedSampler")
    .def(py::init<int64_t, int64_t, bool, uint32_t>(), py::arg("numDev"), py::arg("devId"), py::arg("shuffle"),
         py::arg("seed"));

  (void)py::class_<PKSampler, Sampler, std::shared_ptr<PKSampler>>(*m, "PKSampler")
    .def(py::init<int64_t, bool>(), py::arg("kVal"), py::arg("shuffle"));

  (void)py::class_<RandomSampler, Sampler, std::shared_ptr<RandomSampler>>(*m, "RandomSampler")
    .def(py::init<bool, int64_t>(), py::arg("replacement"), py::arg("numSamples"))
    .def(py::init<bool>(), py::arg("replacement"));

  (void)py::class_<SequentialSampler, Sampler, std::shared_ptr<SequentialSampler>>(*m, "SequentialSampler")
    .def(py::init<>());

  (void)py::class_<SubsetRandomSampler, Sampler, std::shared_ptr<SubsetRandomSampler>>(*m, "SubsetRandomSampler")
    .def(py::init<std::vector<int64_t>>(), py::arg("indices"));

  (void)py::class_<mindrecord::ShardSample, mindrecord::ShardOperator, std::shared_ptr<mindrecord::ShardSample>>(
    *m, "MindrecordSubsetRandomSampler")
    .def(py::init<std::vector<int64_t>, uint32_t>(), py::arg("indices"), py::arg("seed") = GetSeed());
  (void)py::class_<mindrecord::ShardPkSample, mindrecord::ShardOperator, std::shared_ptr<mindrecord::ShardPkSample>>(
    *m, "MindrecordPkSampler")
    .def(py::init([](int64_t kVal, std::string kColumn, bool shuffle) {
      if (shuffle == true) {
        return std::make_shared<mindrecord::ShardPkSample>(kColumn, kVal, std::numeric_limits<int64_t>::max(),
                                                           GetSeed());
      } else {
        return std::make_shared<mindrecord::ShardPkSample>(kColumn, kVal);
      }
    }));

  (void)py::class_<WeightedRandomSampler, Sampler, std::shared_ptr<WeightedRandomSampler>>(*m, "WeightedRandomSampler")
    .def(py::init<std::vector<double>, int64_t, bool>(), py::arg("weights"), py::arg("numSamples"),
         py::arg("replacement"));

  (void)py::class_<PythonSampler, Sampler, std::shared_ptr<PythonSampler>>(*m, "PythonSampler")
    .def(py::init<py::object>(), py::arg("pySampler"));
}

void bindInfoObjects(py::module *m) {
  (void)py::class_<BatchOp::CBatchInfo>(*m, "CBatchInfo")
    .def(py::init<int64_t, int64_t, int64_t>())
    .def("get_epoch_num", &BatchOp::CBatchInfo::get_epoch_num)
    .def("get_batch_num", &BatchOp::CBatchInfo::get_batch_num);
}

// This is where we externalize the C logic as python modules
PYBIND11_MODULE(_c_dataengine, m) {
  m.doc() = "pybind11 for _c_dataengine";
  (void)py::class_<DatasetOp, std::shared_ptr<DatasetOp>>(m, "DatasetOp");

  (void)py::enum_<OpName>(m, "OpName", py::arithmetic())
    .value("STORAGE", OpName::kStorage)
    .value("SHUFFLE", OpName::kShuffle)
    .value("BATCH", OpName::kBatch)
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
    .value("CIFAR10", OpName::kCifar10)
    .value("CIFAR100", OpName::kCifar100)
    .value("RANDOMDATA", OpName::kRandomData)
    .value("CELEBA", OpName::kCelebA)
    .value("TEXTFILE", OpName::kTextFile);

  (void)py::enum_<InterpolationMode>(m, "InterpolationMode", py::arithmetic())
    .value("DE_INTER_LINEAR", InterpolationMode::kLinear)
    .value("DE_INTER_CUBIC", InterpolationMode::kCubic)
    .value("DE_INTER_AREA", InterpolationMode::kArea)
    .value("DE_INTER_NEAREST_NEIGHBOUR", InterpolationMode::kNearestNeighbour)
    .export_values();

  (void)py::enum_<BorderType>(m, "BorderType", py::arithmetic())
    .value("DE_BORDER_CONSTANT", BorderType::kConstant)
    .value("DE_BORDER_EDGE", BorderType::kEdge)
    .value("DE_BORDER_REFLECT", BorderType::kReflect)
    .value("DE_BORDER_SYMMETRIC", BorderType::kSymmetric)
    .export_values();
  bindDEPipeline(&m);
  bindTensor(&m);
  bindTensorOps1(&m);
  bindTensorOps2(&m);
  bindTensorOps3(&m);
  bindTensorOps4(&m);
  bindSamplerOps(&m);
  bindDatasetOps(&m);
  bindInfoObjects(&m);
}
}  // namespace dataset
}  // namespace mindspore
