/**
 * Copyright 2021-2024 Huawei Technologies Co., Ltd
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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pybind11/pybind11.h"

#include "minddata/dataset/api/python/pybind_conversion.h"
#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/include/dataset/transforms.h"
#if defined(ENABLE_D)
#include "minddata/dataset/vision/kernels/dvpp/acl_adapter.h"
#endif
#include "minddata/dataset/vision/kernels/image_utils.h"
#include "minddata/dataset/vision/kernels/pyav/container.h"
#include "minddata/dataset/vision/kernels/pyav/context.h"
#include "minddata/dataset/vision/kernels/pyav/frame.h"
#include "minddata/dataset/vision/kernels/pyav/packet.h"
#include "minddata/dataset/vision/kernels/pyav/stream.h"
#include "minddata/dataset/vision/kernels/video_utils.h"

#include "minddata/dataset/vision/transform/center_crop_ir.h"
#include "minddata/dataset/vision/transform/decode_ir.h"
#include "minddata/dataset/vision/transform/hwc_to_chw_ir.h"
#include "minddata/dataset/vision/transform/normalize_ir.h"
#include "minddata/dataset/vision/transform/random_color_adjust_ir.h"
#include "minddata/dataset/vision/transform/random_crop_decode_resize_ir.h"
#include "minddata/dataset/vision/transform/random_crop_ir.h"
#include "minddata/dataset/vision/transform/random_horizontal_flip_ir.h"
#include "minddata/dataset/vision/transform/random_resized_crop_ir.h"
#include "minddata/dataset/vision/transform/random_vertical_flip_ir.h"
#include "minddata/dataset/vision/transform/rescale_ir.h"
#include "minddata/dataset/vision/transform/resize_ir.h"
#include "minddata/dataset/vision/transform/rgb_to_bgr_ir.h"
#include "minddata/dataset/vision/transform/rotate_ir.h"
#include "minddata/dataset/vision/transform/to_tensor_ir.h"

namespace mindspore {
namespace dataset {

PYBIND_REGISTER(
  CenterCropOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::CenterCropOperation, TensorOperation, std::shared_ptr<vision::CenterCropOperation>>(
      *m, "CenterCropOperation", "Tensor operation to crop and image in the middle. Takes height and width (optional)")
      .def(py::init([](const std::vector<int32_t> &size) {
        auto center_crop = std::make_shared<vision::CenterCropOperation>(size);
        THROW_IF_ERROR(center_crop->ValidateParams());
        return center_crop;
      }));
  }));

PYBIND_REGISTER(DecodeOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::DecodeOperation, TensorOperation, std::shared_ptr<vision::DecodeOperation>>(
                    *m, "DecodeOperation")
                    .def(py::init([](bool rgb, const std::string &device_target) {
                      auto decode = std::make_shared<vision::DecodeOperation>(rgb, device_target);
                      THROW_IF_ERROR(decode->ValidateParams());
                      return decode;
                    }));
                }));

#ifdef ENABLE_FFMPEG

#endif

PYBIND_REGISTER(HwcToChwOperation, 1, ([](const py::module *m) {
                  (void)
                    py::class_<vision::HwcToChwOperation, TensorOperation, std::shared_ptr<vision::HwcToChwOperation>>(
                      *m, "HwcToChwOperation")
                      .def(py::init([]() {
                        auto hwc_to_chw = std::make_shared<vision::HwcToChwOperation>();
                        THROW_IF_ERROR(hwc_to_chw->ValidateParams());
                        return hwc_to_chw;
                      }));
                }));

PYBIND_REGISTER(
  NormalizeOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::NormalizeOperation, TensorOperation, std::shared_ptr<vision::NormalizeOperation>>(
      *m, "NormalizeOperation")
      .def(py::init(
        [](const std::vector<float> &mean, const std::vector<float> &std, bool is_hwc, std::string device_target) {
          auto normalize = std::make_shared<vision::NormalizeOperation>(mean, std, is_hwc, device_target);
          THROW_IF_ERROR(normalize->ValidateParams());
          return normalize;
        }));
  }));

PYBIND_REGISTER(RandomColorAdjustOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomColorAdjustOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomColorAdjustOperation>>(*m,
                                                                                        "RandomColorAdjustOperation")
                    .def(py::init([](const std::vector<float> &brightness, const std::vector<float> &contrast,
                                     const std::vector<float> &saturation, const std::vector<float> &hue) {
                      auto random_color_adjust =
                        std::make_shared<vision::RandomColorAdjustOperation>(brightness, contrast, saturation, hue);
                      THROW_IF_ERROR(random_color_adjust->ValidateParams());
                      return random_color_adjust;
                    }));
                }));

PYBIND_REGISTER(
  RandomCropDecodeResizeOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RandomCropDecodeResizeOperation, TensorOperation,
                     std::shared_ptr<vision::RandomCropDecodeResizeOperation>>(*m, "RandomCropDecodeResizeOperation")
      .def(py::init([](const std::vector<int32_t> &size, const std::vector<float> &scale,
                       const std::vector<float> &ratio, InterpolationMode interpolation, int32_t max_attempts) {
        auto random_crop_decode_resize =
          std::make_shared<vision::RandomCropDecodeResizeOperation>(size, scale, ratio, interpolation, max_attempts);
        THROW_IF_ERROR(random_crop_decode_resize->ValidateParams());
        return random_crop_decode_resize;
      }));
  }));

PYBIND_REGISTER(
  RandomCropOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RandomCropOperation, TensorOperation, std::shared_ptr<vision::RandomCropOperation>>(
      *m, "RandomCropOperation")
      .def(py::init([](const std::vector<int32_t> &size, const std::vector<int32_t> &padding, bool pad_if_needed,
                       const std::vector<uint8_t> &fill_value, BorderType padding_mode) {
        auto random_crop =
          std::make_shared<vision::RandomCropOperation>(size, padding, pad_if_needed, fill_value, padding_mode);
        THROW_IF_ERROR(random_crop->ValidateParams());
        return random_crop;
      }));
  }));

PYBIND_REGISTER(RandomHorizontalFlipOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomHorizontalFlipOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomHorizontalFlipOperation>>(
                    *m, "RandomHorizontalFlipOperation")
                    .def(py::init([](float prob) {
                      auto random_horizontal_flip = std::make_shared<vision::RandomHorizontalFlipOperation>(prob);
                      THROW_IF_ERROR(random_horizontal_flip->ValidateParams());
                      return random_horizontal_flip;
                    }));
                }));

PYBIND_REGISTER(
  RandomResizedCropOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RandomResizedCropOperation, TensorOperation,
                     std::shared_ptr<vision::RandomResizedCropOperation>>(*m, "RandomResizedCropOperation")
      .def(py::init([](const std::vector<int32_t> &size, const std::vector<float> &scale,
                       const std::vector<float> &ratio, InterpolationMode interpolation, int32_t max_attempts) {
        auto random_resized_crop =
          std::make_shared<vision::RandomResizedCropOperation>(size, scale, ratio, interpolation, max_attempts);
        THROW_IF_ERROR(random_resized_crop->ValidateParams());
        return random_resized_crop;
      }));
  }));

PYBIND_REGISTER(RandomVerticalFlipOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomVerticalFlipOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomVerticalFlipOperation>>(*m,
                                                                                         "RandomVerticalFlipOperation")
                    .def(py::init([](float prob) {
                      auto random_vertical_flip = std::make_shared<vision::RandomVerticalFlipOperation>(prob);
                      THROW_IF_ERROR(random_vertical_flip->ValidateParams());
                      return random_vertical_flip;
                    }));
                }));

#ifdef ENABLE_FFMPEG

#endif

PYBIND_REGISTER(RescaleOperation, 1, ([](const py::module *m) {
                  (void)
                    py::class_<vision::RescaleOperation, TensorOperation, std::shared_ptr<vision::RescaleOperation>>(
                      *m, "RescaleOperation")
                      .def(py::init([](float rescale, float shift) {
                        auto rescale_op = std::make_shared<vision::RescaleOperation>(rescale, shift);
                        THROW_IF_ERROR(rescale_op->ValidateParams());
                        return rescale_op;
                      }));
                }));

PYBIND_REGISTER(ResizeOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::ResizeOperation, TensorOperation, std::shared_ptr<vision::ResizeOperation>>(
                    *m, "ResizeOperation")
                    .def(py::init([](const std::vector<int32_t> &size, InterpolationMode interpolation_mode,
                                     std::string device_target) {
                      auto resize = std::make_shared<vision::ResizeOperation>(size, interpolation_mode, device_target);
                      THROW_IF_ERROR(resize->ValidateParams());
                      return resize;
                    }));
                }));

PYBIND_REGISTER(RgbToBgrOperation, 1, ([](const py::module *m) {
                  (void)
                    py::class_<vision::RgbToBgrOperation, TensorOperation, std::shared_ptr<vision::RgbToBgrOperation>>(
                      *m, "RgbToBgrOperation")
                      .def(py::init([]() {
                        auto rgb2bgr = std::make_shared<vision::RgbToBgrOperation>();
                        THROW_IF_ERROR(rgb2bgr->ValidateParams());
                        return rgb2bgr;
                      }));
                }));

PYBIND_REGISTER(ToTensorOperation, 1, ([](const py::module *m) {
                  (void)
                    py::class_<vision::ToTensorOperation, TensorOperation, std::shared_ptr<vision::ToTensorOperation>>(
                      *m, "ToTensorOperation")
                      .def(py::init([](const std::string &output_type) {
                        auto totensor = std::make_shared<vision::ToTensorOperation>(output_type);
                        THROW_IF_ERROR(totensor->ValidateParams());
                        return totensor;
                      }));
                }));

#if defined(ENABLE_D)
PYBIND_REGISTER(DvppCodec, 1, ([](py::module *m) {
                  (void)m->def(
                    "dvpp_sys_init", []() { THROW_IF_ERROR(AclAdapter::GetInstance().DvppSysInit()); },
                    py::call_guard<py::gil_scoped_release>());

                  (void)m->def(
                    "dvpp_sys_exit", []() { THROW_IF_ERROR(AclAdapter::GetInstance().DvppSysExit()); },
                    py::call_guard<py::gil_scoped_release>());

                  (void)m->def(
                    "decode_video_create_chn",
                    [](int ptype) {
                      int64_t chnl;
                      THROW_IF_ERROR(AclAdapter::GetInstance().DvppVdecCreateChnl(ptype, &chnl));
                      return chnl;
                    },
                    py::call_guard<py::gil_scoped_release>());

                  (void)m->def(
                    "decode_video_start_get_frame",
                    [](int chn_id, int total_frame) {
                      THROW_IF_ERROR(AclAdapter::GetInstance().DvppVdecStartGetFrame(chn_id, total_frame));
                    },
                    py::call_guard<py::gil_scoped_release>());

                  (void)m->def(
                    "decode_video_send_stream",
                    [](int chn_id, const std::shared_ptr<DeviceBuffer> &input, int64_t out_format, bool display,
                       std::shared_ptr<DeviceBuffer> *out) {
                      Status status =
                        AclAdapter::GetInstance().DvppVdecSendStream(chn_id, input, out_format, display, out);
                      if (status.IsError()) {
                        MS_LOG(WARNING) << status.ToString();
                      }
                    },
                    py::call_guard<py::gil_scoped_release>());

                  (void)m->def(
                    "decode_video_stop_get_frame",
                    [](int chn_id, int total_frame) {
                      std::shared_ptr<DeviceBuffer> output;
                      THROW_IF_ERROR(AclAdapter::GetInstance().DvppVdecStopGetFrame(chn_id, total_frame, &output));
                      return output;
                    },
                    py::call_guard<py::gil_scoped_release>());

                  (void)m->def(
                    "decode_video_destroy_chnl",
                    [](int chn_id) {
                      Status status = AclAdapter::GetInstance().DvppVdecDestroyChnl(chn_id);
                      if (status.IsError()) {
                        MS_LOG(WARNING) << status.ToString();
                      }
                    },
                    py::call_guard<py::gil_scoped_release>());
                }));
#endif

PYBIND_REGISTER(PyAV, 0, ([](py::module *m) {
                  (void)m->def(
                    "pyav_open",
                    [](const std::string &file) {
                      auto container = std::make_shared<Container>(file);
                      THROW_IF_ERROR(container->Init());
                      return container;
                    },
                    py::call_guard<py::gil_scoped_release>());
                }));

PYBIND_REGISTER(CodecContext, 0, ([](const py::module *m) {
                  (void)py::class_<CodecContext, std::shared_ptr<CodecContext>>(*m, "CodecContext")
                    .def_property_readonly("extradata", ([](CodecContext &codec_context) {
                                             auto extradata = codec_context.GetExtradata();
                                             return py::bytes(extradata);
                                           }));
                }));

PYBIND_REGISTER(Stream, 0, ([](const py::module *m) {
                  (void)py::class_<Stream, std::shared_ptr<Stream>>(*m, "Stream")
                    .def_property_readonly("type", &Stream::GetType)
                    .def_property_readonly("time_base", ([](Stream &stream) {
                                             auto time_base = stream.GetTimeBase();
                                             py::object Fraction = py::module::import("fractions").attr("Fraction");
                                             return Fraction(time_base->num, time_base->den);
                                           }))
                    .def_property_readonly("start_time", ([](Stream &stream) -> py::object {
                                             auto start_time = stream.GetStartTime();
                                             if (start_time == -1) {
                                               return py::none();
                                             }
                                             return py::int_(start_time);
                                           }))
                    .def_property_readonly("bit_rate", ([](Stream &stream) -> py::object {
                                             auto bit_rate = stream.GetBitRate();
                                             if (bit_rate == -1) {
                                               return py::none();
                                             }
                                             return py::int_(bit_rate);
                                           }))
                    .def_property_readonly("duration", ([](Stream &stream) -> py::object {
                                             auto duration = stream.GetDuration();
                                             if (duration == -1) {
                                               return py::none();
                                             }
                                             return py::int_(duration);
                                           }))
                    .def_property_readonly("frames", &Stream::GetFrames)
                    .def_property_readonly("flags", &Stream::GetFlags)
                    .def_property_readonly("codec_context", &Stream::GetCodecContext);
                }));

PYBIND_REGISTER(Packet, 0, ([](const py::module *m) {
                  (void)py::class_<Packet, std::shared_ptr<Packet>>(*m, "Packet")
                    .def_property_readonly("is_keyframe", &Packet::IsKeyFrame)
                    .def_property_readonly("pts", ([](Packet &packet) -> py::object {
                                             auto pts = packet.GetPTS();
                                             if (pts == -1) {
                                               return py::none();
                                             }
                                             return py::int_(pts);
                                           }))
                    .def_property_readonly("dts", ([](Packet &packet) -> py::object {
                                             auto dts = packet.GetDTS();
                                             if (dts == -1) {
                                               return py::none();
                                             }
                                             return py::int_(dts);
                                           }));
                }));

PYBIND_REGISTER(Frame, 0, ([](const py::module *m) {
                  (void)py::class_<Frame, std::shared_ptr<Frame>>(*m, "Frame")
                    .def_property_readonly("pts", ([](Frame &frame) -> py::object {
                                             auto pts = frame.GetPTS();
                                             if (pts == -1) {
                                               return py::none();
                                             }
                                             return py::int_(pts);
                                           }));
                }));

PYBIND_REGISTER(AudioFrame, 1, ([](const py::module *m) {
                  (void)py::class_<AudioFrame, Frame, std::shared_ptr<AudioFrame>>(*m, "AudioFrame")
                    .def("to_ndarray", [](AudioFrame &audio_frame) { return audio_frame.ToNumpy(); });
                }));

PYBIND_REGISTER(VideoStream, 1, ([](const py::module *m) {
                  (void)py::class_<VideoStream, Stream, std::shared_ptr<VideoStream>>(*m, "VideoStream")
                    .def_property_readonly("name", &VideoStream::GetName)
                    .def_property_readonly("average_rate", ([](VideoStream &video_stream) {
                                             auto average_rate = video_stream.GetAverageRate();
                                             py::object Fraction = py::module::import("fractions").attr("Fraction");
                                             return Fraction(average_rate->num, average_rate->den);
                                           }))
                    .def_property_readonly("width", &VideoStream::GetWidth)
                    .def_property_readonly("height", &VideoStream::GetHeight);
                }));

PYBIND_REGISTER(AudioStream, 1, ([](const py::module *m) {
                  (void)py::class_<AudioStream, Stream, std::shared_ptr<AudioStream>>(*m, "AudioStream")
                    .def_property_readonly("rate", &AudioStream::GetRate);
                }));

PYBIND_REGISTER(StreamContainer, 2, ([](const py::module *m) {
                  (void)py::class_<StreamContainer, std::shared_ptr<StreamContainer>>(*m, "StreamContainer")
                    .def_property_readonly("video", &StreamContainer::GetVideo)
                    .def_property_readonly("audio", &StreamContainer::GetAudio);
                }));

PYBIND_REGISTER(Container, 3, ([](const py::module *m) {
                  (void)py::class_<Container, std::shared_ptr<Container>>(*m, "AVContainer")
                    .def(
                      "__enter__", [](Container &container) -> Container & { return container; },
                      py::return_value_policy::reference_internal)
                    .def(
                      "__exit__", [](Container &container, py::object, py::object, py::object) { container.Close(); },
                      py::call_guard<py::gil_scoped_release>())
                    .def_property_readonly("streams", &Container::GetStreams)
                    .def(
                      "demux",
                      [](Container &container, const std::shared_ptr<Stream> &stream) {
                        std::vector<std::shared_ptr<Packet>> packets;
                        THROW_IF_ERROR(container.Demux(stream, &packets));
                        return packets;
                      },
                      py::call_guard<py::gil_scoped_release>())
                    .def(
                      "decode",
                      [](Container &container, int streams, int video, int audio) {
                        auto stream = container.GetStreams().Get(streams, video, audio);
                        std::vector<std::shared_ptr<Frame>> frames;
                        THROW_IF_ERROR(container.Decode(stream, &frames));
                        return frames;
                      },
                      py::arg("streams") = -1, py::arg("video") = -1, py::arg("audio") = -1,
                      py::call_guard<py::gil_scoped_release>())
                    .def(
                      "seek",
                      [](Container &container, int64_t offset, bool backward, bool any_frame,
                         const std::shared_ptr<Stream> &stream) {
                        THROW_IF_ERROR(container.Seek(offset, backward, any_frame, stream));
                      },
                      py::arg("offset"), py::kw_only(), py::arg("backward"), py::arg("any_frame"), py::arg("stream"),
                      py::call_guard<py::gil_scoped_release>());
                }));
}  // namespace dataset
}  // namespace mindspore
