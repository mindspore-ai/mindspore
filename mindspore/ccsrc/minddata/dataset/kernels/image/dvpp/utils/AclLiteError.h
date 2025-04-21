/**
* Copyright 2022-2023 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

* http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_UTILS_ACL_LITE_ERROR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_UTILS_ACL_LITE_ERROR_H_

using AclLiteError = int;

constexpr int ACLLITE_OK = 0;
constexpr int ACLLITE_ERROR = 1;
constexpr int ACLLITE_ERROR_INVALID_ARGS = 2;
constexpr int ACLLITE_ERROR_SET_ACL_CONTEXT = 3;
constexpr int ACLLITE_ERROR_GET_ACL_CONTEXT = 4;
constexpr int ACLLITE_ERROR_CREATE_ACL_CONTEXT = 5;
constexpr int ACLLITE_ERROR_CREATE_THREAD = 6;
constexpr int ACLLITE_ERROR_CREATE_STREAM = 7;
constexpr int ACLLITE_ERROR_GET_RUM_MODE = 8;
constexpr int ACLLITE_ERROR_APP_INIT = 9;
constexpr int ACLLITE_ERROR_DEST_INVALID = 10;
constexpr int ACLLITE_ERROR_INITED_ALREADY = 11;
constexpr int ACLLITE_ERROR_ENQUEUE = 12;
constexpr int ACLLITE_ERROR_WRITE_FILE = 13;
constexpr int ACLLITE_ERROR_THREAD_ABNORMAL = 14;
constexpr int ACLLITE_ERROR_START_THREAD = 15;
constexpr int ACLLITE_ERROR_ADD_THREAD = 16;

// malloc or new memory failed
constexpr int ACLLITE_ERROR_MALLOC = 101;
// aclrtMalloc failed
constexpr int ACLLITE_ERROR_MALLOC_DEVICE = 102;

constexpr int ACLLITE_ERROR_MALLOC_DVPP = 103;
// access file failed
constexpr int ACLLITE_ERROR_ACCESS_FILE = 201;
// the file is invalid
constexpr int ACLLITE_ERROR_INVALID_FILE = 202;
// open file failed
constexpr int ACLLITE_ERROR_OPEN_FILE = 203;

// load model repeated
constexpr int ACLLITE_ERROR_LOAD_MODEL_REPEATED = 301;

constexpr int ACLLITE_ERROR_NO_MODEL_DESC = 302;
// load mode by acl failed
constexpr int ACLLITE_ERROR_LOAD_MODEL = 303;

constexpr int ACLLITE_ERROR_CREATE_MODEL_DESC = 304;

constexpr int ACLLITE_ERROR_GET_MODEL_DESC = 305;

constexpr int ACLLITE_ERROR_CREATE_DATASET = 306;

constexpr int ACLLITE_ERROR_CREATE_DATA_BUFFER = 307;

constexpr int ACLLITE_ERROR_ADD_DATASET_BUFFER = 308;

constexpr int ACLLITE_ERROR_EXECUTE_MODEL = 309;

constexpr int ACLLITE_ERROR_GET_DATASET_BUFFER = 310;

constexpr int ACLLITE_ERROR_GET_DATA_BUFFER_ADDR = 311;

constexpr int ACLLITE_ERROR_GET_DATA_BUFFER_SIZE = 312;

constexpr int ACLLITE_ERROR_COPY_DATA = 313;

constexpr int ACLLITE_ERROR_SET_CAMERA = 400;

constexpr int ACLLITE_ERROR_CAMERA_NO_ACCESSABLE = 401;

constexpr int ACLLITE_ERROR_OPEN_CAMERA = 402;

constexpr int ACLLITE_ERROR_READ_CAMERA_FRAME = 403;

constexpr int ACLLITE_ERROR_UNSURPPORT_PROPERTY = 404;

constexpr int ACLLITE_ERROR_INVALID_PROPERTY_VALUE = 405;

constexpr int ACLLITE_ERROR_UNSURPPORT_VIDEO_CAPTURE = 406;

constexpr int ACLLITE_ERROR_CREATE_DVPP_CHANNEL_DESC = 501;

constexpr int ACLLITE_ERRROR_CREATE_DVPP_CHANNEL = 502;

constexpr int ACLLITE_ERROR_CREATE_PIC_DESC = 503;

constexpr int ACLLITE_ERROR_CREATE_RESIZE_CONFIG = 504;

constexpr int ACLLITE_ERROR_RESIZE_ASYNC = 505;

constexpr int ACLLITE_ERROR_SYNC_STREAM = 506;

constexpr int ACLLITE_ERROR_JPEGE_ASYNC = 507;

constexpr int ACLLITE_ERROR_JPEGD_ASYNC = 508;

constexpr int ACLLITE_ERROR_FFMPEG_DECODER_INIT = 601;

constexpr int ACLLITE_ERROR_OPEN_VIDEO_UNREADY = 602;

constexpr int ACLLITE_ERROR_TOO_MANY_VIDEO_DECODERS = 603;

constexpr int ACLLITE_ERROR_SET_VDEC_CHANNEL_ID = 604;

constexpr int ACLLITE_ERROR_SET_STREAM_DESC_DATA = 605;

constexpr int ACLLITE_ERROR_SET_VDEC_CHANNEL_THREAD_ID = 606;

constexpr int ACLLITE_ERROR_SET_VDEC_CALLBACK = 607;

constexpr int ACLLITE_ERROR_SET_VDEC_ENTYPE = 608;

constexpr int ACLLITE_ERROR_SET_VDEC_PIC_FORMAT = 609;

constexpr int ACLLITE_ERROR_CREATE_VDEC_CHANNEL = 610;

constexpr int ACLLITE_ERROR_CREATE_STREAM_DESC = 611;

constexpr int ACLLITE_ERROR_SET_STREAM_DESC_EOS = 612;

constexpr int ACLLITE_ERROR_SET_STREAM_DESC_SIZE = 613;

constexpr int ACLLITE_ERROR_SET_PIC_DESC_DATA = 614;

constexpr int ACLLITE_ERROR_SET_PIC_DESC_SIZE = 615;

constexpr int ACLLITE_ERROR_SET_PIC_DESC_FORMAT = 616;

constexpr int ACLLITE_ERROR_VDEC_IS_EXITTING = 617;

constexpr int ACLLITE_ERROR_VDEC_SET_WIDTH = 618;

constexpr int ACLLITE_ERROR_VDEC_WIDTH_INVALID = 619;

constexpr int ACLLITE_ERROR_VDEC_HEIGHT_INVALID = 620;

constexpr int ACLLITE_ERROR_VDEC_SET_HEIGHT = 621;

constexpr int ACLLITE_ERROR_VDEC_ENTYPE_INVALID = 622;

constexpr int ACLLITE_ERROR_VDEC_FORMAT_INVALID = 623;

constexpr int ACLLITE_ERROR_VDEC_INVALID_PARAM = 624;

constexpr int ACLLITE_ERROR_VDEC_SEND_FRAME = 625;

constexpr int ACLLITE_ERROR_VDEC_QUEUE_FULL = 626;

constexpr int ACLLITE_ERROR_SET_RTSP_TRANS = 627;

constexpr int ACLLITE_ERROR_READ_EMPTY = 628;

constexpr int ACLLITE_ERROR_VIDEO_DECODER_STATUS = 629;

constexpr int ACLLITE_ERROR_DECODE_FINISH = 630;

constexpr int ACLLITE_ERROR_H26X_FRAME = 631;

constexpr int ACLLITE_ERROR_VENC_STATUS = 701;

constexpr int ACLLITE_ERROR_VENC_QUEUE_FULL = 702;

constexpr int ACLLITE_ERROR_CREATE_VENC_CHAN_DESC = 703;

constexpr int ACLLITE_ERROR_SET_VENC_CHAN_TID = 704;

constexpr int ACLLITE_ERROR_VENC_SET_EOS = 705;

constexpr int ACLLITE_ERROR_VENC_SET_IF_FRAME = 706;

constexpr int ACLLITE_ERROR_CREATE_VENC_CHAN = 707;

constexpr int ACLLITE_ERROR_VENC_CREATE_FRAME_CONFIG = 708;

constexpr int ACLLITE_ERROR_VENC_SEND_FRAME = 709;

constexpr int ACLLITE_ERROR_SUBSCRIBE_REPORT = 710;

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_UTILS_ACL_LITE_ERROR_H_
