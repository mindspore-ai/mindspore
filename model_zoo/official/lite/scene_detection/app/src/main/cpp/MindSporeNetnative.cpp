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
#include <jni.h>
#include <android/bitmap.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <utility>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <set>
#include "include/errorcode.h"
#include "include/ms_tensor.h"
#include "MindSporeNetnative.h"
#include "MSNetWork.h"
#include "lite_cv/lite_mat.h"
#include "lite_cv/image_process.h"

using mindspore::dataset::LiteMat;
using mindspore::dataset::LPixelType;
using mindspore::dataset::LDataType;
#define MS_PRINT(format, ...) __android_log_print(ANDROID_LOG_INFO, "MSJNI", format, ##__VA_ARGS__)


static const int RET_CATEGORY_SUM = 365;
static const char *labels_name_map[RET_CATEGORY_SUM] = {"airfield", "airplane_cabin",
                                                        "airport_terminal", "alcove", "alley",
                                                        "amphitheater", "amusement_arcade",
                                                        "amusement_park",
                                                        "apartment_building/outdoor", "aquarium",
                                                        "aqueduct", "arcade", "arch",
                                                        "archaelogical_excavation", "archive",
                                                        "arena/hockey", "arena/performance",
                                                        "arena/rodeo", "army_base", "art_gallery",
                                                        "art_school", "art_studio", "artists_loft",
                                                        "assembly_line", "athletic_field/outdoor",
                                                        "atrium/public", "attic", "auditorium",
                                                        "auto_factory", "auto_showroom",
                                                        "badlands", "bakery/shop",
                                                        "balcony/exterior", "balcony/interior",
                                                        "ball_pit",
                                                        "ballroom", "bamboo_forest", "bank_vault",
                                                        "banquet_hall", "bar",
                                                        "barn", "barndoor", "baseball_field",
                                                        "basement", "basketball_court/indoor",
                                                        "bathroom", "bazaar/indoor",
                                                        "bazaar/outdoor", "beach", "beach_house",
                                                        "beauty_salon", "bedchamber", "bedroom",
                                                        "beer_garden", "beer_hall",
                                                        "berth", "biology_laboratory", "boardwalk",
                                                        "boat_deck", "boathouse",
                                                        "bookstore", "booth/indoor",
                                                        "botanical_garden", "bow_window/indoor",
                                                        "bowling_alley",
                                                        "boxing_ring", "bridge", "building_facade",
                                                        "bullring", "burial_chamber",
                                                        "bus_interior", "bus_station/indoor",
                                                        "butchers_shop", "butte", "cabin/outdoor",
                                                        "cafeteria", "campsite", "campus",
                                                        "canal/natural", "canal/urban",
                                                        "candy_store", "canyon", "car_interior",
                                                        "carrousel", "castle",
                                                        "catacomb", "cemetery", "chalet",
                                                        "chemistry_lab", "childs_room",
                                                        "church/indoor", "church/outdoor",
                                                        "classroom", "clean_room", "cliff",
                                                        "closet", "clothing_store", "coast",
                                                        "cockpit", "coffee_shop",
                                                        "computer_room", "conference_center",
                                                        "conference_room", "construction_site",
                                                        "corn_field",
                                                        "corral", "corridor", "cottage",
                                                        "courthouse", "courtyard",
                                                        "creek", "crevasse", "crosswalk", "dam",
                                                        "delicatessen",
                                                        "department_store", "desert/sand",
                                                        "desert/vegetation", "desert_road",
                                                        "diner/outdoor",
                                                        "dining_hall", "dining_room", "discotheque",
                                                        "doorway/outdoor", "dorm_room",
                                                        "downtown", "dressing_room", "driveway",
                                                        "drugstore", "elevator/door",
                                                        "elevator_lobby", "elevator_shaft",
                                                        "embassy", "engine_room", "entrance_hall",
                                                        "escalator/indoor", "excavation",
                                                        "fabric_store", "farm",
                                                        "fastfood_restaurant",
                                                        "field/cultivated", "field/wild",
                                                        "field_road", "fire_escape", "fire_station",
                                                        "fishpond", "flea_market/indoor",
                                                        "florist_shop/indoor", "food_court",
                                                        "football_field",
                                                        "forest/broadleaf", "forest_path",
                                                        "forest_road", "formal_garden", "fountain",
                                                        "galley", "garage/indoor", "garage/outdoor",
                                                        "gas_station", "gazebo/exterior",
                                                        "general_store/indoor",
                                                        "general_store/outdoor", "gift_shop",
                                                        "glacier", "golf_course",
                                                        "greenhouse/indoor", "greenhouse/outdoor",
                                                        "grotto", "gymnasium/indoor",
                                                        "hangar/indoor",
                                                        "hangar/outdoor", "harbor",
                                                        "hardware_store", "hayfield", "heliport",
                                                        "highway", "home_office", "home_theater",
                                                        "hospital", "hospital_room",
                                                        "hot_spring", "hotel/outdoor", "hotel_room",
                                                        "house", "hunting_lodge/outdoor",
                                                        "ice_cream_parlor", "ice_floe", "ice_shelf",
                                                        "ice_skating_rink/indoor",
                                                        "ice_skating_rink/outdoor",
                                                        "iceberg", "igloo", "industrial_area",
                                                        "inn/outdoor", "islet",
                                                        "jacuzzi/indoor", "jail_cell",
                                                        "japanese_garden", "jewelry_shop",
                                                        "junkyard",
                                                        "kasbah", "kennel/outdoor",
                                                        "kindergarden_classroom", "kitchen",
                                                        "lagoon",
                                                        "lake/natural", "landfill", "landing_deck",
                                                        "laundromat", "lawn",
                                                        "lecture_room", "legislative_chamber",
                                                        "library/indoor", "library/outdoor",
                                                        "lighthouse",
                                                        "living_room", "loading_dock", "lobby",
                                                        "lock_chamber", "locker_room",
                                                        "mansion", "manufactured_home",
                                                        "market/indoor", "market/outdoor", "marsh",
                                                        "martial_arts_gym", "mausoleum", "medina",
                                                        "mezzanine", "moat/water",
                                                        "mosque/outdoor", "motel", "mountain",
                                                        "mountain_path", "mountain_snowy",
                                                        "movie_theater/indoor", "museum/indoor",
                                                        "museum/outdoor", "music_studio",
                                                        "natural_history_museum",
                                                        "nursery", "nursing_home", "oast_house",
                                                        "ocean", "office",
                                                        "office_building", "office_cubicles",
                                                        "oilrig", "operating_room", "orchard",
                                                        "orchestra_pit", "pagoda", "palace",
                                                        "pantry", "park",
                                                        "parking_garage/indoor",
                                                        "parking_garage/outdoor", "parking_lot",
                                                        "pasture", "patio",
                                                        "pavilion", "pet_shop", "pharmacy",
                                                        "phone_booth", "physics_laboratory",
                                                        "picnic_area", "pier", "pizzeria",
                                                        "playground", "playroom",
                                                        "plaza", "pond", "porch", "promenade",
                                                        "pub/indoor",
                                                        "racecourse", "raceway", "raft",
                                                        "railroad_track", "rainforest",
                                                        "reception", "recreation_room",
                                                        "repair_shop", "residential_neighborhood",
                                                        "restaurant",
                                                        "restaurant_kitchen", "restaurant_patio",
                                                        "rice_paddy", "river", "rock_arch",
                                                        "roof_garden", "rope_bridge", "ruin",
                                                        "runway", "sandbox",
                                                        "sauna", "schoolhouse", "science_museum",
                                                        "server_room", "shed",
                                                        "shoe_shop", "shopfront",
                                                        "shopping_mall/indoor", "shower",
                                                        "ski_resort",
                                                        "ski_slope", "sky", "skyscraper", "slum",
                                                        "snowfield",
                                                        "soccer_field", "stable",
                                                        "stadium/baseball", "stadium/football",
                                                        "stadium/soccer",
                                                        "stage/indoor", "stage/outdoor",
                                                        "staircase", "storage_room", "street",
                                                        "subway_station/platform", "supermarket",
                                                        "sushi_bar", "swamp", "swimming_hole",
                                                        "swimming_pool/indoor",
                                                        "swimming_pool/outdoor",
                                                        "synagogue/outdoor", "television_room",
                                                        "television_studio",
                                                        "temple/asia", "throne_room",
                                                        "ticket_booth", "topiary_garden", "tower",
                                                        "toyshop", "train_interior",
                                                        "train_station/platform", "tree_farm",
                                                        "tree_house",
                                                        "trench", "tundra", "underwater/ocean_deep",
                                                        "utility_room", "valley",
                                                        "vegetable_garden", "veterinarians_office",
                                                        "viaduct", "village", "vineyard",
                                                        "volcano", "volleyball_court/outdoor",
                                                        "waiting_room", "water_park", "water_tower",
                                                        "waterfall", "watering_hole", "wave",
                                                        "wet_bar", "wheat_field",
                                                        "wind_farm", "windmill", "yard",
                                                        "youth_hostel", "zen_garden"};


char *CreateLocalModelBuffer(JNIEnv *env, jobject modelBuffer) {
    jbyte *modelAddr = static_cast<jbyte *>(env->GetDirectBufferAddress(modelBuffer));
    int modelLen = static_cast<int>(env->GetDirectBufferCapacity(modelBuffer));
    char *buffer(new char[modelLen]);
    memcpy(buffer, modelAddr, modelLen);
    return buffer;
}

/**
 * To process the result of mindspore inference.
 * @param msOutputs
 * @return
 */
std::string ProcessRunnetResult(const int RET_CATEGORY_SUM, const char *const labels_name_map[],
                                std::unordered_map<std::string, mindspore::tensor::MSTensor *> msOutputs) {
    // Get the branch of the model output.
    // Use iterators to get map elements.
    std::unordered_map<std::string, mindspore::tensor::MSTensor *>::iterator iter;
    iter = msOutputs.begin();

    // The mobilenetv2.ms model output just one branch.
    auto outputTensor = iter->second;

    int tensorNum = outputTensor->ElementsNum();
    MS_PRINT("Number of tensor elements:%d", tensorNum);

    // Get a pointer to the first score.
    float *temp_scores = static_cast<float *>(outputTensor->MutableData());

    // float scores[RET_CATEGORY_SUM];
    float scores = temp_scores[0];
    int cat_loc = 0;
    for (int i = 0; i < RET_CATEGORY_SUM; ++i) {
        if (scores < temp_scores[i]) {
            scores = temp_scores[i];
            cat_loc = i;
        }
        if (temp_scores[i] > 0.5) {
            MS_PRINT("MindSpore scores[%d] : [%f]", i, temp_scores[i]);
        }
    }

    // Score for each category.
    // Converted to text information that needs to be displayed in the APP.
    std::string categoryScore = "";
    categoryScore += labels_name_map[cat_loc];
    categoryScore += ":";
    std::string score_str = std::to_string(scores);
    categoryScore += score_str;
    return categoryScore;
}

bool BitmapToLiteMat(JNIEnv *env, const jobject &srcBitmap, LiteMat *lite_mat) {
    bool ret = false;
    AndroidBitmapInfo info;
    void *pixels = nullptr;
    LiteMat &lite_mat_bgr = *lite_mat;
    AndroidBitmap_getInfo(env, srcBitmap, &info);
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        MS_PRINT("Image Err, Request RGBA");
        return false;
    }
    AndroidBitmap_lockPixels(env, srcBitmap, &pixels);
    if (info.stride == info.width * 4) {
        ret = InitFromPixel(reinterpret_cast<const unsigned char *>(pixels),
                            LPixelType::RGBA2RGB, LDataType::UINT8,
                            info.width, info.height, lite_mat_bgr);
        if (!ret) {
            MS_PRINT("Init From RGBA error");
        }
    } else {
        unsigned char *pixels_ptr = new unsigned char[info.width * info.height * 4];
        unsigned char *ptr = pixels_ptr;
        unsigned char *data = reinterpret_cast<unsigned char *>(pixels);
        for (int i = 0; i < info.height; i++) {
            memcpy(ptr, data, info.width * 4);
            ptr += info.width * 4;
            data += info.stride;
        }
        ret = InitFromPixel(reinterpret_cast<const unsigned char *>(pixels_ptr),
                            LPixelType::RGBA2RGB, LDataType::UINT8,
                            info.width, info.height, lite_mat_bgr);
        if (!ret) {
            MS_PRINT("Init From RGBA error");
        }
        delete[] (pixels_ptr);
    }
    AndroidBitmap_unlockPixels(env, srcBitmap);
    return ret;
}

bool PreProcessImageData(const LiteMat &lite_mat_bgr, LiteMat *lite_norm_mat_ptr) {
    bool ret = false;
    LiteMat lite_mat_resize;
    LiteMat &lite_norm_mat_cut = *lite_norm_mat_ptr;
    ret = ResizeBilinear(lite_mat_bgr, lite_mat_resize, 256, 256);
    if (!ret) {
        MS_PRINT("ResizeBilinear error");
        return false;
    }
    LiteMat lite_mat_convert_float;
    ret = ConvertTo(lite_mat_resize, lite_mat_convert_float, 1.0 / 255.0);
    if (!ret) {
        MS_PRINT("ConvertTo error");
        return false;
    }
    LiteMat lite_mat_cut;
    ret = Crop(lite_mat_convert_float, lite_mat_cut, 16, 16, 224, 224);
    if (!ret) {
        MS_PRINT("Crop error");
        return false;
    }
    std::vector<float> means = {0.485, 0.456, 0.406};
    std::vector<float> stds = {0.229, 0.224, 0.225};
    SubStractMeanNormalize(lite_mat_cut, lite_norm_mat_cut, means, stds);
    return true;
}


/**
 * The Java layer reads the model into MappedByteBuffer or ByteBuffer to load the model.
 */
extern "C"
JNIEXPORT jlong JNICALL
Java_com_mindspore_scene_gallery_classify_TrackingMobile_loadModel(JNIEnv *env,
                                                                   jobject thiz,
                                                                   jobject model_buffer,
                                                                   jint num_thread) {
    if (nullptr == model_buffer) {
        MS_PRINT("error, buffer is nullptr!");
        return (jlong) nullptr;
    }
    jlong bufferLen = env->GetDirectBufferCapacity(model_buffer);
    if (0 == bufferLen) {
        MS_PRINT("error, bufferLen is 0!");
        return (jlong) nullptr;
    }

    char *modelBuffer = CreateLocalModelBuffer(env, model_buffer);
    if (modelBuffer == nullptr) {
        MS_PRINT("modelBuffer create failed!");
        return (jlong) nullptr;
    }

    // To create a mindspore network inference environment.
    void **labelEnv = new void *;
    MSNetWork *labelNet = new MSNetWork;
    *labelEnv = labelNet;

    mindspore::lite::Context *context = new mindspore::lite::Context;
    context->thread_num_ = num_thread;
    context->device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = mindspore::lite::NO_BIND;
    context->device_list_[0].device_info_.cpu_device_info_.enable_float16_ = false;
    context->device_list_[0].device_type_ = mindspore::lite::DT_CPU;

    labelNet->CreateSessionMS(modelBuffer, bufferLen, context);
    delete context;

    if (labelNet->session() == nullptr) {
        MS_PRINT("MindSpore create session failed!.");
        delete labelNet;
        delete labelEnv;
        return (jlong) nullptr;
    }

    if (model_buffer != nullptr) {
        env->DeleteLocalRef(model_buffer);
    }

    return (jlong) labelEnv;
}

/**
 * After the inference environment is successfully created,
 * sending a picture to the model and run inference.
 */
extern "C" JNIEXPORT jstring JNICALL
Java_com_mindspore_scene_gallery_classify_TrackingMobile_runNet(JNIEnv *env, jclass type,
                                                                jlong netEnv,
                                                                jobject srcBitmap) {
    LiteMat lite_mat_bgr, lite_norm_mat_cut;

    if (!BitmapToLiteMat(env, srcBitmap, &lite_mat_bgr)) {
        MS_PRINT("BitmapToLiteMat error");
        return NULL;
    }
    if (!PreProcessImageData(lite_mat_bgr, &lite_norm_mat_cut)) {
        MS_PRINT("PreProcessImageData error");
        return NULL;
    }

    ImgDims inputDims;
    inputDims.channel = lite_norm_mat_cut.channel_;
    inputDims.width = lite_norm_mat_cut.width_;
    inputDims.height = lite_norm_mat_cut.height_;

    // Get the mindsore inference environment which created in loadModel().
    void **labelEnv = reinterpret_cast<void **>(netEnv);
    if (labelEnv == nullptr) {
        MS_PRINT("MindSpore error, labelEnv is a nullptr.");
        return NULL;
    }
    MSNetWork *labelNet = static_cast<MSNetWork *>(*labelEnv);

    auto mSession = labelNet->session();
    if (mSession == nullptr) {
        MS_PRINT("MindSpore error, Session is a nullptr.");
        return NULL;
    }
    MS_PRINT("MindSpore get session.");

    auto msInputs = mSession->GetInputs();
    if (msInputs.size() == 0) {
        MS_PRINT("MindSpore error, msInputs.size() equals 0.");
        return NULL;
    }
    auto inTensor = msInputs.front();

    float *dataHWC = reinterpret_cast<float *>(lite_norm_mat_cut.data_ptr_);
    // Copy dataHWC to the model input tensor.
    memcpy(inTensor->MutableData(), dataHWC,
           inputDims.channel * inputDims.width * inputDims.height * sizeof(float));

    // After the model and image tensor data is loaded, run inference.
    auto status = mSession->RunGraph();

    if (status != mindspore::lite::RET_OK) {
        MS_PRINT("MindSpore run net error.");
        return NULL;
    }

    /**
     * Get the mindspore inference results.
     * Return the map of output node name and MindSpore Lite MSTensor.
     */
    auto names = mSession->GetOutputTensorNames();
    std::unordered_map<std::string, mindspore::tensor::MSTensor *> msOutputs;
    for (const auto &name : names) {
        auto temp_dat = mSession->GetOutputByTensorName(name);
        msOutputs.insert(std::pair<std::string, mindspore::tensor::MSTensor *>{name, temp_dat});
    }

    std::string resultStr = ProcessRunnetResult(::RET_CATEGORY_SUM,
                                                ::labels_name_map, msOutputs);

    const char *resultCharData = resultStr.c_str();
    return (env)->NewStringUTF(resultCharData);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mindspore_scene_gallery_classify_TrackingMobile_unloadModel(JNIEnv *env,
                                                                     jclass type,
                                                                     jlong netEnv) {
    MS_PRINT("MindSpore release net.");
    void **labelEnv = reinterpret_cast<void **>(netEnv);
    if (labelEnv == nullptr) {
        MS_PRINT("MindSpore error, labelEnv is a nullptr.");
    }
    MSNetWork *labelNet = static_cast<MSNetWork *>(*labelEnv);

    labelNet->ReleaseNets();

    return (jboolean) true;
}
