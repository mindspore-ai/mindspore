package com.mindspore.flclient.model;

import com.mindspore.Graph;
import com.mindspore.MSTensor;
import com.mindspore.Model;
import com.mindspore.config.MSContext;
import com.mindspore.config.ModelType;
import com.mindspore.config.TrainCfg;
import com.mindspore.flclient.LocalFLParameter;
import com.mindspore.flclient.common.FLLoggerGenerater;
import mindspore.schema.FeatureMap;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;
import java.util.stream.IntStream;

/**
 * while the train/eval model path is same,  train/eval share input/feature/output
 *
 * @author : [zhangzhaoju]
 * @since : [2022/5/30]
 */
public class ModelProxy {
    private static final Logger logger = FLLoggerGenerater.getModelLogger(Client.class.toString());
    /**
     * Mindspore model object.
     */
    private Model model;

    /**
     * dataset map.
     */
    private Map<RunType, DataSet> dataSets = new HashMap<>();

    private final List<ByteBuffer> inputsBuffer = new ArrayList<>();

    private List<MSTensor> inputs;

    private HashMap<String, MSTensor> featureMap = new HashMap<>();

    private float uploadLoss = 0.0f;

    public Model getModel() {
        return model;
    }

    public float getUploadLoss() {
        return uploadLoss;
    }

    public void setUploadLoss(float uploadLoss) {
        this.uploadLoss = uploadLoss;
    }

    /**
     * Free client.
     */
    public void free() {
        if (model != null) {
            inputs.forEach(MSTensor::free);
            featureMap.forEach((t, v) -> v.free());
            model.free();
            model = null;
        }
    }

    private MSContext getMsContext() {
        int deviceType = LocalFLParameter.getInstance().getDeviceType();
        int threadNum = LocalFLParameter.getInstance().getThreadNum();
        int cpuBindMode = LocalFLParameter.getInstance().getCpuBindMode();
        boolean enableFp16 = LocalFLParameter.getInstance().isEnableFp16();
        MSContext msContext = new MSContext();
        // use default param init context
        if(!msContext.init(threadNum, cpuBindMode)){
            logger.severe("Call msContext.init failed, threadNum " + threadNum + ", cpuBindMode " + cpuBindMode);
            msContext.free();
            return null;
        }

        if (!msContext.addDeviceInfo(deviceType, enableFp16, 0)) {
            logger.severe("Call msContext.addDeviceInfo failed, deviceType " + deviceType + ", enableFp16 " + enableFp16);
            msContext.free();
            return null;
        }
        return msContext;
    }

    /**
     * init model without shapes, this could be train or infer model
     *
     * @return init status.
     */
    private boolean initModelWithoutShape(String modelPath, MSContext msContext) {
        TrainCfg trainCfg = new TrainCfg();
        if(!trainCfg.init()){
            logger.severe("Call trainCfg.init failed ...");
            msContext.free();
            trainCfg.free();
            return false;
        }
        Graph graph = new Graph();
        if(!graph.load(modelPath)){
            logger.severe("Call graph.load failed, modelPath: " + modelPath);
            graph.free();
            trainCfg.free();
            msContext.free();
            return false;
        }
        model = new Model();
        if (!model.build(graph, msContext, trainCfg)) {
            // The Jni implement will change msContext & graph to shared_ptr, no need free here
            logger.severe("Call model.build failed ... ");
            graph.free();
            model.free();
            return false;
        }
        graph.free();

        inputs = model.getInputs();
        for (MSTensor input : inputs) {
            ByteBuffer inputBuffer = ByteBuffer.allocateDirect((int) input.size());
            inputBuffer.order(ByteOrder.nativeOrder());
            inputsBuffer.add(inputBuffer);
//            input.free();
        }
        List<MSTensor> features = model.getFeatureMaps();
        for(MSTensor item: features){
            featureMap.put(item.tensorName(), item);
        }
        return true;
    }

    /**
     * init model with shapes, this is a infer model
     *
     * @return init status.
     */
    private boolean initModelWithShape(String modelPath, MSContext msContext, int[][] inputShapes) {
        model = new Model();
        if (!model.build(modelPath, ModelType.MT_MINDIR, msContext)) {
            // The Jni implement will change msContext & graph to shared_ptr, no need free here
            logger.severe("Call model.build failed ... ");
            model.free();
            return false;
        }
        inputs = model.getInputs();
        boolean isSuccess = model.resize(inputs, inputShapes);
//        inputs.forEach(MSTensor::free);
        if (!isSuccess) {
            model.free();
            logger.severe("session resize failed");
            return false;
        }
        for (int[] shapes : inputShapes) {
            int size = IntStream.of(shapes).reduce((a, b) -> a * b).getAsInt() * Integer.BYTES;
            ByteBuffer inputBuffer = ByteBuffer.allocateDirect(size);
            inputBuffer.order(ByteOrder.nativeOrder());
            inputsBuffer.add(inputBuffer);
        }
        List<MSTensor> features = model.getFeatureMaps();
        for(MSTensor item: features){
            featureMap.put(item.tensorName(), item);
        }
        return true;
    }

    /**
     * Init Model.
     *
     * @param modelPath model file path.
     * @param inputShapes input shape of model.
     * @return execute status.
     */
    public Status initModel(String modelPath, int[][] inputShapes) {
        if (modelPath == null) {
            logger.severe("session init failed");
            return Status.FAILED;
        }

        MSContext msContext = getMsContext();
        if (msContext == null) {
            return Status.FAILED;
        }
        boolean initModelRet = inputShapes == null ?
                initModelWithoutShape(modelPath, msContext) : initModelWithShape(modelPath, msContext, inputShapes);
        return initModelRet ? Status.SUCCESS : Status.FAILED;
    }

    private void fillModelInput(DataSet dataSet, int batchIdx) {
        dataSet.fillInputBuffer(inputsBuffer, batchIdx);
        for (int i = 0; i < inputs.size(); i++) {
            inputs.get(i).setData(inputsBuffer.get(i));
        }
    }

    public Status runModel(int epochs, List<Callback> callbacks, DataSet dataSet) {
        LocalFLParameter localFLParameter = LocalFLParameter.getInstance();
        long startTime = System.currentTimeMillis();
        for (int i = 0; i < epochs; i++) {
            for (int j = 0; j < dataSet.batchNum; j++) {
                if (localFLParameter.isStopJobFlag()) {
                    logger.info("the stopJObFlag is set to true, the job will be stop");
                    return Status.FAILED;
                }
                fillModelInput(dataSet, j);
                boolean isSuccess = model.runStep();
                if (!isSuccess) {
                    logger.severe("run graph failed");
                    return Status.FAILED;
                }
                for (Callback callBack : callbacks) {
                    callBack.stepEnd();
                }
            }
            for (Callback callBack : callbacks) {
                callBack.epochEnd();
                if (callBack instanceof LossCallback && i == epochs - 1) {
                    LossCallback lossCallback = (LossCallback)callBack;
                    setUploadLoss(lossCallback.getUploadLoss());
                }
            }
        }
        long endTime = System.currentTimeMillis();
        logger.info("total run time:" + (endTime - startTime) + "ms");
        return Status.SUCCESS;
    }

    /**
     * Get model feature maps with float value.
     *
     * @return model weights.
     */
    public Map<String, float[]> getFeatureMap() {
        Map<String, float[]> features = new HashMap<>(featureMap.size());
        for (Map.Entry<String, MSTensor> entry : featureMap.entrySet()) {
            features.put(entry.getKey(), entry.getValue().getFloatData());
        }
        return features;
    }

    /**
     * Get model feature with name.
     *
     * @return model weight.
     */
    public float[] getFeature(String weightName) {
        if (featureMap.containsKey(weightName)) {
            return featureMap.get(weightName).getFloatData();
        }
        return null;
    }

    /**
     * update model feature maps.
     *
     * @param modelName model file name.
     * @param featureMaps new weights.
     * @return update status.
     */
    public Status updateFeatures(String modelName, List<FeatureMap> featureMaps) {
        if (model == null || featureMaps == null || modelName == null || modelName.isEmpty()) {
            logger.severe("trainSession,featureMaps modelName cannot be null");
            return Status.NULLPTR;
        }

        List<MSTensor> tensors = new ArrayList<>(featureMaps.size());
        for (FeatureMap newFeature : featureMaps) {
            if (newFeature == null) {
                logger.severe("newFeature cannot be null");
                return Status.NULLPTR;
            }

            if (newFeature.weightFullname().isEmpty() || !featureMap.containsKey(newFeature.weightFullname())) {
                logger.severe("Can't get feature for name:" + newFeature.weightFullname());
                return Status.NULLPTR;
            }
            MSTensor tensor = featureMap.get(newFeature.weightFullname());
            ByteBuffer by = newFeature.dataAsByteBuffer();
            ByteBuffer newData = ByteBuffer.allocateDirect(by.remaining());
            newData.order(ByteOrder.nativeOrder());
            newData.put(by);
            if (!tensor.setData(newData)) {
                logger.severe("Set tensor value failed, name:" + tensor.tensorName());
                return Status.FAILED;
            }
        }
        model.export(modelName, 0, false, null);
        return Status.SUCCESS;
    }

    /**
     * update model feature
     *
     * @param newFeature new weights.
     * @return update status.
     */
    public Status updateFeature(FeatureMap newFeature) {
        if (newFeature == null) {
            logger.severe("newFeature cannot be null");
            return Status.NULLPTR;
        }

        if (newFeature.weightFullname().isEmpty() || !featureMap.containsKey(newFeature.weightFullname())) {
            logger.severe("Can't get feature for name:" + newFeature.weightFullname());
            return Status.NULLPTR;
        }
        MSTensor tensor = featureMap.get(newFeature.weightFullname());
        ByteBuffer by = newFeature.dataAsByteBuffer();
        ByteBuffer newData = ByteBuffer.allocateDirect(by.remaining());
        newData.order(ByteOrder.nativeOrder());
        newData.put(by);
        if (!tensor.setData(newData)) {
            logger.severe("Set tensor value failed, name:" + tensor.tensorName());
            return Status.FAILED;
        }
        return Status.SUCCESS;
    }
}
