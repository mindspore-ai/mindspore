package com.mindspore.hiobject.help;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import java.io.FileNotFoundException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.HashMap;

public class TrackingMobile {
    private final static String TAG = "TrackingMobile";

    static {
        try {
            System.loadLibrary("mlkit-label-MS");
            Log.i(TAG, "load libiMindSpore.so successfully.");
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "UnsatisfiedLinkError " + e.getMessage());
        }
    }

    public static HashMap<Integer, String> synset_words_map = new HashMap<>();

    public static float[] threshold = new float[494];

    private long netEnv = 0;

    private final Context mActivity;

    public TrackingMobile(Context activity) throws FileNotFoundException {
        this.mActivity = activity;
    }

    /**
     * jni加载模型
     *
     * @param assetManager assetManager
     * @param buffer buffer
     * @param numThread numThread
     * @return 加载模型数据
     */
    public native long loadModel(AssetManager assetManager, ByteBuffer buffer, int numThread);

    /**
     * jni运行模型
     * 
     * @param netEnv 加载模型数据
     * @param img 当前图片
     * @return 运行模型数据
     */
    public native String runNet(long netEnv, Bitmap img);

    /**
     * 解绑模型数据
     * 
     * @param netEnv 模型数据
     * @return 解绑状态
     */
    public native boolean unloadModel(long netEnv);

    /**
     * C++侧封装成了MSNetWorks类的方法
     * 
     * @param assetManager 模型文件位置
     * @return 加载模型文件状态
     */
    public boolean loadModelFromBuf(AssetManager assetManager) {
//        String ModelPath = "model/model_hebing_3branch.ms";
        String ModelPath = "model/ssd.ms";

        ByteBuffer buffer = loadModelFile(ModelPath);
        netEnv = loadModel(assetManager, buffer, 2);
        return true;
    }

    /**
     * 运行Mindspore
     * 
     * @param img 当前图片识别
     * @return 识别出来的文字信息
     */
    public String MindSpore_runnet(Bitmap img) {
        String ret_str = runNet(netEnv, img);
        return ret_str;
    }

    /**
     * 解绑模型
     * @return true
     */
    public boolean unloadModel() {
        unloadModel(netEnv);
        return true;
    }

    /**
     * 加载模型文件流
     * @param modelPath 模型文件路径
     * @return  加载模型文件流
     */
    public ByteBuffer loadModelFile(String modelPath) {
        InputStream is = null;
        try {
            is = mActivity.getAssets().open(modelPath);
            byte[] bytes = new byte[is.available()];
            is.read(bytes);
            return ByteBuffer.allocateDirect(bytes.length).put(bytes);
        } catch (Exception e) {
            Log.d("loadModelFile", " Exception occur ");
            e.printStackTrace();
        }
        return null;
    }

}
