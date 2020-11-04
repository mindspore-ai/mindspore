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
     * JNI loading model
     *
     * @param assetManager assetManager
     * @param buffer buffer
     * @param numThread numThread
     * @return Load model data
     */
    public native long loadModel(AssetManager assetManager, ByteBuffer buffer, int numThread);

    /**
     *  JNI run model
     * 
     * @param netEnv Load model data
     * @param img Current picture
     * @return Run model data
     */
    public native String runNet(long netEnv, Bitmap img);

    /**
     * Unbind model data
     * 
     * @param netEnv model data
     * @return Unbound state
     */
    public native boolean unloadModel(long netEnv);

    /**
     * C++ Methods encapsulated into the msnetworks class
     * 
     * @param assetManager Model file location
     * @return Loading model file status
     */
    public boolean loadModelFromBuf(AssetManager assetManager) {
        String ModelPath = "model/ssd.ms";
        ByteBuffer buffer = loadModelFile(ModelPath);
        netEnv = loadModel(assetManager, buffer, 2);
        return true;
    }

    /**
     * Run Mindspore
     * 
     * @param img Current image recognition
     * @return Recognized text information
     */
    public String MindSpore_runnet(Bitmap img) {
        String ret_str = runNet(netEnv, img);
        return ret_str;
    }

    /**
     * Unbound model
     * @return true
     */
    public boolean unloadModel() {
        unloadModel(netEnv);
        return true;
    }

    /**
     * Load model file stream
     * @param modelPath Model file path
     * @return  Load model file stream
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
