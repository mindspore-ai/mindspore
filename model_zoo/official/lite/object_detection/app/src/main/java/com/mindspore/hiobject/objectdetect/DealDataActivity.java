package com.mindspore.hiobject.objectdetect;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import com.mindspore.hiobject.R;
import com.mindspore.hiobject.help.TrackingMobile;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * [入口主页面]
 * <p>
 * 向JNI传入图片，测试MindSpore模型加载推理等.
 */

public class DealDataActivity extends AppCompatActivity {
    private final String TAG = "DealDataActivity";

    //自行将v2017的图片放入手机sdcard的位置
    private final static String IMGPATH = "/sdcard/val2017";
    private final static String IMG_RESULT_PATH = "/sdcard/val2017result/result.txt";
    private final static String IMG_RESULT_SINGLE_PATH = "/sdcard/val2017result/result2.txt";

    private Bitmap mBitmap;
    private TrackingMobile mTrackingMobile;

    private static final String PERMISSION_READ_EXTERNAL_STORAGEA = Manifest.permission.READ_EXTERNAL_STORAGE;
    private static final String PERMISSION_WRITE_EXTERNAL_STORAGEA = Manifest.permission.WRITE_EXTERNAL_STORAGE;

    private static final int PERMISSIONS_REQUEST = 1;


    private Handler handler = new Handler() {
        @Override
        public void handleMessage(@NonNull Message msg) {
            super.handleMessage(msg);
            if (1 == msg.what) {
                dealData();
              //  dealSingleData();
            }
        }
    };


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_dealdata);


        try {
            mTrackingMobile = new TrackingMobile(this);
        } catch (FileNotFoundException e) {
            Log.e(TAG, Log.getStackTraceString(e));
        }
        mTrackingMobile.loadModelFromBuf(getAssets());

        if (hasPermission()) {
            getImgFileList();
        } else {
            requestPermission();
        }

    }


    private List<String> imgFileList;

    private void getImgFileList() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                imgFileList = getFilesAllName(IMGPATH);
                Message message = new Message();
                message.what = 1;
                handler.sendMessage(message);
            }
        }).start();
    }


    List<String> dealList = new ArrayList<>();

    private void dealData() {
        if (imgFileList != null && imgFileList.size() > 0) {
            Log.d(TAG, "imgFileList size()>>" + imgFileList.size());
            for (int i = 0; i < imgFileList.size(); i++) {
                Bitmap bitmap = BitmapFactory.decodeFile(imgFileList.get(i)).copy(Bitmap.Config.ARGB_8888, true);

                String result = mTrackingMobile.MindSpore_runnet(bitmap);
                String fileName = imgFileList.get(i).substring(imgFileList.get(i).lastIndexOf("/") + 1);
                Log.d(TAG, "index>>>" + i + ">>" + fileName + ">>result" + result);
                StringBuilder sb = new StringBuilder();
                sb.append(fileName).append("_").append(result);
                dealList.add(sb.toString());
            }
            Log.d(TAG, "dealList >>>" + dealList.size());
            writeListIntoSDcard(IMG_RESULT_PATH, dealList);
        }
    }

    private void dealSingleData() {
        String fileFullName = IMGPATH + "/error.jpg";
        Bitmap bitmap =  BitmapFactory.decodeResource(getResources(),R.drawable.error).copy(Bitmap.Config.ARGB_8888, true);
//        Bitmap bitmap = BitmapFactory.decodeFile(fileFullName).copy(Bitmap.Config.ARGB_8888, true);
        if (bitmap != null) {

            String result = mTrackingMobile.MindSpore_runnet(bitmap);
            Log.d(TAG,  ">>result" + result);
            StringBuilder sb = new StringBuilder();
            sb.append("error.jpg").append("_").append(result);
//            writeStringIntoSDcard(IMG_RESULT_SINGLE_PATH, sb.toString());
        }
    }


    public boolean writeListIntoSDcard(String fileName, List<String> list) {
        if (Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED)) {
            File sdFile = new File(fileName);
            try {
                FileOutputStream fos = new FileOutputStream(sdFile);
                ObjectOutputStream oos = new ObjectOutputStream(fos);
                oos.writeObject(list);//写入
                fos.close();
                oos.close();
                return true;
            } catch (FileNotFoundException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
                return false;
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
                return false;
            }
        } else {
            return false;
        }
    }


    public boolean writeStringIntoSDcard(String fileName, String content) {
        if (Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED)) {
            File sdFile = new File(fileName);
            try {
                FileOutputStream fos = new FileOutputStream(sdFile);
                ObjectOutputStream oos = new ObjectOutputStream(fos);
                oos.writeObject(content);//写入
                fos.close();
                oos.close();
                return true;
            } catch (FileNotFoundException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
                return false;
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
                return false;
            }
        } else {
            return false;
        }
    }

    @Override
    public void onRequestPermissionsResult(final int requestCode, final String[] permissions,
                                           final int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSIONS_REQUEST) {
            if (allPermissionsGranted(grantResults)) {
                getImgFileList();
            } else {
                requestPermission();
            }
        }
    }

    private static boolean allPermissionsGranted(final int[] grantResults) {
        for (int result : grantResults) {
            if (result != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(PERMISSION_READ_EXTERNAL_STORAGEA) == PackageManager.PERMISSION_GRANTED &&
                    checkSelfPermission(PERMISSION_WRITE_EXTERNAL_STORAGEA) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (shouldShowRequestPermissionRationale(PERMISSION_READ_EXTERNAL_STORAGEA)) {
                Toast.makeText(this, "Read permission is required for this demo", Toast.LENGTH_LONG)
                        .show();
            }
            if (shouldShowRequestPermissionRationale(PERMISSION_WRITE_EXTERNAL_STORAGEA)) {
                Toast.makeText(this, "WRITE permission is required for this demo", Toast.LENGTH_LONG)
                        .show();
            }
            requestPermissions(new String[]{PERMISSION_READ_EXTERNAL_STORAGEA, PERMISSION_WRITE_EXTERNAL_STORAGEA}, PERMISSIONS_REQUEST);
        }
    }


    public List<String> getFilesAllName(String path) {
        //传入指定文件夹的路径
        File file = new File(path);
        if (null == file || !file.isDirectory()) {
            return null;
        }
        File[] files = file.listFiles();
        List<String> imagePaths = new ArrayList<>();
        for (int i = 0; i < files.length; i++) {
            if (checkIsImageFile(files[i].getPath())) {
                imagePaths.add(files[i].getPath());
            }
        }
        return imagePaths;
    }

    /**
     * 判断是否是照片
     */
    public boolean checkIsImageFile(String fName) {
        boolean isImageFile = false;
        //获取拓展名
        String fileEnd = fName.substring(fName.lastIndexOf(".") + 1,
                fName.length()).toLowerCase();
        if (fileEnd.equals("jpg") || fileEnd.equals("png") || fileEnd.equals("gif")
                || fileEnd.equals("jpeg") || fileEnd.equals("bmp")) {
            isImageFile = true;
        } else {
            isImageFile = false;
        }
        return isImageFile;
    }
}
