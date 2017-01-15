package com.posh.opencv1;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.View.OnTouchListener;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;


import static org.opencv.imgcodecs.Imgcodecs.imwrite;

public class ColorBlobDetectionActivity extends Activity implements OnTouchListener, CvCameraViewListener2 {
    private static final String  TAG              = "OCVSample::Activity";

    private boolean              mIsColorSelected = false;
    private Mat                  mRgba;
    private Scalar               mBlobColorRgba;
    private Scalar               mBlobColorHsv;
    private ColorBlobDetector    mDetector;
    private Mat                  mSpectrum;
    private Size                 SPECTRUM_SIZE;
    private Scalar               CONTOUR_COLOR;
    private String               cpicturename;
    private ImageView            imageView;
    private Mat                  mH;
    private Mat                  mS;
    private Mat                  mV;
    private Mat                  mRGBcut;


    private CameraBridgeViewBase mOpenCvCameraView;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    mOpenCvCameraView.setOnTouchListener(ColorBlobDetectionActivity.this);
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public ColorBlobDetectionActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.color_blob_detection_surface_view);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.color_blob_detection_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);

        // Take photo
        ImageButton takePhoto = (ImageButton)findViewById(R.id.tpictureimageButton);
        takePhoto.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                cpicturename = "" + new SimpleDateFormat("dd-MM-yyyy_HH:mm:ss").format(new java.util.Date()).toString();
                saveImageToDisk(mRgba, cpicturename, "picture", ColorBlobDetectionActivity.this, 1);

            }
        });
        Button takeDivideHSV = (Button)findViewById(R.id.HSVbutton);
        takeDivideHSV.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        Mat bn = createBinaryImage(mRgba);
                        Mat op = openning(bn);
                        Mat colorC = colorComponent(op);
                        Mat coms = connectionComponent(op);
                        Mat onlycom = Onlyteeth(coms, 100);

                        divideHSV(onlycom);
                        // save HSV image that only teeth
                        File root = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
                        String dir = root+"/"+"Matching Teeth";
                        cpicturename = "" + new SimpleDateFormat("dd-MM-yyyy_HH:mm:ss").format(new java.util.Date()).toString();
                        Mat histH = calHist(mH);
                        Mat histS = calHist(mS);
                        Mat histV = calHist(mV);
                        int hH = maxF(histH);
                        int hS = maxF(histS);
                        int hV = maxF(histV);
                        Log.d(TAG, "DivideHSV: HiHist H : "+hH+" S : "+hS+" V : "+hV);

                        //imwrite(dir+cpicturename+"H"+".jpg",mH);
                        //imwrite(dir+cpicturename+"S"+".jpg",mS);
                        imwrite(dir+cpicturename+"V"+".jpg",mV);
                        imwrite(dir+cpicturename+"colorC"+".jpg",colorC);
                        imwrite(dir+cpicturename+"only"+".jpg",onlycom);
                    }
                }
        );

        /*ImageButton viewPicture = (ImageButton) findViewById(R.id.viewpic);
        viewPicture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                ;
                //mp.start();
                cpicturename = ""+ new SimpleDateFormat("yyyyMMdd_HHmmss").format(new java.util.Date()).toString();
                loadImageFromFile("picture",ColorBlobDetectionActivity.this);

            }
        });*/


    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mDetector = new ColorBlobDetector();
        mSpectrum = new Mat();
        mBlobColorRgba = new Scalar(255);
        mBlobColorHsv = new Scalar(255);
        SPECTRUM_SIZE = new Size(200, 64);
        CONTOUR_COLOR = new Scalar(255,0,0,255);
    }

    public void onCameraViewStopped() {
        mRgba.release();
    }

    public boolean onTouch(View v, MotionEvent event) {

        /*if (this.getPackageManager().hasSystemFeature(PackageManager.FEATURE_CAMERA_FLASH))
        {
            // open flashlight
          //  mJavaCameraView.openFlash();
            //Camera cam = Camera.open();
           // mJavaCameraView.mCamera
           // Camera.Parameters p = cam.getParameters();
           // p.setFlashMode(Camera.Parameters.FLASH_MODE_TORCH);
            //cam.setParameters(p);
            //cam.startPreview();
        }*/

       /* int cols = mRgba.cols();
        int rows = mRgba.rows();

        int xOffset = (mOpenCvCameraView.getWidth() - cols) / 2;
        int yOffset = (mOpenCvCameraView.getHeight() - rows) / 2;

        int x = (int)event.getX() - xOffset;
        int y = (int)event.getY() - yOffset;

        Log.i(TAG, "Touch image coordinates: (" + x + ", " + y + ")");

        if ((x < 0) || (y < 0) || (x > cols) || (y > rows)) return false;

        Rect touchedRect = new Rect();

        touchedRect.x = (x>4) ? x-4 : 0;
        touchedRect.y = (y>4) ? y-4 : 0;

        touchedRect.width = (x+4 < cols) ? x + 4 - touchedRect.x : cols - touchedRect.x;
        touchedRect.height = (y+4 < rows) ? y + 4 - touchedRect.y : rows - touchedRect.y;

        Mat touchedRegionRgba = mRgba.submat(touchedRect);

        Mat touchedRegionHsv = new Mat();
        Imgproc.cvtColor(touchedRegionRgba, touchedRegionHsv, Imgproc.COLOR_RGB2HSV_FULL);

        // Calculate average color of touched region
        mBlobColorHsv = Core.sumElems(touchedRegionHsv);
        int pointCount = touchedRect.width*touchedRect.height;
        for (int i = 0; i < mBlobColorHsv.val.length; i++)
            mBlobColorHsv.val[i] /= pointCount;

        mBlobColorRgba = converScalarHsv2Rgba(mBlobColorHsv);

        Log.i(TAG, "Touched rgba color: (" + mBlobColorRgba.val[0] + ", " + mBlobColorRgba.val[1] +
                ", " + mBlobColorRgba.val[2] + ", " + mBlobColorRgba.val[3] + ")");

        mDetector.setHsvColor(mBlobColorHsv);

        Imgproc.resize(mDetector.getSpectrum(), mSpectrum, SPECTRUM_SIZE);

        mIsColorSelected = true;

        touchedRegionRgba.release();
        touchedRegionHsv.release();
    */
        return false; // don't need subsequent touch events
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();

        if (mIsColorSelected) {
            mDetector.process(mRgba);
            List<MatOfPoint> contours = mDetector.getContours();
            Log.e(TAG, "Contours count: " + contours.size());
            Imgproc.drawContours(mRgba, contours, -1, CONTOUR_COLOR);

            Mat colorLabel = mRgba.submat(4, 68, 4, 68);
            colorLabel.setTo(mBlobColorRgba);

            Mat spectrumLabel = mRgba.submat(4, 4 + mSpectrum.rows(), 70, 70 + mSpectrum.cols());
            mSpectrum.copyTo(spectrumLabel);
        }

        return mRgba;
    }

    private Scalar converScalarHsv2Rgba(Scalar hsvColor) {
        Mat pointMatRgba = new Mat();
        Mat pointMatHsv = new Mat(1, 1, CvType.CV_8UC3, hsvColor);
        Imgproc.cvtColor(pointMatHsv, pointMatRgba, Imgproc.COLOR_HSV2RGB_FULL, 4);

        return new Scalar(pointMatRgba.get(0, 0));
    }

    private void saveImageToDisk(Mat source, String filename, String directoryName,
                                 Context ctx, int colorConversion){

        Mat mat = source.clone();

        if(colorConversion != -1)
            Imgproc.cvtColor(mat, mat, colorConversion, 4); //1 3 ภาพดิบๆ ไม่แถบ สเปคตรัม
        //Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGB2GRAY); // openCV
        //Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGB2HSV);
        //COLOR_RGB2BGR COLOR_RGB2BGRA COLOR_RGB2GRAY COLOR_RGB2HLS COLOR_RGB2HSV COLOR_RGB2Luv

        // change matrix(Mat) to pixel(Bitmap)
        // Bitmap.Config.ARGB_8888 is stored 4 bytes per pixel
        Bitmap bmpOut = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat, bmpOut);

        //create Binary Image
        Mat mat2 = createBinaryImage(mRgba);
        Bitmap bmpOut2 = Bitmap.createBitmap(mat2.cols(), mat2.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat2, bmpOut2);

        //Openning ,mat2 is binaryImage
        Mat mat3 = openning(mat2);
        Bitmap bmpOut3 = Bitmap.createBitmap(mat3.cols(), mat3.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat3,bmpOut3);

        //grabCut  cutOnlyTeeth(mat3)
        Mat mat4 = connectionComponent(mat3);
        Mat mat44 = colorComponent(mat3);
        //int[] sizeOfComponent = numberOfComponent(mat4);
        //Mat mat5 = Onlyteeth(mat4, sizeOfComponent);
        Bitmap bmpOut4 = Bitmap.createBitmap(mat44.cols(), mat44.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat44,bmpOut4);
        //int[][] centroid = centroid(mat4);

        Mat mat5 = Onlyteeth(mat4, 200);
        Bitmap bmpOut5 = Bitmap.createBitmap(mat5.cols(), mat5.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat5,bmpOut5);

        //Mat mat6 = hist(mat5); //express  onlyteeth Value
        divideHSV(mat5); // mH,mS,mV
        Mat HistH = calHist(mH);
        Mat HistS = calHist(mS);
        Mat HistV = calHist(mV);
        int H = maxF(HistH);
        int S = maxF(HistS);
        int V = maxF(HistV);

        cluster(H, S, V);


        if (bmpOut != null){
            mat.release();
            //String root = Environment.getExternalStorageDirectory().getAbsolutePath();
            File root = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
            //String dir = root +"/"+"Android"+"/"+"data" +"/"+ "Matching Teeth" + "/" + directoryName;

            //File file = new File(dir);
            String dir = root+"/"+"Matching Teeth";
            String fileName = filename + ".jpg";
            File file = new File(dir);
            file.mkdirs();
            File imagefile = new File(dir, fileName);

            if(bmpOut2 != null){
                Log.d(TAG, "saveImageToDisk: bmpOut2");
                String test = "BinaryImage"+".jpg";
                imwrite(dir+cpicturename+test,mat2);
            }
            if(bmpOut3 != null){
                Log.d(TAG, "saveImageToDisk: bmpOut3");
                String test = "Openning"+".jpg";
                imwrite(dir+cpicturename+test,mat3);
            }

            if(bmpOut4 != null){
                Log.d(TAG, "saveImageToDisk: bmpOut4");
                String test = "Connected"+".jpg";
                imwrite(dir+cpicturename+test,mat44);
            }
           if(bmpOut5 != null){
                Log.d(TAG, "saveImageToDisk: bmpOut5");
                String test = "onlyteeth"+".jpg";
                imwrite(dir+cpicturename+test,mat5);
               String test2 = "hsvCut"+".jpg";
               //imwrite(dir+cpicturename+test2,mat6);
            }


            // Test Save image with imwrite function of opencv that develope by c++
            // We must install jniLibs(sdk/native/libs) in opencv module
            //String test = "test"+".jpg";
            //imwrite(dir+test,source);



            try {
                OutputStream fileOut = null;
                fileOut = new FileOutputStream(imagefile);
                BufferedOutputStream bos = new BufferedOutputStream(fileOut);
                bmpOut.compress(Bitmap.CompressFormat.JPEG, 100, bos);
                bos.flush();
                bos.close();
                bmpOut.recycle();
            }
            catch (FileNotFoundException e) {
                e.printStackTrace();
            }

            catch (IOException e) {
                e.printStackTrace();
            }
        }
        bmpOut.recycle();

        //loadImageFromFile("picture",ColorBlobDetectionActivity.this);
    }

    public Mat BitmapToMat(Bitmap iBitmmap){
        Mat oMat = new Mat(iBitmmap.getWidth(),iBitmmap.getHeight(),CvType.CV_8UC4);
        // Default oMat is CV_8UC4 (RGBA)
        Utils.bitmapToMat(iBitmmap,oMat);
        return oMat;
    }

    public Mat loadImageFromFile() {
//      parameter : String directoryName, Context ctx
//        String root = Environment.getExternalStorageDirectory().getAbsolutePath();
//        String dir = root + "/"+"Android"+"/"+"data" +"/"+ "Matching Teeth" + "/" + directoryName;
//        String cImage = cpicturename+".jpg";
        Mat originalImage = new Mat();
        File root = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
        String dir = root+"/"+"Matching Teeth";
        // select  Image that just capture (cpicturename)
        String cImage = cpicturename + ".jpg";

        BitmapFactory.Options option = new BitmapFactory.Options();
        option.inPreferredConfig = Bitmap.Config.ARGB_8888;
        Bitmap myBitmap = BitmapFactory.decodeFile(dir+"/"+cImage, option);
        if (myBitmap != null) {
            System.out.print("read image is complete");
        } else {
            System.out.print("myBitmap is null, Do not read image");
        }
        //imageView = (ImageView)findViewById(R.id.cview);
        //imageView.setImageBitmap(myBitmap);
        originalImage = BitmapToMat(myBitmap);
        return originalImage;


    }

    public Mat createBinaryImage(Mat sourceImage){
        Mat grayImage = new Mat();
        Mat thresholdImage = new Mat();
        Imgproc.cvtColor(sourceImage,grayImage,Imgproc.COLOR_BGRA2GRAY);
        // select Threshold
        Imgproc.threshold(grayImage,thresholdImage,120,255,Imgproc.THRESH_BINARY);

        return thresholdImage;

    }

    public Mat openning(Mat binaryImage){
        //MORPH_OPEN, MORPH_ELLIPSE (ทรงวงรี)
        Mat oMat = new Mat();
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(30,30));
        Imgproc.morphologyEx(binaryImage,oMat,Imgproc.MORPH_OPEN,kernel);

        return oMat;

    }

    public  void closing(){
        //MORPH_CLOSE

    }
    public int[] numberOfComponent(Mat iMat){
        int[] check = {0,0,0,0,0,0,0,0,0,0,0,0};
        double[] value;
        // find number of each component
        for (int i=0; i<iMat.rows(); i++)
            for (int j=0; j<iMat.cols(); j++)
            {
                value = (iMat.get(i, j));
                if( value[0] == 0 ) {
                    check[0]++;
                    //iMat.put(i, j, 0);
                }
                else if(value[0] == 1) {
                    check[1]++;
                    //iMat.put(i, j, 50);
                }
                else if(value[0] == 2) {
                    check[2]++;
                    //iMat.put(i, j, 100);
                }
                else if(value[0] == 3) {
                    check[3]++;
                    //iMat.put(i, j, 120);
                }
                else if(value[0] == 4) {
                    check[4]++;
                    //iMat.put(i, j, 150);
                }
                else if(value[0] == 5){
                    check[5]++;
                    //iMat.put(i, j, 170);
                }
                else if(value[0] == 6){
                    check[6]++;
                    //iMat.put(i, j, 200);
                }
                else if(value[0] == 7){
                    check[7]++;
                    //iMat.put(i, j, 255);
                }
                else if (value[0] == 8){
                    check[8]++;
                }
                else if (value[0] == 9){
                    check[9]++;
                }
                else  if (value[0] == 10){
                    check[10]++;
                }
                else
                    check[11]++;
            }

        Log.d(TAG, "numberOfComponent:  " + " 0 = " + check[0] + " 1 =  " + check[1] +
                " 2 = " + check[2] + " 3 = " + check[3] + " 4 = " + check[4] + " 5 = " + check[5]  +
                " 6 = " + check[6] + " 7 = " + check[7]+"8 = "+check[8]+"9 = "+check[9]+"10 = "+check[10]+"Other = "+check[11]+"size = "+iMat.total());
        return check;
    }
    public Mat connectionComponent(Mat binaryImage) {
        Mat nlabel = new Mat();
        Imgproc.connectedComponents(binaryImage, nlabel, 8, CvType.CV_32S);
        nlabel.convertTo(nlabel, CvType.CV_8UC1);
        return nlabel;
    }

    public Mat colorComponent(Mat binaryImage) {
        Mat nlabel = new Mat();
        Imgproc.connectedComponents(binaryImage, nlabel, 8, CvType.CV_32S);
        nlabel.convertTo(nlabel, CvType.CV_8UC1);
        //return nlabel;


        int[] check = {0, 0, 0, 0, 0, 0, 0, 0};
        double[] value;
        // find number of each component
        for (int i = 0; i < nlabel.rows(); i++)
            for (int j = 0; j < nlabel.cols(); j++) {
                value = (nlabel.get(i, j));
                if (value[0] == 0) {
                    check[0]++;
                    nlabel.put(i, j, 0);
                } else if (value[0] == 1) {
                    check[1]++;
                    nlabel.put(i, j, 50);
                } else if (value[0] == 2) {
                    check[2]++;
                    nlabel.put(i, j, 100);
                } else if (value[0] == 3) {
                    check[3]++;
                    nlabel.put(i, j, 120);
                } else if (value[0] == 4) {
                    check[4]++;
                    nlabel.put(i, j, 150);
                } else if (value[0] == 5) {
                    check[5]++;
                    nlabel.put(i, j, 170);
                } else if (value[0] == 6) {
                    check[6]++;
                    nlabel.put(i, j, 200);
                } else {
                    check[7]++;
                    nlabel.put(i, j, 255);
                }
            }
        Log.d(TAG, "connectionComponent: channel : " + nlabel.channels() + " 0 = " + check[0] + " 1 =  " + check[1] +
                " 2 = " + check[2] + " 3 = " + check[3] + " 4 = " + check[4] + " 5 = " + check[5] + "\n size : " + nlabel.size() +
                " 6 = " + check[6] + " 7 = " + check[7] + " total = " + nlabel.total());
        return nlabel;
    }
    public int[][] centroid(Mat iMat){
        Log.d(TAG, "centroid: rows : "+iMat.rows()); // x  = row = 0
        Log.d(TAG, "centroid: cols : "+iMat.cols()); // y = col = 1 ???
        int Sumx0=0,Sumy0=0,Sumx1=0,Sumy1=0,Sumx2=0,Sumy2=0,Sumx3=0,Sumy3=0,Sumx4=0,Sumy4=0,Sumx5=0,Sumy5=0,Sumx6=0,Sumy6=0,Sumx7=0,Sumy7=0,Sumx8=0,Sumy8=0;
        double[] value;
        int n[]= {1,1,1,1,1,1,1,1,1};
        // x is rows, y is cols
        for (int i=0; i<iMat.rows(); i++){
            for (int j=0; j<iMat.cols(); j++){
                value = iMat.get(i, j);
                if (value[0] ==  0){
                    Sumx0 = Sumx0+i;
                    Sumy0 = Sumy0+j;
                    n[0]++;
                }
                else if (value[0] == 1){
                    Sumx1 = Sumx1+i;
                    Sumy1 = Sumy1+j;
                    n[1]++;
                }
                else if (value[0] == 2){
                    Sumx2 = Sumx2+i;
                    Sumy2 = Sumy2+j;
                    n[2]++;
                }
                else if (value[0] == 3){
                    Sumx3 = Sumx3+i;
                    Sumy3 = Sumy3+j;
                    n[3]++;
                }
                else if (value[0] == 4){
                    Sumx4 = Sumx4+i;
                    Sumy4 = Sumy4+j;
                    n[4]++;
                }
                else if (value[0] == 5){
                    Sumx5 = Sumx5+i;
                    Sumy5 = Sumy5+j;
                    n[5]++;
                }
                else if (value[0] == 6){
                    Sumx6 = Sumx6+i;
                    Sumy6 = Sumy6+j;
                    n[6]++;
                }
                else if (value[0] == 7){
                    Sumx7 = Sumx7+i;
                    Sumy7 = Sumy7+j;
                    n[7]++;
                }
                else if (value[0] == 8){
                    Sumx8 = Sumx8+i;
                    Sumy8 = Sumy8+j;
                    n[8]++;
                }

            }
        }

        for (int i=0; i<9; i++){
            if (n[i] > 1)
            {
                n[i] = n[i]-1;
            }
        }
        // mean
        int mx0 = Sumx0/n[0];
        int my0 = Sumy0/n[0];
        int mx1 = Sumx1/n[1];
        int my1 = Sumy1/n[1];
        int mx2 = Sumx2/n[2];
        int my2 = Sumy2/n[2];
        int mx3 = Sumx3/n[3];
        int my3 = Sumy3/n[3];
        int mx4 = Sumx4/n[4];
        int my4 = Sumy4/n[4];
        int mx5 = Sumx5/n[5];
        int my5 = Sumy5/n[5];
        int mx6 = Sumx6/n[6];
        int my6 = Sumy6/n[6];
        int mx7 = Sumx7/n[7];
        int my7 = Sumy7/n[7];
        int mx8 = Sumx8/n[8];
        int my8 = Sumy8/n[8];

        int[][] mean = {{mx0, mx1, mx2, mx3, mx4, mx5, mx6, mx7, mx8}, {my0, my1, my2, my3, my4, my5, my6, my7, my8}};

        Log.d(TAG, "centroid: x0 = "+mx0+"y0 = "+my0+"x1 = "+mx1+"y1 = "+my1+"mx2 = "+mx2+"my2 = "+my2+"mx3 = "+mx3+"my3 = "+my3);
        Log.d(TAG, "centroid: x4 = "+mx4+"y4 = "+my4+"x5 = "+mx5+"y5 = "+my5+"mx6 = "+mx6+"my6 = "+my6+"mx7 = "+mx7+"my7 = "+my7);
        Log.d(TAG, "centroid: x8 = "+mx8+"y8 = "+my8);

        return mean;
    }
    public int[] componentDonotCut(int[][] centroid, int dist, int[] sizeComponent){
        int[] component = {11,11,11,11,11,11,11,11,11};
        int k=0;
        double distance = 0;
        double mRow = 720/2;
        double mCol = 1280/2;

         for (int j=0; j<9; j++){
             // check centroid of component
            distance = Math.pow(mRow-centroid[0][j],2) - Math.pow(mCol-centroid[1][j], 2);
            distance = Math.abs(distance);
            distance = Math.sqrt(distance);
             Log.d(TAG, "Distance: distance : "+distance);
            if (distance >= 0 && distance < dist){ // 200
                Log.d(TAG, "Distance : "+ j);
                component[k] = j;
                k++;
            }

         }
        int count = 0; // check number of component that don't cut
        for (int i=0; i<component.length; i++) {
            if (component[i] != 11) {
                count++;
                Log.d(TAG, "componentDonotCut: "+i+"component.length : "+component.length);
            }
        }
        int outputCentroidOfComponentDonotCut[] = new int[count];

//        if (sizeComponent[component[0]] > sizeComponent[component[1]])
//            outputCentroidOfComponentDonotCut[0] = component[1];
//        else if (sizeComponent[component[0]] < sizeComponent[component[1]])
//            outputCentroidOfComponentDonotCut[0] = component[0];

        //int minSize = findMin(sizeComponent);
        //int j = 0;
        for (int i=0; i<count; i++) {

                outputCentroidOfComponentDonotCut[i] = component[i];
                Log.d(TAG, "componentDonotCut: "+component[i]+"count : "+count);
        }



        return outputCentroidOfComponentDonotCut;
    }
    public int findMin(int[] sizeComponent){
        int min = 921600;
        int component = 0;
        for (int i=0; i<sizeComponent.length; i++){
            if (sizeComponent[i] < min) {
                min = sizeComponent[i];
                component = i;
                Log.d(TAG, "findMin: min: "+min+"component : "+component);
            }
        }
        return component;
    }

    public Mat Onlyteeth(Mat iMat, int distance){
        Mat oMat = new Mat(iMat.size(), CvType.CV_8UC1);
        //Mat oMat = iMat.clone();
        double[] value;
        //double size = iMat.total();
       // int[] case1 = {99,99,99,99,99,99,99,99,99,99,99,99};
       // int[] case2 = {99,99,99,99,99,99,99,99,99,99,99,99};
       // int  y = 0; //index of array store index of data that cuted
        //int z = 0; //index of array store index of data that  don't cuted


        // loop find region that cuted and don't cuted
/*        for (int i=0; i<nConnected.length; i++) {
            if (nConnected[i] >= (0.1 * size)){
                case1[y] = i;
                y++;
            }
            else if(nConnected[i] >= 2000 && nConnected[i] <= 4000 ) {
                case2[z] = i;
                z++;
            }

        }

        Log.d(TAG, "Onlyteeth component size morn than 10 per of image : "+case1[0]+case1[1]+case1[2]+case1[3]+case1[4]+case1[5]);
        Log.d(TAG, "Onlyteeth: component size as same as a teeth : "+case2[0]+case2[1]+case2[2]+case2[3]);
        */
        // find centriod of component
        int[][] centroid = centroid(iMat);
        int[] componentDonotCut = componentDonotCut(centroid, distance,numberOfComponent(iMat));
        int statusROI = 0;
        int[] sizeC = numberOfComponent(iMat);

        for (int i=0; i<iMat.rows(); i++) {
            for (int j = 0; j < iMat.cols(); j++) {
                for (int k=0; k<componentDonotCut.length; k++){

                    value = (iMat.get(i, j));

                    if ((value[0] == componentDonotCut[k]) && (sizeC[componentDonotCut[k]] < 0.3*921600)){
                        oMat.put(i, j, 255); // component that centroid in teeth group
                        Log.d(TAG, "Onlyteeth1: componentDonotCut"+componentDonotCut[k]+"size: "+sizeC[componentDonotCut[k]]);
                        k = componentDonotCut.length + 9; // end loop
                        statusROI = 1;
                    }

                    /*
                    else if ((value[0] == case1[0]) || (value[0] == case1[1]) || (value[0] == case1[2]) || (value[0] == case1[3]) || (value[0] == case1[4]) || (value[0] == case1[5]) || (value[0] == case1[6])  || (value[0] == case1[7])|| (value[0] == case1[8])|| (value[0] == case1[9])|| (value[0] == case1[10])|| (value[0] == case1[11])) { // index same background
                        oMat.put(i, j, 0); // component size that more than 10% of image
                        k = componentDonotCut.length + 9; // end loop
                    }
                    else if (value[0] == case2[0] || value[0] == case2[1] || value[0] == case2[2] || value[0] == case2[3] || value[0] == case2[4] || value[0] == case2[5] || value[0] == case2[6] || value[0] == case2[7]) { // index like a teeth
                        oMat.put(i, j, 255); // component size same as a teeth
                        k = componentDonotCut.length + 9; // end loop
                    }
                    else {
                        oMat.put(i, j, 0);
                        k = componentDonotCut.length + 9; // end loop
                        //Log.d(TAG, "Onlyteeth2: componentDonotCut : "+componentDonotCut[k]+"value : "+value[0]);
                    }*/
                }
                if ( statusROI != 1){
                    oMat.put(i, j, 0);
                }

            }
        }
        return oMat;
    }

    public void divideHSV(Mat onlyteeth){
        // mH, mS, mV are attribute of this class
        Mat hsv = new Mat();
        Imgproc.cvtColor(mRgba,hsv,Imgproc.COLOR_RGB2HSV);
        List<Mat> chsv = new ArrayList<Mat>(3);
        Core.split(hsv,chsv);
        mH = chsv.get(0);
        mS = chsv.get(1);
        mV = chsv.get(2);

        for (int i=0; i<onlyteeth.rows(); i++){
            for (int j=0; j<onlyteeth.cols(); j++){
                double[] value = onlyteeth.get(i, j);
                if (value[0] == 255){
                    ;
                }
                else
                {
                    mH.put(i, j, 0);
                    mS.put(i, j, 0);
                    mV.put(i, j, 0);
                }

            }
        }

    }

    public Mat hist(Mat onlyteeth){
        Mat hsv = new Mat();
        Imgproc.cvtColor(mRgba,hsv,Imgproc.COLOR_RGB2HSV);
        List<Mat> chsv = new ArrayList<Mat>(3);
        Core.split(hsv,chsv);
        Mat H = chsv.get(0);
        Mat S = chsv.get(1);
        Mat V = chsv.get(2);

        ArrayList<Mat> list = new ArrayList<Mat>();
        //list.add(mV);
        MatOfInt channels = new MatOfInt(0); // H
        Mat hist= new Mat();
        MatOfInt histSize = new MatOfInt(256);
        MatOfFloat ranges = new MatOfFloat(0f, 256f);
        //Imgproc.calcHist(list, channels, new Mat(), hist, histSize, ranges);

        for (int i=0; i<onlyteeth.rows(); i++){
            for (int j=0; j<onlyteeth.cols(); j++){
                double[] value = onlyteeth.get(i, j);
                if (value[0] == 255){
                    ;
                }
                else
                {
                    H.put(i, j, 0);
                    S.put(i, j, 0);
                    V.put(i, j, 0);
                }

            }
        }
        list.add(V);
        Imgproc.calcHist(list,channels , onlyteeth, hist, histSize, ranges);
        double sum = 0,sum1=0;
        for (int i=0; i<256; i++){
                double[] vHist = hist.get(i, 0);
                Log.d(TAG, "hist: index = "+i+"Value = "+vHist[0]); // j = 0 always
                sum = sum + vHist[0];
        }
        Core.normalize(V, V, 0, 100, Core.NORM_MINMAX, -1); // range 0 - 100
        Imgproc.calcHist(list,channels , onlyteeth, hist, histSize, ranges);
        for (int i=0; i<256; i++){
            double[] vHist = hist.get(i, 0);
            Log.d(TAG, "hist: index = "+i+"Value = "+vHist[0]); // j = 0 always
            sum1 = sum1 + vHist[0];
        }
        Log.d(TAG, "hist sum :  "+sum+"sum1 : "+sum1);
        return V;
    }
    public Mat calHist(Mat grayImage){
        // grayImage is only teeth component in H S V
        MatOfInt channels = new MatOfInt(0);
        Mat hist= new Mat();
        MatOfInt histSize = new MatOfInt(256);
        MatOfFloat ranges = new MatOfFloat(0f, 256f);
        ArrayList<Mat> list = new ArrayList<Mat>();
        list.add(grayImage);
        Imgproc.calcHist(list,channels ,new Mat(), hist, histSize, ranges);
        for (int i=1; i<256; i++){
            double[] Hist = hist.get(i, 0);
            Log.d(TAG, "calHist: index : "+i+"value = "+Hist[0]);
        }

        return hist;
    }
    public int maxF(Mat hist){
        double hightFrequency = 0;
        int level = 0;
        Log.d(TAG, "maxF: "+level);
        for (int i=1; i<256; i++){
            double[] Hist = hist.get(i, 0);
            if ( Hist[0] > hightFrequency ) {
                hightFrequency = Hist[0];
                level = i;
            }
            Log.d(TAG, "maxF: level : "+i+" value : "+Hist[0]);
        }
        Log.d(TAG, "maxF: endLoop level : "+level+" frequency : "+hightFrequency);
        Toast.makeText(this,"max : "+level, Toast.LENGTH_LONG).show();
        return level;
    }
    public void cluster(int HistH,int HistS, int HistV){

        int[] databaseValue = {220, 200, 180, 160, 150};
        int[][] databaseHue = {{110,60},{30,},{},{},{},{}};
        int[][] databaseSaturation = {{},{},{},{},{},{}};

        int min = 255;
        int H=0,S=0,V=0;
        for (int i=0; i<databaseValue.length; i++){
            if ( Math.abs(databaseValue[i]-(HistV+28)) < min){
                min = Math.abs(databaseValue[i]-(HistV+38));
                V = i+1;
            }
        }
        Toast.makeText(this,"V : "+V, Toast.LENGTH_LONG).show();

        //Imgproc.cvtColor(rgba, mHSV, Imgproc.COLOR_RGBA2RGB,3);
        //Imgproc.cvtColor(rgba, mHSV, Imgproc.COLOR_RGB2HSV,3);
        //List<Mat> hsv_planes = new ArrayList<Mat>(3);
        //Core.split(mHSV, hsv_planes);

        /*
        Mat channel = hsv_planes.get(2);
        channel = Mat.zeros(mHSV.rows(),mHSV.cols(),CvType.CV_8UC1);
        hsv_planes.set(2,channel);
        Core.merge(hsv_planes,mHSV);



        Mat clusteredHSV = new Mat();
        mHSV.convertTo(mHSV, CvType.CV_32FC3);
        TermCriteria criteria = new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER,100,0.1);
        Core.kmeans(mHSV, 2, clusteredHSV, criteria, 10, Core.KMEANS_PP_CENTERS);
        */
    }


}
