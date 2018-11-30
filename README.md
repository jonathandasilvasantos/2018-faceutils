FaceUtils: a simple tool to prepare images that contain faces to be used in machine learning algorithms.
===

> This is a simple tool to prepare images for a machine learning algorithm.


#### Setup:
1. Run `pip install -r requirements.txt`
2. Download an haar cascade (aka: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalcatface.xml )
3. Download the shpae predictor compatible with dlib. (This project supports: shape_predictor_68_face_landmarks.dat)
4. Place the haar cascade file in the same folder of the project.

#### Run:
```bash
python faceutils.py -i \<image input\> -o \<image output\> -c <haar cascade file> -s <dlib shape predictor file>
```

### Usage: extract.py [OPTIONS]

Options:
  -i, --inputpath PATH    Path of input file or directory  [required]
  -o, --outputpath PATH   Path of output file or directory  [required]
  -s, --shapefile PATH    Determines a shape predictor for image rotation and
                          alignment
  --verbose               Will print verbose messages.
  --grayscale             Save images as grayscale
  -w, --width INTEGER     Output image width size (in pixels)
  -h, --height INTEGER    Output image height size (in pixels)
  -c, --haarcascade PATH  Specify the haar-cascade file that will be used by
                          the object classifier  [required]
  -eqh, --equalizehist    Equalize Histogram of Images
  --help                  Show this message and exit.

                          the object classifier  [required]
#### Examples

Align all images with faces in a folder and save in another one:
```bash
python faceutils.py -i \<image input\> -o \<image output\> -c <haar cascade file> -s <dlib shape predictor file>
```

Crop, resize, convert to grayscale and equalize histogram of images
```bash
python faceutils.py -i \<image input\> -o \<image output\> -c <haar cascade file> --grayscale -eqh -w 92 -h92
```


##### Dependencies:
- [dlib](http://dlib.net/)
- [numpy](http://www.numpy.org/)
- [opencv-python](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html)
- [click](https://click.palletsprojects.com/en/7.x/)
