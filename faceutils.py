# Author: Jonathan da Silva Santos, 11/2018, silva.santos.jonathan@gmail.com
# This project is license with GPL 2.0


import cv2
import dlib
import numpy as np
import click
from os import path, listdir


def show(message, verbose):
    if verbose:
        click.echo(message)

def align_face(img, gray, region, predictor):

    (x, y, w, h) = region


    rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))                                     
    shape = predictor(gray, rect)

    shape = np.array([(shape.part(j).x, shape.part(j).y) for j in range(shape.num_parts)])

    # center and scale face around mid point between eyes
    center_eyes = shape[27].astype(np.int)
    eyes_d = np.linalg.norm(shape[36]-shape[45])
    face_size_x = int(eyes_d * 2.)
    if face_size_x < 50: return img

    output_size = 256


    # Rotation
    d = (shape[45] - shape[36]) / eyes_d # normalized eyes-differnce vector (direction)
    a = np.rad2deg(np.arctan2(d[1],d[0])) # angle
    scale_factor = float(output_size) / float(face_size_x * 2.) # scale to fit in output_size

    M = np.append(cv2.getRotationMatrix2D((center_eyes[0], center_eyes[1]),a,scale_factor),[[0,0,1]], axis=0)
    M1 = np.array([[1.,0.,-center_eyes[0]+output_size/2.],
                        [0.,1.,-center_eyes[1]+output_size/2.],
                        [0,0,1.]])
    M = M1.dot(M)[:2]
    # warp
    try:
        object = cv2.warpAffine(img, M, (output_size, output_size), borderMode=cv2.BORDER_REPLICATE)
    except:
        return img

    return object


def detect(img, grayscale, width, height, classifier, predictor, equalizehist):

    # Before try to detect objects, we need to convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    regions = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    objects = []
    for region in regions:
        (x, y, w, h) = region
        if grayscale:
            object = gray[y:y+w, x:x+h]
            if equalizehist:
                object = cv2.equalizeHist(object)
        else:
            object = img[y:y+w, x:x+h]

        if predictor is not None:
            object = align_face(img, gray, region, predictor)

        if (width != -1) and (height != -1):
            object = cv2.resize(object, (width,height), interpolation = cv2.INTER_CUBIC)



        objects.append(object)

    return objects


def process(inputpath, outputpath, grayscale, width, height, classifier, predictor, equalizehist):
    # Loads image
    img = cv2.imread(inputpath)

    if img is None: return

    # fetch the roi's
    objects = detect(img, grayscale, width, height, classifier, predictor, equalizehist)


    # Save the new image
    filename_index = 0
    for object in objects:
        write_to_path = ''
        if path.isfile(outputpath):
            write_to_path = outputpath
        else:
            path_parts = inputpath.split("/")
            generated_filename = path_parts[len(path_parts)-1]
            write_to_path = click.format_filename(outputpath) + generated_filename
            if len(objects) > 1:
                write_to_path = click.format_filename(outputpath) + str(filename_index) + generated_filename
                filename_index += 1
        cv2.imwrite(write_to_path, object)
            




@click.command()
@click.option('--inputpath', '-i', type=click.Path(exists=True), required=True, help='Path of input file or directory')
@click.option('--outputpath', '-o', type=click.Path(exists=False), required=True, help='Path of output file or directory')
@click.option('--shapefile', '-s', type=click.Path(exists=False), required=False, help='Determines a shape predictor for image rotation and alignment')
@click.option('--verbose', is_flag=True, help="Will print verbose messages.")
@click.option('--grayscale', is_flag=True, help="Save images as grayscale")
@click.option('--width','-w',  default=-1, help='Output image width size (in pixels)')
@click.option('--height','-h',  default=-1, help='Output image height size (in pixels)')
@click.option('--haarcascade', '-c', type=click.Path(exists=True), required=True, help='Specify the haar-cascade file that will be used by the object classifier')
@click.option('--equalizehist', '-eqh', is_flag=True, help="Equalize Histogram of Images")

def extract(inputpath, outputpath, verbose, grayscale, width, height, haarcascade, shapefile, equalizehist):
    

    classifier = cv2.CascadeClassifier(haarcascade)
    predictor = None


    if shapefile is not None:
        predictor = dlib.shape_predictor(shapefile)


    show("FaceUtils a simple tool to prepare facial images to be used in a machine learning algorithm  v0.0.1", verbose)
    show("Author: Jonathan S. Santos - 11/2018 - silva.santos.jonathan@gmail.com", verbose)

    
    # Check if the input path is a file or a directory
    isfile = path.isfile(inputpath)

    if isfile:
        show("Processing: " + str(inputpath), verbose)
        process(inputpath, outputpath, grayscale, width, height, classifier, predictor, equalizehist)
        
    else:
        # Input is a directory, so we presume that there are multiple files to be proccessed.
        for content in listdir(inputpath):
            input_file_path = str(inputpath) + content
            show("Processing: " + str(input_file_path), verbose)
            process(input_file_path, outputpath, grayscale, width, height, classifier, predictor, equalizehist)


    



if __name__ == '__main__':
    extract()

