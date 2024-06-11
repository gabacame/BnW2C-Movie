import numpy as np
import cv2
import os

def colorize_image(image_path, net, pts_in_hull):
    # Load the serialized black and white image and scale it to the range [0, 1]
    image = cv2.imread(image_path)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # Resize the Lab image to 224x224 (the dimensions the Caffe model was trained on)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Predict the 'a' and 'b' channels from the input L channel (i.e., the grayscale image)
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize the predicted 'ab' volume to the same dimensions as our input image
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    # Grab the L channel from the original input image, combine with the predicted 'ab' channels, and convert back to RGB
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    # Scale the pixel intensities to the range [0, 255]
    colorized = (255 * colorized).astype("uint8")
    
    return colorized

# Usar rutas absolutas para los archivos de modelo con cadenas sin procesar
prototxt_path = os.path.abspath(r'D:\gits\BnW2C-Movie\colorization_deploy_v2.prototxt')
caffemodel_path = os.path.abspath(r'D:\gits\BnW2C-Movie\colorization_release_v2_norebal.caffemodel')
pts_in_hull_path = os.path.abspath(r'D:\gits\BnW2C-Movie\pts_in_hull.npy')

# Verificar la existencia de los archivos
if not os.path.isfile(prototxt_path):
    print(f"Prototxt file not found at: {prototxt_path}")
    raise FileNotFoundError(f"File not found: {prototxt_path}")
else:
    print(f"Prototxt file found at: {prototxt_path}")

if not os.path.isfile(caffemodel_path):
    print(f"Caffemodel file not found at: {caffemodel_path}")
    raise FileNotFoundError(f"File not found: {caffemodel_path}")
else:
    print(f"Caffemodel file found at: {caffemodel_path}")

if not os.path.isfile(pts_in_hull_path):
    print(f"pts_in_hull.npy file not found at: {pts_in_hull_path}")
    raise FileNotFoundError(f"File not found: {pts_in_hull_path}")
else:
    print(f"pts_in_hull.npy file found at: {pts_in_hull_path}")

# Cargar el modelo
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Load cluster centers
pts_in_hull = np.load(pts_in_hull_path)

# Add the cluster centers as 1x1 convolutions to the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts_in_hull.transpose().reshape(2, 313, 1, 1)

net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

frames_folder = 'frames'
output_folder = 'colorized_frames'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for frame_file in os.listdir(frames_folder):
    frame_path = os.path.join(frames_folder, frame_file)
    colorized_frame = colorize_image(frame_path, net, pts_in_hull)
    cv2.imwrite(os.path.join(output_folder, frame_file), colorized_frame)

def create_video_from_frames(frames_folder, output_video_path, fps=24):
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".png")])
    first_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    height, width, _ = first_frame.shape

    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)
    
    video_writer.release()

create_video_from_frames('colorized_frames', 'colorized_movie.mp4')