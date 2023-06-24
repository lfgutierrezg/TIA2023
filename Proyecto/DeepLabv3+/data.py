import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
'''tqdm es un pequeño módulo que permite crear una barra de progreso 
basada en texto, que es desplegada en pantalla a partir de un bucle.
'''
from sklearn.model_selection import train_test_split
'''La función train_test_split permite hacer una división 
de un conjunto de datos en dos bloques de entrenamiento y 
prueba de un modelo (train and test).

test_sizefloat or int, default=None
If float, should be between 0.0 and 1.0 and represent 
the proportion of the dataset to include in the test split. 
If int, represents the absolute number of test samples. 
If None, the value is set to the complement of the train size. 
If train_size is also None, it will be set to 0.25.
'''

from albumentations import HorizontalFlip, GridDistortion, OpticalDistortion, ChannelShuffle, CoarseDropout, CenterCrop, Crop, Rotate
'''

GridDistortion: Esta función realiza una distorsión en una imagen al mover y
 deformar una cuadrícula regular. Cada punto de la imagen se desplaza según
   una interpolación de la posición de los puntos de la cuadrícula vecinos. 
   Esto crea un efecto de distorsión en la imagen.

OpticalDistortion: Esta función aplica una distorsión a una imagen simulando
los efectos de una lente óptica. La distorsión se basa en el modelo de distorsión
radial de la lente, que puede incluir una distorsión de barril o una distorsión de cojín.
La cantidad de distorsión se puede controlar mediante los parámetros específicos de la función.

ChannelShuffle: Esta función permuta aleatoriamente los canales de color de una 
imagen. Por ejemplo, en una imagen RGB, los canales R, G y B se pueden reorganizar 
de forma aleatoria. Esto puede ser útil para agregar variabilidad a los datos y 
mejorar la robustez del modelo.

CoarseDropout: Esta función realiza una eliminación aleatoria de bloques rectangulares 
en una imagen. Los bloques se eliminan seleccionando una posición y un tamaño aleatorio, 
y se rellenan con un valor específico, como 0 o el valor medio de los píxeles de la imagen. 
Esta técnica se utiliza para simular datos faltantes o para agregar regularización a modelos 
de aprendizaje automático.

CenterCrop: Esta función recorta una imagen alrededor de su centro, manteniendo las 
dimensiones especificadas. Por ejemplo, si tienes una imagen de 1000x1000 píxeles y 
deseas recortarla a 500x500 píxeles, CenterCrop recortará 250 píxeles en cada borde de la imagen.

Crop: Esta función realiza un recorte arbitrario de una imagen, especificando las 
coordenadas del rectángulo de recorte. Puedes definir la posición y el tamaño del 
recorte, lo que te permite extraer una región de interés específica de una imagen.

Rotate: Esta función realiza una rotación de una imagen en un ángulo determinado. 
La rotación puede ser en sentido horario o antihorario, y se especifica en grados. 
La imagen se ajusta para que no se pierdan píxeles durante la rotación.

'''


""" Crear directorio si no existe """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path, split=0.3):
    """ Cargar imagenes y mascaras """
    X = sorted(glob(os.path.join(path, "images", "*.jpg")))
    Y = sorted(glob(os.path.join(path, "masks", "*.png")))

    """ Dividir la data en test y train y se usa 70% para train """
    split_size = int(len(X) * split)

    train_x, test_x = train_test_split(X, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(Y, test_size=split_size, random_state=42)
    split_size = int(len(test_x) * split)
    test_x, valid_x = train_test_split(test_x, test_size=split_size, random_state=42)
    test_y, valid_y = train_test_split(test_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (test_x, test_y),(valid_x, valid_y)

def augment_data(images, masks, save_path, augment=True):
    H = 512
    W = 512

    for x, y in tqdm(zip(images, masks), total=len(images)):
        """ Extract the name """
        name = x.split("\\")[-1].split(".")[0]

        """ Reading the image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)

        """ Augmentation """
        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            x2 = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
            y2 = y

            aug = ChannelShuffle(p=1)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = CoarseDropout(p=1, min_holes=3, max_holes=10, max_height=32, max_width=32)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x5 = augmented["image"]
            y5 = augmented["mask"]

            X = [x, x1, x2, x3, x4, x5]
            Y = [y, y1, y2, y3, y4, y5]

        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            try:
                """ Center Cropping """
                aug = CenterCrop(H, W, p=1.0)
                augmented = aug(image=i, mask=m)
                i = augmented["image"]
                m = augmented["mask"]

            except Exception as e:
                i = cv2.resize(i, (W, H))
                m = cv2.resize(m, (W, H))

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"
            
            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1




if __name__ == "__main__":
    """ Inicializando el punto semilla de la generacion 
    de numeros aleatorios en python"""
    np.random.seed(42)

    """ Cargar el dataset """
    data_path = "people_segmentation"
    (train_x, train_y), (test_x, test_y),(valid_x,valid_y) = load_data(data_path)

    print(f"Train:\t {len(train_x)} - {len(train_y)}")
    print(f"Test:\t {len(test_x)} - {len(test_y)}")
    print(f"valid:\t {len(valid_x)} - {len(valid_y)}")

    """ crear directorios para guardar la data """
    create_dir("new_data/train/image/")
    create_dir("new_data/train/mask/")
    create_dir("new_data/valid/image/")
    create_dir("new_data/valid/mask/")
    create_dir("new_data/test/image/")
    create_dir("new_data/test/mask/")

    """ Data augmentation """
    augment_data(train_x, train_y, "new_data/train/", augment=True)
    augment_data(test_x, test_y, "new_data/test/", augment=False)
    augment_data(valid_x, valid_y, "new_data/valid/", augment=False)

