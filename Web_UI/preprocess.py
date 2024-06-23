import numpy as np
import pandas as pd
from glob import glob
import os

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


import scipy
from scipy import ndimage as ndi
from skimage.filters import roberts
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image, disk, binary_closing
from skimage.segmentation import clear_border
import time


# CT Scan Utility
import scipy.misc
import SimpleITK as sitk
from ast import literal_eval




class CTScan(object):
    def __init__(self, seriesuid, centers=None, radii=None, clazz=None, annotationExist= True,
                resourcePath=None, outputPath = None):
        
        self.annotationExist = annotationExist
        self.resourcePath = resourcePath
        self.outputPath = outputPath
        self._seriesuid = seriesuid
        self._centers = centers
        paths = glob(f'''{self.resourcePath}\\{self._seriesuid}.mhd''')
        path = paths[0]
        self._ds = sitk.ReadImage(path)
        self._spacing = np.array(list(reversed(self._ds.GetSpacing())))
        self._origin = np.array(list(reversed(self._ds.GetOrigin())))
        self._image = sitk.GetArrayFromImage(self._ds)
        self._radii = radii
        self._clazz = clazz
        self._mask = None
        

    def preprocess(self, info=True):
        if(info):
          print("Resampling")
        self._resample()
        if(info):
          print("Segmenting Lung")
        self._segment_lung_from_ct_scan()
        if(info):
          print("Normalizing")
        self._normalize()
        if(info):
          print("Zero Centering")
        self._zero_center()

        if(self.annotationExist):
          if(info):
            print("Changing Coords")
          self._change_coords()

    def save_preprocessed_image(self, plot=False):
        if(self._clazz == 2):
          subdir = 'unlabeled'
          file_path = f'''preprocessed\\{subdir}\\{self._seriesuid}.npy'''
          print("Saving to file_path")
          np.save(f'{self.outputPath}\\{file_path}', self._image)  
        else:
          subdir = 'negatives' if self._clazz == 0 else 'positives'
          file_path = f'''preprocessed\\{subdir}\\{self._seriesuid}.npy'''
          np.save(f'{self.outputPath}\\{file_path}', self._image)

    def get_info_dict(self):
        (min_z, min_y, min_x, max_z, max_y, max_x) = (None, None, None, None, None, None)
        for region in regionprops(self._mask):
            min_z, min_y, min_x, max_z, max_y, max_x = region.bbox
        assert (min_z, min_y, min_x, max_z, max_y, max_x) != (None, None, None, None, None, None)
        min_point = (min_z, min_y, min_x)
        max_point = (max_z, max_y, max_x)
        return {'seriesuid': self._seriesuid, 'radii': self._radii, 'centers': self._centers,
                'spacing': list(self._spacing), 'lungs_bounding_box': [min_point, max_point], 'class': self._clazz}

    def _resample(self):
        spacing = np.array(self._spacing, dtype=np.float32)
        new_spacing = [1, 1, 1]
        imgs = self._image
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = scipy.ndimage.zoom(imgs, resize_factor, mode='nearest')
        self._image = imgs
        self._spacing = true_spacing

    def _segment_lung_from_ct_scan(self):
        result_img = []
        result_mask = []
        num = 0
        for slicee in self._image:
            rimg, rmsk = self.get_segmented_lungs(slicee)
            result_img.append(rimg)
            result_mask.append(rmsk)
        self._image = np.asarray(result_img)
        self._mask = np.asarray(result_mask, dtype=int)

    def _world_to_voxel(self, worldCoord):
        stretchedVoxelCoord = np.absolute(np.array(worldCoord) - np.array(self._origin))
        voxelCoord = stretchedVoxelCoord / np.array(self._spacing)
        return voxelCoord.astype(int)

    def _get_world_to_voxel_coords(self, idx):
        return tuple(self._world_to_voxel(self._centers[idx]))

    def _get_voxel_coords(self):
        voxel_coords = [self._get_world_to_voxel_coords(j) for j in range(len(self._centers))]
        return voxel_coords

    def _change_coords(self):
        new_coords = self._get_voxel_coords()
        self._centers = new_coords

    def _normalize(self):
        MIN_BOUND = -1200
        MAX_BOUND = 600.
        self._image = (self._image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        self._image[self._image > 1] = 1.
        self._image[self._image < 0] = 0.
        self._image *= 255.

    def _zero_center(self):
        PIXEL_MEAN = 0.25 * 256
        self._image = self._image - PIXEL_MEAN

    def get_segmented_lungs(self, im, plot=False):
        '''
        This funtion segments the lungs from the given 2D slice.
        '''
        plt_number = 0
        # Original image label: 0
        if plot:
            f, plots = plt.subplots(12, 1, figsize=(5, 40))
            plots[plt_number].axis('off')
            plots[plt_number].set_title(f'{plt_number}')
            plots[plt_number].imshow(im, cmap=plt.cm.bone)
            plt_number += 1

        # Step 1: Convert into a binary image.
        # image label: 1
        binary = im < -604
        cleared = clear_border(binary)
        label_image = label(cleared)
        areas = [r.area for r in regionprops(label_image)]
        areas.sort()
        labels = []
        if len(areas) > 2:
            for region in regionprops(label_image):
                if region.area < areas[-2]:
                    for coordinates in region.coords:
                        label_image[coordinates[0], coordinates[1]] = 0
                else:
                    coordinates = region.coords[0]
                    labels.append(label_image[coordinates[0], coordinates[1]])
        else:
            labels = [1, 2]
        rig = label_image == labels[0]
        lef = label_image == labels[1]
        r_edges = roberts(rig)
        l_edges = roberts(lef)
        rig = ndi.binary_fill_holes(r_edges)
        lef = ndi.binary_fill_holes(l_edges)

        rig = convex_hull_image(rig)
        lef = convex_hull_image(lef)

        sum_of_lr = rig + lef
        binary = sum_of_lr > 0

        selem = disk(10)
        binary = binary_closing(binary, selem)

        # Step 9: Superimpose the binary mask on the input image.
        # image label: 11
        get_high_vals = binary == 0
        im[get_high_vals] = 0
        if plot:
            plots[plt_number].axis('off')
            plots[plt_number].set_title(f'{plt_number}')
            plots[plt_number].imshow(im, cmap=plt.cm.bone)
            plt_number += 1
            

        return im, convex_hull_image(binary)

  
class Preprocess:

    def __init__(self, resourcePath , outputPath, annotationExist=False ):
        self.resourcePath = resourcePath
        self.outputPath = outputPath
        self.blockSize = 128
        self.annotationExist = annotationExist
        self.padding = 10

        if(self.annotationExist):
            self.annotations = pd.read_csv(self.resourcePath + '\\annotations.csv')
            self.candidates = pd.read_csv(self.resourcePath + '\\candidates.csv')

    def get_positive_series(self):
        paths = glob(self.resourcePath + '\\' + "*.mhd")
        file_list = [f.split('\\')[-1][:-4] for f in paths]
        series = self.annotations['seriesuid'].tolist()
        infected = [f for f in file_list if f in series]
        return infected

    def get_negative_series(self):
        paths = glob(self.resourcePath + '\\' + "*.mhd")
        file_list = [f.split('\\')[-1][:-4] for f in paths]
        series = self.annotations['seriesuid'].tolist()
        cleans = [f for f in file_list if f not in series]
        return cleans
    
    def get_unlabeled_series(self):
        paths = glob(self.resourcePath + '\\' + "*.mhd")
        file_list = [f.split('\\')[-1][:-4] for f in paths]
        # print("File list: ",file_list)
        return file_list    

    def save_preprocessed_data(self):
        print("Preprocessing Start")
        if(self.annotationExist):
            print("Preprocessing With Annotation")
            [os.makedirs(d, exist_ok=True) for d in
            [f'{self.outputPath}\\preprocessed\\positives', f'{self.outputPath}\\preprocessed\\negatives']]
            meta_data = pd.DataFrame(columns=['seriesuid', 'spacing', 'lungs_bounding_box', 'centers', 'radii', 'class'])


            total_length_pos = len(self.get_positive_series())
            total_length_neg = len(self.get_negative_series())

            processed_num = 0
            neg_processed_num = 0

            for series_id in self.get_positive_series():
                start_time = time.time()
                print("Processing id pos: ",series_id, " num-left: ", total_length_pos - processed_num)
                processed_num += 1

                nodule_coords_annot = self.annotations[self.annotations['seriesuid'] == series_id]
                tp_co = [(a['coordZ'], a['coordY'], a['coordX']) for a in nodule_coords_annot.iloc]
                radii = [(a['diameter_mm'] / 2) for a in nodule_coords_annot.iloc]
                ct = CTScan(seriesuid=series_id, centers=tp_co, radii=radii, clazz=1, annotationExist=self.annotationExist,
                            resourcePath=self.resourcePath, outputPath=self.outputPath )
                ct.preprocess()
                ct.save_preprocessed_image()
                diction = ct.get_info_dict()
                print("diction: ",diction)
                meta_data.loc[len(meta_data)] = pd.Series(diction)            

                end_time = time.time()
                time_spent = end_time - start_time
                print("Time spent:", time_spent, "seconds")            
                
            for series_id in self.get_negative_series():
                print("Processing id neg: ",series_id, " num-left: ", total_length_neg - neg_processed_num)
                neg_processed_num += 1

                nodule_coords_candid = self.candidates[self.candidates['seriesuid'] == series_id]
                tp_co = [(a['coordZ'], a['coordY'], a['coordX']) for a in nodule_coords_candid.iloc]
                radii = list(np.random.randint(40, size=len(tp_co)))
                max_numbers_to_use = min(len(tp_co), 3)
                tp_co = tp_co[:max_numbers_to_use]
                radii = radii[:max_numbers_to_use]

                ct = CTScan( seriesuid=series_id, centers=tp_co, radii=radii, clazz=0, annotationExist=self.annotationExist,
                            resourcePath=self.resourcePath, outputPath=self.outputPath)
                ct.preprocess()
                ct.save_preprocessed_image()
                diction = ct.get_info_dict()
                meta_data.loc[len(meta_data)] = pd.Series(diction)
            
            meta_data.to_csv(f'{self.outputPath}\\preprocessed_meta.csv')

        else:
            print("Preprocessing Without Annotation")

            [os.makedirs(d, exist_ok=True) for d in
            [f'{self.outputPath}\\preprocessed\\unlabeled']]

            meta_data = pd.DataFrame(columns=['seriesuid', 'spacing', 'lungs_bounding_box', 'centers', 'radii', 'class'])


            for series_id in self.get_unlabeled_series():
                print("Processing id unlabeled: ",series_id)
                ct = CTScan(resourcePath=self.resourcePath, outputPath=self.outputPath, annotationExist=self.annotationExist,
                            seriesuid=series_id,centers=[(0,0,0)], radii=[0], clazz=2)
                ct.preprocess()
                ct.save_preprocessed_image()

                diction = ct.get_info_dict()
                meta_data.loc[len(meta_data)] = pd.Series(diction)
                break
            
            meta_data.to_csv(f'{self.outputPath}\\preprocessed_meta_unlabeled.csv')
        
        print("Preprocessing Finished")


class Augmentation:
    def __init__(self, resourcePath , outputPath, annotationExist=False,blockSize = 128 ):
        self.resourcePath = resourcePath
        self.outputPath = outputPath
        self.blockSize = blockSize
        self.annotationExist = annotationExist
        self.padding = 10

        if(self.annotationExist):
            print("With annotation")
            self.p_meta = pd.read_csv(f'{self.outputPath}\\preprocessed_meta.csv', index_col=0)
        else:
            print("No annotation")
            self.p_meta = pd.read_csv(f'{self.outputPath}\\preprocessed_meta_unlabeled.csv', index_col=0)

        if(self.annotationExist):
            self.annotations = pd.read_csv(self.resourcePath + '\\annotations.csv')
            self.candidates = pd.read_csv(self.resourcePath + '\\candidates.csv')


    def argmax_3d(self, img: np.array):
        max1 = np.max(img, axis=0)
        argmax1 = np.argmax(img, axis=0)
        max2 = np.max(max1, axis=0)
        argmax2 = np.argmax(max1, axis=0)
        argmax3 = np.argmax(max2, axis=0)
        argmax_3d = (argmax1[argmax2[argmax3], argmax3], argmax2[argmax3], argmax3)
        return argmax_3d, img[argmax_3d]

    def _get_patches(self,record):

            rec = record
            seriesuid = rec['seriesuid']
            spacing = literal_eval(rec['spacing'])
            lungs_bounding_box = literal_eval(rec['lungs_bounding_box'])
            centers = literal_eval(rec['centers'])
            radii = literal_eval(rec['radii'])
            clazz = int(rec['class'])

            if(self.annotationExist):
                file_directory = 'preprocessed\\positives' if clazz == 1 else 'preprocessed\\negatives'
            else:
                file_directory = 'preprocessed\\unlabeled'

            file_path = f'{self.outputPath}\\{file_directory}\\{seriesuid}.npy'

            pm = PatchMaker(seriesuid=seriesuid, coords=centers, radii=radii, spacing=spacing,
                            lungs_bounding_box=lungs_bounding_box,
                            file_path=file_path, clazz=clazz, blockSize=self.blockSize, annotationExist=self.annotationExist, outputPath=self.outputPath )

            return pm.get_augmented_patches_normal()

    def save_augmented_data(self, preprocess_meta=None):
        print("Augmentation Start")
        if(preprocess_meta == None):
            preprocess_meta = self.p_meta

        [os.makedirs(d, exist_ok=True) for d in
        [f'{self.outputPath}\\augmented\\positives', f'{self.outputPath}\\augmented\\negatives',f'{self.outputPath}\\augmented\\unlabeled']]
        augmentation_meta = pd.DataFrame(columns=['seriesuid',  'sub_index', 'centers', 'lungs_bounding_box', 'radii',
                                                'class'])
        
        print("Creating files: ",)
        
        list_of_positives = []
        list_of_negatives = []
        for rec in preprocess_meta.loc[preprocess_meta['class'] == 1].iloc:
            list_of_positives += self._get_patches(rec)
        for rec in preprocess_meta.loc[preprocess_meta['class'] == 0].iloc:
            list_of_negatives += self._get_patches(rec)
            # 33 percent of the data will be negative samples
            if len(list_of_negatives) > len(list_of_positives) / 2:
                break
        for rec in preprocess_meta.loc[preprocess_meta['class'] == 2].iloc:
            list_of_positives += self._get_patches(rec)

        newRows = list_of_positives + list_of_negatives
        for row in newRows:
            augmentation_meta.loc[len(augmentation_meta)] = row

        if(self.annotationExist):
            augmentation_meta.to_csv(f'{self.outputPath}\\augmented_meta.csv')
        else:
            augmentation_meta.to_csv(f'{self.outputPath}\\augmented_meta_unlabeled.csv')
        
        print("Augmentation Finished")

class PatchMaker(object):
    def __init__(self, seriesuid: str, coords: list, radii: list, spacing: list, lungs_bounding_box: list,
                file_path: str, clazz: int, outputPath, blockSize=128, annotationExist= True, ):
        self._seriesuid = seriesuid
        self._coords = coords
        self._spacing = spacing
        self._radii = radii
        self._image = np.load(file=f'{file_path}')
        self._clazz = clazz
        self._lungs_bounding_box = lungs_bounding_box
        self.blockSize = blockSize
        self.annotationExist = annotationExist
        self.outputPath = outputPath

    def get_cube_from_img_new(sefl, img, origin: tuple, block_size=128, pad_value=106.):
        assert 2 <= len(origin) <= 3
        final_image_shape = tuple([block_size] * len(origin))
        result = np.ones(final_image_shape) * pad_value
        start_at_original_images = []
        end_at_original_images = []
        start_at_result_images = []
        end_at_result_images = []
        for i, center_of_a_dim in enumerate(origin):
            start_at_original_image = int(center_of_a_dim - block_size / 2)
            end_at_original_image = start_at_original_image + block_size
            if start_at_original_image < 0:
                start_at_result_image = abs(start_at_original_image)
                start_at_original_image = 0
            else:
                start_at_result_image = 0
            if end_at_original_image > img.shape[i]:
                end_at_original_image = img.shape[i]
                end_at_result_image = start_at_result_image + (end_at_original_image - start_at_original_image)
            else:
                end_at_result_image = block_size
            start_at_original_images.append(start_at_original_image)
            end_at_original_images.append(end_at_original_image)
            start_at_result_images.append(start_at_result_image)
            end_at_result_images.append(end_at_result_image)
        # for simplicity
        sri = start_at_result_images
        eri = end_at_result_images
        soi = start_at_original_images
        eoi = end_at_original_images

        print("sri {}, eri {}, soi {}, eoi {}. img shape: {}".format(sri,eri,soi,eoi, img.shape))
        if len(origin) == 3:
            result[sri[0]:eri[0], sri[1]:eri[1], sri[2]:eri[2]] = img[soi[0]:eoi[0], soi[1]:eoi[1], soi[2]:eoi[2]]
        elif len(origin) == 2:
            result[sri[0]:eri[0], sri[1]:eri[1]] = img[soi[0]:eoi[0], soi[1]:eoi[1]]

        return result

    def normal_crop(self,img: np.array, centers: list, lungs_bounding_box: list, radii: list, center_of_cube: list,
                    spacing: tuple,
                    block_size: int,
                    pad_value: float, margin: int):

        out_img = self.get_cube_from_img_new(img, origin=tuple(center_of_cube), block_size=block_size, pad_value=pad_value)
        out_centers = []
        out_lungs_bounding_box = []
        print("centers: ",centers)
        for i in range(len(centers)):
            diff = np.array(center_of_cube) - np.array(centers[i])
            out_centers.append(
                tuple(np.array([int(block_size / 2)] * len(centers[i]), dtype=int) - diff))
        for i in range(len(lungs_bounding_box)):
            diff = np.array(center_of_cube) - np.array(lungs_bounding_box[i])
            out_lungs_bounding_box.append(tuple(
                np.array([int(block_size / 2)] * len(lungs_bounding_box[i]), dtype=int) - diff))

        return out_img, out_centers, out_lungs_bounding_box
    
    def get_augmented_cube_normal(self,img: np.array, radii: list, centers: list, center_of_cube: list, spacing: tuple,
                        lungs_bounding_box: list, block_size=128, pad_value=106, margin=10, rot_id=None):

        
        img2, centers2, lungs_bounding_box2 = self.normal_crop(img=img, centers=centers,
                                                        lungs_bounding_box=lungs_bounding_box, radii=radii,
                                                        center_of_cube=center_of_cube, spacing=spacing,
                                                        block_size=block_size, pad_value=pad_value, margin=margin)
        existing_centers_in_patch = []
        for i in range(len(centers2)):
            dont_count = False
            for ax in centers2[i]:
                if not (0 <= ax <= block_size):
                    dont_count = True
                    break
            if not dont_count:
                existing_centers_in_patch.append(i)

        return img2, radii, centers2, lungs_bounding_box2, spacing, existing_centers_in_patch

    def _get_augmented_patch_normal(self, center_of_cube, rot_id=None):
        return self.get_augmented_cube_normal(img=self._image, radii=self._radii, centers=self._coords,
                                  spacing=tuple(self._spacing), rot_id=rot_id, center_of_cube=center_of_cube,
                                  lungs_bounding_box=self._lungs_bounding_box)
  
    def get_augmented_patches_normal(self):
        radii = self._radii
        list_of_dicts = []
        slices = []
        z_slices = [self.blockSize //2, self._image.shape[0] // 2, self._image.shape[0]-self.blockSize//2]
        x_slices = [0.6, 1.8]
        y_slices = [0.9,1.5]

        x_center = ((self._lungs_bounding_box[0][2] + self._lungs_bounding_box[1][2]) // 2) - 10
        y_center = ((self._lungs_bounding_box[0][1] + self._lungs_bounding_box[1][1]) // 2) - 40

        print("Center: x: {}  y: {}".format(x_center,y_center))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    origin = (
                        int(z_slices[k]),
                        int(max( min(y_slices[j] * y_center, self._image.shape[1]-self.blockSize//2), self.blockSize//2 )),
                        int(max( min(x_slices[i] * x_center, self._image.shape[2]-self.blockSize//2), self.blockSize//2 ))
                    )

                    print("Origin: ",origin, " Image Shape: ",self._image.shape)

                    img, radii2, centers, lungs_bounding_box, spacing, existing_nodules_in_patch = \
                        self._get_augmented_patch_normal(center_of_cube=origin)
                    existing_radii = [radii2[i] for i in existing_nodules_in_patch]
                    existing_centers = [centers[i] for i in existing_nodules_in_patch]

                    if(self.annotationExist):
                        subdir = 'negatives' if self._clazz == 0 else 'positives'
                    else:
                        subdir = 'unlabeled'
                    
                    
                    file_path = f'''augmented\\{subdir}\\{self._seriesuid}_{i}_{j}_{k}.npy'''
                    list_of_dicts.append(
                        {'seriesuid': self._seriesuid, 'centers': existing_centers, 'sub_index': f'{i}_{j}_{k}',
                            'lungs_bounding_box': lungs_bounding_box, 'radii': existing_radii, 'class': self._clazz})
                    np.save(f'{self.outputPath}\\{file_path}', img)
                    print("Saving: ",{file_path})


        return list_of_dicts


RESOURCE_PATH = "C:\\Users\\kutay\\OneDrive\\Masaüstü\\Projeler\\Bitirme\\uploads"
OUTPUT_PATH = "C:\\Users\\kutay\\OneDrive\\Masaüstü\\Projeler\\Bitirme\\preprocessed"
BLOCK_SIZE = 128
ANNOTATION_EXIST = True

if __name__ == "__main__":
    preprocess = Preprocess(resourcePath=RESOURCE_PATH, outputPath=OUTPUT_PATH,annotationExist=ANNOTATION_EXIST)
    preprocess.save_preprocessed_data()

    augmentation = Augmentation(resourcePath=RESOURCE_PATH, outputPath=OUTPUT_PATH, annotationExist=ANNOTATION_EXIST)
    augmentation.save_augmented_data()