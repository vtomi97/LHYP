# Reads the sa folder wiht dicom files and contours
# then draws the contours on the images.

from con_reader import CONreaderVM
from dicom_reader import DCMreaderVM
from con2img import draw_contourmtcs2image as draw
from os.path import join as pjoin
import os


def contours2images(patient_folder):
    image_folder = pjoin(patient_folder, 'sa', 'images')
    con_file = pjoin(patient_folder, 'sa', 'contours.con')
    # reading the dicom files
    dr = DCMreaderVM(image_folder)

    # reading the contours
    cr = CONreaderVM(con_file)
    contours = cr.get_hierarchical_contours()

    # drawing the contours for the images
    for slc in contours:
        for frm in contours[slc]:
            image = dr.get_image(slc, frm)  # numpy array
            cntrs = []
            rgbs = []
            for mode in contours[slc][frm]:
                # choose color
                if mode == 'ln':  # left endocardium -> red
                    rgb = [1, 0, 0]
                elif mode == 'lp':  # left epicardium -> green
                    rgb = [0, 1, 0]
                elif mode == 'rn':  # right endocardium -> yellow
                    rgb = [1, 1, 0]
                else:
                    rgb = None
                if rgb is not None:
                    cntrs.append(contours[slc][frm][mode])
                    rgbs.append(rgb)
            if len(cntrs) > 0:
                draw(image, cntrs, rgbs)


if __name__ == '__main__':
    path_to_sample = 'samples'
    for pf in [pjoin(path_to_sample, folder_name) for folder_name in os.listdir(path_to_sample)]:
        print(pf)
        contours2images(pf)
