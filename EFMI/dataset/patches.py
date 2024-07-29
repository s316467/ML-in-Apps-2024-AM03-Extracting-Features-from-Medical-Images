import argparse
import numpy as np
import cv2 as cv
import os
import openslide
import xml.etree.cElementTree as ET
import os
import re


def parse_xml(xml_path):
    """
    builds the list that represent the ROI path starting from an xml annots file
    """
    tree = ET.ElementTree(file=xml_path)
    annolist = []
    root = tree.getroot()
    for coords in root.iter("Vertices"):
        for coord in coords:
            x = int(float(coord.attrib.get("X")))
            y = int(float(coord.attrib.get("Y")))
            annolist.append((x, y))
    return annolist


def is_not_white_or_gray(patch, threshold=0.95, gray_threshold=200):
    """
    Check if the patch is not predominantly white or light gray.
    """
    gray = cv.cvtColor(np.array(patch), cv.COLOR_RGB2GRAY)
    _, binary = cv.threshold(gray, gray_threshold, 255, cv.THRESH_BINARY)
    foreground_pixels = cv.countNonZero(binary)
    foreground_ratio = foreground_pixels / (patch.shape[0] * patch.shape[1])
    return foreground_ratio < threshold


def point_inside_polygon(x, y, poly):
    """
    Check if a point (x, y) is inside a polygon defined by vertices in `poly`.
    The algorithm used is Ray Casting Method.

    Args:
    - x (float): x-coordinate of the point.
    - y (float): y-coordinate of the point.
    - poly (list of tuples): List of (x, y) coordinates defining the polygon vertices.

    Returns:
    - bool: True if the point is inside the polygon, False otherwise.
    """
    num = len(poly)
    j = num - 1
    odd_nodes = False
    for i in range(num):
        if poly[i][1] < y and poly[j][1] >= y or poly[j][1] < y and poly[i][1] >= y:
            if (
                poly[i][0]
                + (y - poly[i][1])
                / (poly[j][1] - poly[i][1])
                * (poly[j][0] - poly[i][0])
                < x
            ):
                odd_nodes = not odd_nodes
        j = i
    return odd_nodes


def patch_inside_roi(patch_vertices, roi_vertices):
    """
    Check if a patch defined by its vertices is inside (partially or fully) the region of interest (ROI)
    defined by its vertices.

    Args:
    - patch_vertices (list of tuples): List of (x, y) coordinates defining the patch vertices.
    - roi_vertices (list of tuples): List of (x, y) coordinates defining the ROI vertices.

    Returns:
    - bool: True if at least one vertex of the patch is inside the ROI, False otherwise.
    """
    for vertex in patch_vertices:
        x, y = vertex
        if point_inside_polygon(x, y, roi_vertices):
            return True
    return False


def get_patch_vertices(x, y, patch_size):
    tl_x, tl_y = x, y
    tr_x, tr_y = tl_x + patch_size, tl_y
    br_x, br_y = tl_x + patch_size, tl_y + patch_size
    bl_x, bl_y = br_x - patch_size, br_y
    return [(tl_x, tl_y), (tr_x, tr_y), (bl_x, bl_y), (br_x, br_y)]


def scale_coords(coords, downsampling_factor):
    return [(x / downsampling_factor, y / downsampling_factor) for x, y in coords]


# Function to extract patches from the WSI
# TODO: Let this returns (patch, label), handle saving outside of this
def extract_patches(
    image_path, not_roi_path, in_roi_path, roi_coords, patch_size=512, mag_level=1
):
    """
    Extract patches from WSI image at a certain mag level, patches inside roi_coords are saved in a different folder.

    Args:
    - image_path (str): the original WSI image path
    - nc_path (str): path to folder for patches that are outside of roi_cords (no-cancer)
    - wc_path (str): path to folder for patches that are inside of roi_cords (with-cancer)
    - patch_size (int)
    - mag_level (int)

    Returns:
    - It saves patches that are inside the ROI into in_roi_path folder, others into not_roi_path
    """
    slide = openslide.open_slide(image_path)

    downsampling_factor = slide.level_downsamples[mag_level]

    roi_coords = scale_coords(roi_coords, downsampling_factor)

    width, height = slide.level_dimensions[mag_level]
    
    for y in range(0, height - patch_size, patch_size):
        for x in range(0, width - patch_size, patch_size):

            patch = slide.read_region(
                (int(x * downsampling_factor), int(y * downsampling_factor)),
                mag_level,
                (patch_size, patch_size),
            )
          
            if is_not_white_or_gray(np.array(patch)):
                patch_vertices = get_patch_vertices(x, y, patch_size)

                if patch_inside_roi(patch_vertices, roi_coords):
                    patch_path = f"{in_roi_path}/"
                    patch_name = f"{x}_{y}_mag{mag_level}.png"
                else:
                    patch_path = f"{not_roi_path}/"
                    patch_name = f"{x}_{y}_mag{mag_level}.png"

                if not os.path.exists(patch_path):
                    os.makedirs(patch_path)

                # print(f"print_patch({patch_vertices[0][0]}, {patch_vertices[0][1]})")
                patch.save(f"{patch_path}/{patch_name}")


def get_images_and_roi_file_names(ds_path):
    svs_names = []
    xml_names = []

    for file_name in os.listdir(ds_path):
        if file_name.endswith("svs"):
            svs_names.append(file_name)
        elif file_name.endswith("xml"):
            xml_names.append(file_name)

    svs_names = sorted(svs_names, key=lambda x: int(re.search(r"\d+", x).group()))
    xml_names = sorted(xml_names, key=lambda x: int(re.search(r"\d+", x).group()))

    return (svs_names, xml_names)


def extract_all(ds_path, not_roi_path, in_roi_path, patch_size=512, mag_level=1):
    svs_names, xml_names = get_images_and_roi_file_names(ds_path)

    for svs_name, xml_name in zip(svs_names, xml_names):
        image_path = os.path.join(ds_path, svs_name)
        coords_path = os.path.join(ds_path, xml_name)
        roi_coords = parse_xml(coords_path)

        extract_patches(
            image_path,
            f"{not_roi_path}/{svs_name}",
            f"{in_roi_path}/{svs_name}",
            roi_coords,
            patch_size=patch_size,
            mag_level=mag_level,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract patches from WSI images.")
    parser.add_argument(
        "--ds_path", type=str, required=True, help="Path to the dataset directory"
    )
    parser.add_argument(
        "--not_roi_path",
        type=str,
        required=True,
        help="Path to save patches outside ROI",
    )
    parser.add_argument(
        "--roi_path", type=str, required=True, help="Path to save patches inside ROI"
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=512,
        help="Size of the patches to extract, defaults to 512",
    )
    parser.add_argument(
        "--mag_level",
        type=int,
        default=1,
        help="Magnification level to use, defaults to 1",
    )

    args = parser.parse_args()

    extract_all(
        args.ds_path, args.not_roi_path, args.roi_path, args.patch_size, args.mag_level
    )
