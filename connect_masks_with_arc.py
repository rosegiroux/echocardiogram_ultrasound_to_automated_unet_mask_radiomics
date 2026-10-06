#! C:/Users/rlong5/AppData/Local/anaconda3/envs/python39/python

#in conda terminal: conda activate python39
from pathlib import Path
import os
import cv2
import nibabel as nib
import numpy as np
import sys

def score_arc(img, pts, width=8):
    #score small band around arc 

    arc_mask = np.zeros(img.shape, np.uint8)

    for i in range(len(pts) - 1):
        cv2.line(
            arc_mask,
            tuple(pts[i]),
            tuple(pts[i+1]),
            255,
            width
        )

    values = img[arc_mask > 0]

    if len(values) == 0:
        return -np.inf

    return (
        np.mean(values)
        + 0.5 * np.percentile(values, 90)
        - 0.25 * np.std(values)
    )

def list_output_masks(
        the_mask: np.ndarray): #-> Tuple [np.ndarray, List[Dict]]:
    import cv2
    h, w = the_mask.shape

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        the_mask.astype(np.uint8),
        connectivity=8,
    )

    objects = []

    for i in range(0, n_labels):  # starting at one skips background (label 0)
        objects.append({
            "id": i,
            "center": centroids[i],
            "bbox": {
                "x": stats[i, cv2.CC_STAT_LEFT],
                "y": stats[i, cv2.CC_STAT_TOP],
                "w": stats[i, cv2.CC_STAT_WIDTH],
                "h": stats[i, cv2.CC_STAT_HEIGHT],
            }
        })

    return labels, objects

def score_arc_dark_bad(img, pts, width=8):

    arc_mask = np.zeros(img.shape[:2], np.uint8)

    for i in range(len(pts)-1):
        cv2.line(
            arc_mask,
            tuple(pts[i]),
            tuple(pts[i+1]),
            255,
            width
        )

    values = img[arc_mask > 0]

    if len(values) == 0:
        return -np.inf

    dark_fraction = np.mean(values < 75)

    return (
        np.mean(values)
        - 100 * dark_fraction
    )

def find_file_recursive(root, filename):
    """
    find first mathing filename in root
    arguments: the directory to search, the filename to search
    returns full path or None"""

    for dirpath, dirnames, filenames in os.walk(root):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None

def save_nifti(path_to_export_nifti,
    image,
    mask,
    img_nii_name="original_image",
    mask_nii_name="first_mask_unet",
    text = None): #connected_mask, last: connected_mask_nii_name="connected_mask"
    """
    Save image and masks as NIfTI files.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    mask : np.ndarray
        Original mask.
    connected_mask : np.ndarray
        Connected mask.
    """

    os.makedirs(path_to_export_nifti, exist_ok=True)

    header = nib.Nifti1Header()
    if text:
        ext = nib.nifti1.Nifti1Extension(
        code="comment",
        content=str(text).encode("utf-8")
        )

        header.extensions.append(ext)

    nib.save(nib.Nifti1Image(image.astype(np.float32), None, header),
        os.path.join(path_to_export_nifti, img_nii_name))

    nib.save(nib.Nifti1Image(mask.astype(np.uint8), None, header),
        os.path.join(path_to_export_nifti, mask_nii_name))

    # nib.save(nib.Nifti1Image(connected_mask.astype(np.uint8), None, header),
    #     os.path.join(path_to_export_nifti, connected_mask_nii_name))


if __name__ == "__main__":
    #decide to choose upper right 80% of mask

    # connect masks with arc, I want to connect the masks in the upper right 4/5ths 
    # def upper_right():
    #     h, w = mask.shape
    #     # Upper-right region:
    #     # top 80% of rows
    #     # right 80% of columns
    #     roi = np.zeros_like(mask)

    #     roi[: int(0.8 * h), int(0.2 * w) :] = 1

    #     mask_roi = mask * roi

    # subprocess.run([r"path to virtual env", "script name.py"])
    # output = rf"C:\Users\rlong5\Desktop\results_20260715_full\nifti_mask_image_20260820"

    MASK_ROOT = rf"C:\Users\rlong5\Desktop\results_20260715_full\masks\Echo_amyloid_HCM_pngs"#\6865620"

    IMAGE_ROOT = rf"C:\Users\rlong5\SAMUS\test_images\Echo_amyloid_HCM_pngs"#6865620"


    OUTPUT_ROOT = rf"C:\Users\rlong5\Desktop\results_20260715_full\masks_arc_connected"#\6865620"

    for dirpath, dirnames, filenames in os.walk(MASK_ROOT):

        for mask_filename in filenames:

            mask_file_path = os.path.join(dirpath, mask_filename)

            print(mask_file_path)

            image_file_path = find_file_recursive(IMAGE_ROOT, filename=mask_filename)
            print("matching image in ", image_file_path)

            #search for the file in IMAGE_ROOT

            # create new mask
            #1) optional--skip for now...choose upper right masks
            #2) c

            mask_img = cv2.imread(mask_file_path) #the mask image is in two values 0, 1, may be non-binarized though
            #this will import s greyscale
            image_img = cv2.imread(image_file_path)

            mask_img = np.any(mask_img > 0, axis = 2).astype(np.uint8) #it is already binarized by test_clinical.py

            if mask_img.shape != image_img.shape:
                print("the image is ", image_img.shape, "while the mask is ", mask_img.shape)

                image_img = cv2.resize(
                    image_img,
                    (mask_img.shape[1], mask_img.shape[0]), # width, height
                    interpolation=cv2.INTER_LINEAR
                    )

            print("Resized image to", image_img.shape)
            # connect mask object to its nearest neighbor, generate an arc, and score the arc using your score_arc().

            def closest_contour_points(labels, id1, id2):
                """

                Find the closest pair of contour points between two components.
                Returns:
                p1, p2, distance

                """
                mask1 = (labels == id1).astype(np.uint8)
                mask2 = (labels == id2).astype(np.uint8)

                cnt1, _ = cv2.findContours(
                    mask1,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_NONE
                )

                cnt2, _ = cv2.findContours(
                    mask2,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_NONE
                )

                pts1 = cnt1[0][:, 0, :]
                pts2 = cnt2[0][:, 0, :]

                dists = np.sqrt(
                    np.sum((pts1[:, None] - pts2[None, :]) ** 2, axis=2)
                )

                i, j = np.unravel_index(np.argmin(dists), dists.shape)

                return pts1[i], pts2[j], dists[i, j]

            def connect_masks_with_best_arc(img, mask, remove_left_percent = 0.10):
                """
                Find connected components in a binary mask.

                For each component:
                    - Create a curved arc between their closest controur points.
                    - Score the arc using image intensity.
                    - Draw the highest-scoring arc.

                Returns:
                    Updated binary mask with connecting arcs added.
                """
                h, w = mask.shape
                left_cutoff = int(w * remove_left_percent)
                objects = []

                # Find connected components
                n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    mask.astype(np.uint8),
                    connectivity=8,
                )
                print("n_labels =", n_labels)
                # Nothing to connect
                # (label 0 = background)
                if n_labels <= 2:
                    return mask

                # Store component IDs and centroids
                for i in range(1, n_labels):
                    # Leftmost x coordinate of this component
                    left_edge = stats[i, cv2.CC_STAT_LEFT]
                    # Skip components touching the left side
                    if left_edge <= left_cutoff:
                        print(
                            f"Removing component {i}: "
                            f"left edge={left_edge}, cutoff={left_cutoff}"
                        )
                        continue
                    objects.append({"id": i,"center": centroids[i]})

                result = mask.copy() #copy oriingal mask array incl dimesions
                used = set()
                connections = []

                for obj1 in objects:
                    # Skip components already connected
                    if obj1["id"] in used:
                        continue

                    best_score = -np.inf
                    best_pts = None
                    best_id = None
                    best_dist = None

                    # Search all other components
                    for obj2 in objects:

                        if obj2["id"] == obj1["id"]:
                            continue

                        # closest edge points between masks
                        p1, p2, dist = closest_contour_points(
                            labels,
                            obj1["id"],
                            obj2["id"]
                        )

                        # Ignore very distant components
                        #this does not work. distances are np.float32 100
                        if dist > 80:
                            continue

                        p1 = np.asarray(p1, dtype=float)
                        p2 = np.asarray(p2, dtype=float)
                        # Arc control point:
                        # midpoint between closest contour points
                        midpoint = (p1 + p2) / 2
                        #control point to bow arc upward
                        control = np.array([
                            midpoint[0],
                            min(p1[1], p2[1]) - 0.2 * dist,
                        ])

                        # Sample points along quadratic Bézier curve
                        t = np.linspace(0, 1, 100)

                        pts = np.array([
                            (1 - tt) ** 2 * p1
                            + 2 * (1 - tt) * tt * control
                            + tt ** 2 * p2
                            for tt in t
                        ]).astype(np.int32)

                        # Score arc using underlying image intensity
                        score = score_arc_dark_bad(img, pts)

                        if score > best_score:
                            best_score = score
                            best_pts = pts
                            best_id = obj2["id"]
                            best_dist = dist

                    # Draw best connection
                    if best_pts is not None:

                        cv2.polylines(result,[best_pts], isClosed=False, color=1, thickness=8)

                        connections.append({"id1": obj1["id"], "id2": best_id,"score": best_score,"distance": best_dist})

                        used.add(obj1["id"])
                        used.add(best_id)
                print(result.dtype)
                return result, connections #copy of oringinal mask with best scoring arcs drawn onto it

            connected_mask, connections = connect_masks_with_best_arc(image_img, mask_img)

            print(connections)

            # Preserve folder structure
            rel_dir = os.path.relpath(dirpath, MASK_ROOT)
            output_dir = os.path.join(OUTPUT_ROOT, rel_dir)
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(mask_filename)[0]

            save_nifti(OUTPUT_ROOT, image_img, mask_img, connected_mask,
                    img_nii_name=base_name + "_image.nii.gz",
                        mask_nii_name=base_name + "_mask.nii.gz", 
                        connected_mask_nii_name=base_name + "_connected_mask.nii.gz")
            # save old and new mask in nifti

            # Save connected mask as PNG
            connected_mask_path = os.path.join(output_dir, base_name + "connected_mask.png")
            cv2.imwrite(connected_mask_path, (connected_mask * 255).astype(np.uint8))

            # # close arc
            # kernel = cv2.getStructuringElement(
            #     cv2.MORPH_ELLIPSE,
            #     (15,15)
            # )

            # new_mask = cv2.morphologyEx(
            #     new_mask,
            #     cv2.MORPH_CLOSE,
            #     kernel
            # )