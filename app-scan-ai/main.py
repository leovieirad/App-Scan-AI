import cv2
import os
import numpy as np
from machine_learning import analyze_full_images
from PIL import Image


def extract_data_logos(image_logos, image_paths, app):
    
    metadata = {
        
        "width": [],
        "height": [],
        
    }
    
    metadata_img = {
        "name": [],
        "width": [],
        "height": []
  
    }
    
    image_dimensions = {}   
    
    for img_path in image_paths:
        image_name = os.path.basename(img_path)
        with Image.open(img_path) as img:
            img_width, img_height = img.size
            image_dimensions[img_path] = (img_width, img_height)
        
        metadata_img["name"].append(image_name)    
        metadata_img["width"].append(img_width)
        metadata_img["height"].append(img_height)  
    
    app.metadata_images =  metadata_img
                
    
    for logo_path in image_logos:
        
        with Image.open(logo_path) as logo_img:
            
            width, height = logo_img.size

        img_path = max(image_dimensions, key = lambda path: image_dimensions[path][0] * image_dimensions[path][1])
        img_width, img_height = image_dimensions[img_path]
        
        metadata["width"].append(width)
        metadata["height"].append(height)
    
    app.metadata_logos = metadata

                        
def format_data(app):
    
    metadata_logos = app.metadata_logos
    metadata_img = app.metadata_images
    
    for i in range(len(metadata_logos["width"])):
        
        logo_width = int(metadata_logos["width"][i])
        logo_height = int(metadata_logos["height"][i])    
        
        if i >= len(app.format_data):
            app.format_data.append({})

        app.format_data[i]['logo_width'] = logo_width
        app.format_data[i]['logo_height'] = logo_height
    
    for i in range(len(metadata_img["name"])):
        img_name = metadata_img["name"][i]
        img_width = metadata_img["width"][i]
        img_height = metadata_img["height"][i]
        
        if i >= len(app.format_data):
            app.format_data.append({})
        
        app.format_data[i]['name'] = img_name
        app.format_data[i]['width'] = img_width
        app.format_data[i]['height'] = img_height  
                    

def detect_logo(image_paths, image_logos, folder, models, update_progress = None, app = None):
    
    analyze_full_images(image_paths, image_logos, models, app)
    format_data(app)
    sift = cv2.SIFT_create(contrastThreshold=0.01, edgeThreshold=5)
    logos = []
    logo_widths = []
    logo_heights = []
    logo_keypoints_descriptors = []
    
    total_images = len(image_paths)
    svm_detections = app.svm_detections
    metadata_img = app.format_data
    
    for logo_path in image_logos:
        logo = cv2.imread(logo_path, 0)
        if logo is not None:
            logos.append(logo)
            kp_logo, desc_logo = sift.detectAndCompute(logo, None)
            logo_keypoints_descriptors.append((kp_logo, desc_logo))
    
    for kp_logo, _ in logo_keypoints_descriptors:
        if kp_logo:
            x_coords = [kp.pt[0] for kp in kp_logo]
            y_coords = [kp.pt[1] for kp in kp_logo]
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            logo_widths.append(width)
            logo_heights.append(height)

    if not logo_widths or not logo_heights:
        raise ValueError("Nenhuma largura ou altura válida foi encontrada nas logos.")

    avg_logo_width = np.mean(logo_widths)
    avg_logo_height = np.mean(logo_heights)
    
    all_windows = svm_detections["best_windows"] + svm_detections["fallback_windows"]

    for idx, image_path in enumerate(image_paths):
        
        image = cv2.imread(image_path)
        image_name = os.path.basename(image_path)
        best_M = None
        best_detections = []
        
        params_img = next((item for item in metadata_img if item["name"] == image_name), None)
        img_width = params_img["width"]
        img_height = params_img["height"]

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_windows = [win for win in all_windows if win["name"] == image_name]

        for i, detection in enumerate(image_windows):
            left = detection["left"]
            top = detection["top"]
            right = detection["right"]
            bottom = detection["bottom"]
            
            region_gray = gray_image[top:bottom, left:right]
            
            if region_gray.size == 0:
                print(f"Região inválida ou fora dos limites para a imagem {image_name}")
                continue

            region_gray = cv2.equalizeHist(region_gray)
            region_gray = cv2.GaussianBlur(region_gray, (5, 5), 0)

            region_kp_image, region_desc_image = sift.detectAndCompute(region_gray, None)
            
            if region_desc_image is None:
                print("Nenhum descritor encontrado na região.")
                continue
            
            for kp_logo, desc_logo in logo_keypoints_descriptors:

                trees = 30
                checks = 70
                lowe = 0.8
                ransac = 5.0
                good_matches = 10

                index_params = dict(algorithm = 1, trees = trees)
                search_params = dict(checks = checks)
                flann = cv2.FlannBasedMatcher(index_params, search_params)

                k = min(2, len(region_desc_image))
                if k > 1:
                    matches = flann.knnMatch(desc_logo, region_desc_image, k=k)
                else:
                    print("Correspondências insuficientes para esta região.")
                    continue

                good_matches_filtered = []

                for m, n in matches:
                    if m.distance < lowe * n.distance:
                        good_matches_filtered.append(m)
                
                if len(good_matches_filtered) < good_matches:
                    print("Correspondências boas insuficientes para esta região.")
                    continue
                
                inlier_scores = []   
                
                if len(good_matches_filtered) >= good_matches:
                    src_pts = np.float32([kp_logo[m.queryIdx].pt for m in good_matches_filtered]).reshape(-1, 1, 2)
                    dst_pts = np.float32([region_kp_image[m.trainIdx].pt for m in good_matches_filtered]).reshape(-1, 1, 2)

                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac)

                    if M is not None and mask is not None:
                        num_inliers = np.sum(mask)
                        inlier_scores.append((num_inliers, M, mask))
                        print(f"[MATCH] {num_inliers} inliers encontrados para região ({left}, {top}, {right}, {bottom})")
                    inlier_scores = sorted(inlier_scores, key=lambda x: x[0], reverse=True)
            
            for rank, (num_inliers, M, mask) in enumerate(inlier_scores[:3]):
                h, w = avg_logo_height, avg_logo_width
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                transformed_pts = cv2.perspectiveTransform(pts, M)

                x_min = np.min(transformed_pts[:, 0, 0]) + left
                y_min = np.min(transformed_pts[:, 0, 1]) + top
                x_max = np.max(transformed_pts[:, 0, 0]) + left
                y_max = np.max(transformed_pts[:, 0, 1]) + top

                best_detections.append((x_min, y_min, x_max, y_max))
                break

            if not best_detections and inlier_scores:
                num_inliers, best_M, best_mask = inlier_scores[0]
                h, w = avg_logo_height, avg_logo_width
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                transformed_pts = cv2.perspectiveTransform(pts, best_M)

                x_min = np.min(transformed_pts[:, 0, 0]) + left
                y_min = np.min(transformed_pts[:, 0, 1]) + top
                x_max = np.max(transformed_pts[:, 0, 0]) + left
                y_max = np.max(transformed_pts[:, 0, 1]) + top

                best_detections.append((x_min, y_min, x_max, y_max))

            if best_detections:
                x_min, y_min, x_max, y_max = best_detections[0]

                # Ajuste único com média
                width_detected = x_max - x_min
                height_detected = y_max - y_min

                if width_detected < avg_logo_width:
                    expand_width = avg_logo_width - width_detected
                    x_min -= expand_width / 2
                    x_max += expand_width / 2

                if height_detected < avg_logo_height:
                    expand_height = avg_logo_height - height_detected
                    y_min -= expand_height / 2
                    y_max += expand_height / 2

                # Garantir limites válidos
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(img_width, x_max)
                y_max = min(img_height, y_max)

                rect_pts = np.array([
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max]
                ], dtype=np.float32).reshape(-1, 1, 2)

                image = cv2.polylines(image, [np.int32(rect_pts)], True, (0, 255, 0), 3)

        save_path = os.path.join(folder, os.path.splitext(image_name)[0] + "_marked.jpg")
        cv2.imwrite(save_path, image)

        if update_progress:
            current_progress = (idx + 1) / total_images
            update_progress(current_progress)