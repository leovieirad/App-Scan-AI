import cv2
import numpy as np
import os
import random


def apply_perspective_transform(image):
    h, w = image.shape[:2]
    src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst_points = np.float32([
        [random.randint(0, int(w * 0.1)), random.randint(0, int(h * 0.1))],
        [w - random.randint(0, int(w * 0.1)), random.randint(0, int(h * 0.1))],
        [random.randint(0, int(w * 0.1)), h - random.randint(0, int(h * 0.1))],
        [w - random.randint(0, int(w * 0.1)), h - random.randint(0, int(h * 0.1))]
    ])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(image, matrix, (w, h))


def apply_random_crop(image, crop_size=(0.8, 0.8)):
    h, w = image.shape[:2]
    crop_h, crop_w = int(h * crop_size[0]), int(w * crop_size[1])
    start_x = random.randint(0, w - crop_w)
    start_y = random.randint(0, h - crop_h)
    cropped = image[start_y:start_y + crop_h, start_x:start_x + crop_w]
    return cv2.resize(cropped, (w, h))


def apply_hsv_transform(image, saturation_scale=1.5, hue_shift=10):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_image[..., 1] *= saturation_scale
    hsv_image[..., 0] += hue_shift
    hsv_image = np.clip(hsv_image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def apply_quality_reduction(image, quality=30):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_image = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(encoded_image, 1)


def augment_image(image, base_filename, output_dir):
    counter = 1

    for angle in [-15, 0, 15]:
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        rotated = cv2.warpAffine(image, M, (w, h))
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_rotated_{counter}.jpg"), rotated)
        counter += 1

    flipped_h = cv2.flip(image, 1)
    cv2.imwrite(os.path.join(output_dir, f"{base_filename}_flipped_h.jpg"), flipped_h)

    perspective = apply_perspective_transform(image)
    cv2.imwrite(os.path.join(output_dir, f"{base_filename}_perspective.jpg"), perspective)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise = np.random.normal(0, 25, gray_image.shape).astype(np.uint8)
    noisy_gray = cv2.add(gray_image, noise)
    cv2.imwrite(os.path.join(output_dir, f"{base_filename}_gray_noisy.jpg"), noisy_gray)

    gamma = random.uniform(0.5, 1.5)
    gamma_correction = np.array(255 * (image / 255) ** gamma, dtype='uint8')
    cv2.imwrite(os.path.join(output_dir, f"{base_filename}_gamma.jpg"), gamma_correction)

    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-20, 20)
    bright_contrast = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    cv2.imwrite(os.path.join(output_dir, f"{base_filename}_bright.jpg"), bright_contrast)

    cropped = apply_random_crop(image)
    cv2.imwrite(os.path.join(output_dir, f"{base_filename}_cropped.jpg"), cropped)

    hsv_transformed = apply_hsv_transform(image)
    cv2.imwrite(os.path.join(output_dir, f"{base_filename}_hsv.jpg"), hsv_transformed)

    low_quality = apply_quality_reduction(image)
    cv2.imwrite(os.path.join(output_dir, f"{base_filename}_low_quality.jpg"), low_quality)


def augment_dataset(image_paths, output_dir):
    os.makedirs(output_dir, exist_ok = True)

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is not None:
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            augment_image(image, base_filename, output_dir)


path = r"D:\Base_de_dados\Logo_Itau"
original_image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

augmented_path = os.path.join(path, "augmented_images")

augment_dataset(original_image_paths, augmented_path)
