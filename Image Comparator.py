from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_ssim_iou(predicted_path, actual_path, threshold=0.5):
    # Load and convert images to grayscale
    predicted_img = Image.open(predicted_path).convert('L')
    actual_img = Image.open(actual_path).convert('L')

    # Resize actual to match predicted if needed
    if predicted_img.size != actual_img.size:
        actual_img = actual_img.resize(predicted_img.size)

    # Convert to numpy arrays
    predicted_array = np.array(predicted_img) / 255.0
    actual_array = np.array(actual_img) / 255.0

    # Compute SSIM
    ssim_index, _ = ssim(predicted_array, actual_array, full=True, data_range=1.0)

    # Create binary masks
    pred_mask = predicted_array > threshold
    actual_mask = actual_array > threshold

    # Compute IoU
    intersection = np.logical_and(pred_mask, actual_mask)
    union = np.logical_or(pred_mask, actual_mask)
    iou_score = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0

    return ssim_index, iou_score

# Example usage:
predicted_image_path = "C:/Users/Ritchie Strachan\Desktop/2024-25 Science Fair/IMG_4298a prediction v.png"
actual_image_path = "C:/Users/Ritchie Strachan\Desktop/2024-25 Science Fair/IMG_4298a ice.PNG"

ssim_score, iou_score = compute_ssim_iou(predicted_image_path, actual_image_path)

print(f"Structural Similarity Index (SSIM): {ssim_score:.4f}")
print(f"Intersection over Union (IoU): {iou_score:.4f}")
