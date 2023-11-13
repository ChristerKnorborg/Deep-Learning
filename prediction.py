import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from model_constants import S, B, DEVICE
from yolo_network import Yolo_network


def display_predictions(model_path, original_image, confidence_threshold=0.5):
    """
    Display the original and predicted images side-by-side.

    :param model: The YOLO model to use for prediction.
    :param original_image: The original image (PIL Image or a tensor that can be converted).
    :param device: The device on which the model and image are located.
    :param S: The number of grid cells along each dimension (default is 7 for YOLO).
    :param confidence_threshold: Minimum confidence required to display a predicted bounding box.
    """

    # Prepare the image
    transform = transforms.Compose([
        transforms.CenterCrop((512, 512)),
        transforms.ToTensor(),
    ])

    # Apply the transformations and add a batch dimension
    image_tensor = transform(original_image).unsqueeze(0).to(DEVICE)

    # Put model in eval mode and perform a forward pass to get the predictions
    model = Yolo_network()  
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(DEVICE)  
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)

    # Squeeze the predictions in case we have a batch dimension
    predictions = torch.squeeze(predictions, 0)

    # Create a figure with plots
    fig, axarr = plt.subplots(1, 2, figsize=(12, 6))

    # Helper function to draw the SxS grid on an image
    def draw_grid(ax, width, height):
        for i in range(1, S):
            ax.axvline(x=i * width / S, color='blue', linewidth=0.2)
            ax.axhline(y=i * height / S, color='blue', linewidth=0.2)

    # Show the original image on the left
    axarr[0].imshow(original_image)
    draw_grid(axarr[0], *original_image.size)  # Draw the grid
    axarr[0].set_title('Original Image')
    axarr[0].axis('off')

    # Prepare to show the image with predictions on the right
    axarr[1].imshow(original_image)  # We re-use the original image
    draw_grid(axarr[1], *original_image.size)  # Draw the grid
    axarr[1].set_title('Predictions')
    axarr[1].axis('off')

    # Process predictions
    cell_width = original_image.size[0] / S
    cell_height = original_image.size[1] / S

    for i in range(S):
        for j in range(S):
            # Extract data for a single cell
            cell_data = predictions[i, j]

            # Two sets of predictions; only consider the one with the highest confidence
            if cell_data[0] > cell_data[5]:
                confidence = cell_data[0]
                bbox = cell_data[1:5]
            else:
                confidence = cell_data[5]
                bbox = cell_data[6:10]

            if confidence >= confidence_threshold:
                # Calculate absolute coordinates and dimensions
                x_center, y_center, width, height = bbox
                x_center, y_center = x_center * cell_width + i * \
                    cell_width, y_center * cell_height + j * cell_height
                width, height = width * \
                    original_image.size[0], height * original_image.size[1]

                # Convert to top-left corner coordinates
                x_corner, y_corner = x_center - width / 2, y_center - height / 2

                # Draw the bounding box
                rect = patches.Rectangle(
                    (x_corner, y_corner), width, height, linewidth=1, edgecolor='r', facecolor='none')
                axarr[1].add_patch(rect)

    plt.tight_layout()
    plt.show()


test_image = "data/val2017/person/000000001296.jpg"  # Snowboarder image

original_image = Image.open(test_image)
model_path = "./models/model_11-12_00_epoch-80_LR-0.0001_step-none_gamma-none_subset-30000_batch-64.pth"


display_predictions(model_path, original_image, confidence_threshold=0.5)
