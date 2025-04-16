import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt


def load_image(image_path):
    """Load and convert image to grayscale."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    return image


def apply_wavelet_decomposition(image, wavelet='db1', level=1):
    """Apply Daubechies wavelet decomposition to the input image."""
    coeffs = pywt.dwt2(image, wavelet)
    LL, (LH, HL, HH) = coeffs
    return LL, LH, HL, HH


def display_subbands(LL, LH, HL, HH):
    """Display the four subbands obtained from wavelet decomposition."""
    titles = ['LL', 'LH',
              'HL', 'HH']
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for ax, title, subband in zip(axes.ravel(), titles, [LL, LH, HL, HH]):
        ax.imshow(subband, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('fig0c.png')

def reconstruct_image(LL, LH, HL, HH, wavelet='db1'):
    """Reconstruct the original image from the wavelet subbands."""
    coeffs = LL, (LH, HL, HH)
    reconstructed_image = pywt.idwt2(coeffs, wavelet)
    return reconstructed_image.astype(np.uint8)

def save_reconstructed_image(reconstructed_image, output_path):
    """Save the reconstructed image to a file."""
    cv2.imwrite(output_path, reconstructed_image)
    print(f"Reconstructed image saved to {output_path}")

def main():
    # Path to your input image
    image_path = 'images/fig0c.jpg'  # Replace with your image path

    try:
        # Load the image
        image = load_image(image_path)

        # Apply Daubechies wavelet decomposition
        LL, LH, HL, HH = apply_wavelet_decomposition(image, wavelet='db1')

        # Display the results
        print("Wavelet decomposition completed. Subbands:")
        display_subbands(LL, LH, HL, HH)

        # Reconstruct the image
        reconstructed_image = reconstruct_image(LL, LH, HL, HH, wavelet='db1')

        # Save the reconstructed image
        output_path = 'reconstructed_image.jpg'
        save_reconstructed_image(reconstructed_image, output_path)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()