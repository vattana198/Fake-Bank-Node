import streamlit as st
from PIL import Image
import io
import torch
from torchvision import transforms
from models import Discriminator
import math


def load_discriminator(checkpoint_path, device):
    # Define the Discriminator architecture with the correct parameters
    ndf = 64  # Number of base filters
    im_size = 256  # Image size (must match training)
    netD = Discriminator(ndf=ndf, im_size=im_size)

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print("Checkpoint keys:", checkpoint.keys())  # Debugging checkpoint structure

    if 'd' in checkpoint:
        # Strip 'module.' prefix from keys if saved with DataParallel
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['d'].items()}
        netD.load_state_dict(state_dict)
    else:
        raise KeyError("'d' key not found in the checkpoint file.")

    netD.eval()  # Set the model to evaluation mode
    netD.to(device)  # Move to the specified device
    print("Discriminator successfully loaded.")
    return netD

# Preprocess the input image
def preprocess_image(image, im_size):
    transform = transforms.Compose([
        transforms.Resize((im_size, im_size)),  # Resize image to match input size
        transforms.ToTensor(),                 # Convert image to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


def predict_discriminator(discriminator, image_tensor, label, device):
    """
    Predicts whether the input image is real or fake using the discriminator.

    Args:
        discriminator (torch.nn.Module): The discriminator model.
        image_tensor (torch.Tensor): The input image tensor.
        label (torch.Tensor): The label tensor corresponding to the input.
        device (torch.device): The device to run the model on.

    Returns:
        torch.Tensor: The output prediction score.
    """
    discriminator.eval()  # Set to evaluation mode
    image_tensor = image_tensor.to(device)
    label = label.to(device)  # Move label to the same device

    with torch.no_grad():  # Disable gradient calculation
        output = discriminator(image_tensor, label)

    # Check if the output is a batch (more than one element)
    if isinstance(output, torch.Tensor) and output.dim() > 1:
        output = output.squeeze()  # Remove extra dimensions (e.g., batch size)

    # Ensure the output is a scalar (if it is a batch, take the first element)
    if output.numel() == 1:
        return output.item()  # Return scalar value
    elif output.numel() > 1:
        return output.mean().item()  # Use the average of the batch if multiple values are returned

    return output


# Main logic
if __name__ == "__main__":
    # Set device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the discriminator and preprocess the image
    netD = load_discriminator("/Users/macbookpro/Desktop/Fake-Bank-Note/all_50000.pth", device)

    # Set page configuration
    st.set_page_config(page_title="Fake Banknote Detector", layout="centered")

    # Import image for the home page
    try:
        image = Image.open("Fakebanknote.png")
    except FileNotFoundError:
        image = None  # Gracefully handle the absence of the image

    # Sidebar navigation
    menu = st.sidebar.selectbox("Navigation", ["Home", "About Us", "Fake Banknote Detector"])

    # Define each page
    if menu == "Home":
        st.title("Welcome to Fake Banknote Detector")
        if image:
            st.image(image, caption="Detect the authenticity of your banknotes.")
        else:
            st.warning("Home page image not found.")
        st.markdown("""
            <div style="text-align: center;">
                This app helps you verify whether a Cambodian riel banknote is genuine or counterfeit using advanced image analysis techniques.<br>
                Navigate through the app to learn more and try the detection tool.
            </div>
        """, unsafe_allow_html=True)

    elif menu == "About Us":
        st.title("About Us")
        st.write("""
            **Fake Banknote Detector** is developed by a team of dedicated professionals 
            committed to enhancing financial security. Our goal is to provide individuals 
            and businesses with reliable tools to detect counterfeit currency.
        """)
        st.image("https://via.placeholder.com/400", caption="Our Mission")

    elif menu == "Fake Banknote Detector":
        st.title("Fake Banknote Detector")

        # Step 1: Upload Image Functionality
        st.subheader("Upload or Capture an Image of a Riel Banknote")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
        picture = st.camera_input("Or capture a photo")

        image = None
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Riel Banknote", use_column_width=True)
        elif picture:
            image = Image.open(io.BytesIO(picture.getvalue()))
            st.image(image, caption="Captured Riel Banknote", use_column_width=True)

        # Step 2: Predict Button
        if image is not None:
            if st.button("Predict"):
                st.write("Analyzing the banknote...")
                image_tensor = preprocess_image(image, im_size=256)  # Preprocess the uploaded image
                label_tensor = torch.zeros(1, dtype=torch.float32).to(device)  # Assuming "0" means "fake"
                
                # Predict using the discriminator
                prediction = predict_discriminator(netD, image_tensor, label_tensor, device)
                # Step 3: Display Results with Prediction Score
                prediction_score = abs(prediction)
                st.write(f"Prediction Score: {prediction_score:.4f}") 
                # Step 3: Display Results
                if prediction_score >= 0.4 and prediction_score < 0.9: 
                    st.success("This banknote appears to be genuine.")
                else:
                    st.error("This banknote is likely counterfeit.")
        else:
            st.info("Please upload or capture an image to enable prediction.")


    # Footer
    st.sidebar.info("Developed by [Group 9]. For any inquiries, contact us.")
