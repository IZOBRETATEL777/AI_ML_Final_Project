import torch
from torchvision import transforms
from models.example_model import ExModel
from PIL import Image

def load_model(checkpoint_path, device):
    model = ExModel()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model

def predict(image_path, model, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([100, 100]),
        transforms.RandomHorizontalFlip(),   # Flip horizontally
        transforms.RandomRotation(10),       # Rotation by 20 degrees
        transforms.RandomVerticalFlip(),     # Flip vertically
        transforms.RandomAffine(degrees=0, translate=(0.3, 0.3)),  # Random affine transformation for zoom
        transforms.ColorJitter(contrast=0.5) 
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=0)
    
    return prediction.item()




def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("checkpoints/saved_model.pth", device)
    image_path = "datasets/real_images/test_image_1.jpg"

    prediction = predict(image_path, model, device)
    if (prediction == 1):
        print(f"Predicted class: No Glasses")
    else:
        print(f"Predicted class: Glasses")

if __name__ == "__main__":
    main()
