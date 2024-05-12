import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from batchnorm import ResNet_BN
from nonorm import ResNet18 as ResNet_NN
from groupnorm import ResNet_GN
from layernorm import ResNet_LN
from instancenorm import ResNet_IN
from batchinstancenorm import ResNet_BIN
from defaultnorm import ResNet18 as ResNet
import os
from PIL import Image






# # Load pre-trained GloVe vectors (specify the dimensions, e.g., 100, 200, etc.)
glove = GloVe(name='6B', dim=100)

#Import your model class here

# Define command-line arguments
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on trained models')
    parser.add_argument('--model_file', type=str, help='Path to the trained model')
    parser.add_argument('--normalization', type=str, choices=['bn', 'in', 'bin', 'ln', 'gn', 'nn', 'inbuilt'],
                        help='Normalization scheme')
    parser.add_argument('--n', type=int, choices=[1, 2, 3], help='Value of n')
    parser.add_argument('--test_data_file', type=str, help='Path to the directory containing the test images')
    parser.add_argument('--output_file', type=str, help='Path to the output file')

    # Parse command-line arguments
    args = parser.parse_args()

    # Load the appropriate model based on normalization scheme and n value
    if args.normalization == 'bn':
        net = ResNet_BN(2, 25)
        weights_path = args.model_file
    elif args.normalization == 'in':
        weights_path = args.model_file
        net = ResNet_IN(2, 25)
    elif args.normalization == 'bin':
        weights_path = args.model_file
        net = ResNet_BIN(2, 25)
    elif args.normalization == 'ln':
        weights_path = args.model_file
        net = ResNet_LN(2, 25)
    elif args.normalization == 'gn':
        net = ResNet_GN(2, 25)
        weights_path = args.model_file
    elif args.normalization == 'nn':
        net = ResNet_NN(2, 25)
        weights_path = args.model_file
    elif args.normalization == 'inbuilt':
        weights_path = args.model_file
        net = ResNet(2, 25)
    else:
        raise ValueError("Normalization scheme not supported")

    # Load the model
    pretrained_weights = torch.load(weights_path, map_location=torch.device('cpu'))
    net.load_state_dict(pretrained_weights)
    net.eval()


    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_images = []
    for filename in os.listdir(args.test_data_file):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(args.test_data_file, filename)
            image = Image.open(image_path).convert('RGB')
            image = transform(image)
            test_images.append(image)

    batch_size = 32
    test_loader = torch.utils.data.DataLoader(test_images, batch_size=batch_size, shuffle=False, num_workers=2)

    # Perform inference
    predictions = []
    with torch.no_grad():
        for images in test_loader:
            outputs = net(images)
            predictions.extend(outputs.argmax(dim=1).tolist())


    # Write predictions to output file
    with open(args.output_file, 'w') as f:
        for pred in predictions:
            f.write(str(pred) + '\n')

    print("Inference completed. Predictions saved to", args.output_file)
    '''
    eg - python3 infer.py -model_file models/part_1.2_bn.pth --normalization bn --n 3 --test_data_file birds_test --output_file output.txt
    '''