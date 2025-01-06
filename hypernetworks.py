import torch
import torch.nn as nn

# Load the pre-trained generator and discriminator
# generator = torch.load('F:\Final Year Project\Final_Year_Project\GeneratorModel.pth')
# discriminator = torch.load('F:\Final Year Project\Final_Year_Project\DiscriminatorModel.pth')

latent_dimension=128

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dimension, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Instantiate the generator model
generator = Generator()

# Load the state_dict
state_dict = torch.load('GeneratorModel.pth')

# Rename keys to match the model's keys
new_state_dict = {f'model.{k}': v for k, v in state_dict.items()}

# Load the modified state_dict into the generator model
generator.load_state_dict(new_state_dict)

# Calculate the output_dim
# output_dim = sum(p.numel() for p in generator.parameters())
# print(f"Total number of parameters in the generator: {output_dim}")

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Instantiate the discriminator model
discriminator = Discriminator()

# Load the state_dict into the model
state_dict = torch.load('F:\Final Year Project\Final_Year_Project\DiscriminatorModel.pth')
new_state_dict = {f'model.{k}': v for k, v in state_dict.items()}
discriminator.load_state_dict(new_state_dict)

# Ensure you have generated_masks defined
# Apply pruning to the discriminator
# apply_pruning(discriminator, generated_masks)



class Hypernetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(Hypernetwork, self).__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Define dimensions
input_dim = 100  # Latent vector size
hidden_dims = [256, 512]  # Example hidden layer sizes
output_dim = sum(p.numel() for p in generator.parameters())  # Total number of weights in the generator

# Instantiate hypernetwork
hypernet = Hypernetwork(input_dim, hidden_dims, output_dim)

def apply_pruning(model, generated_masks):
    index = 0
    for param in model.parameters():
        mask_size = param.numel()
        mask = generated_masks[:, index:index + mask_size].view_as(param)
        param.data *= mask
        index += mask_size

def proximal_gradient_descent(param, threshold):
    with torch.no_grad():
        param.abs_().sub_(threshold).clamp_(min=0).sign_().mul_(param.sign())

# Generate masks for pruning using a latent vector
latent_vector = torch.randn(1, input_dim)  # Single latent vector for simplicity
generated_masks = hypernet(latent_vector)

# Apply pruning to generator and discriminator
apply_pruning(generator, generated_masks)
apply_pruning(discriminator, generated_masks)

# Save the pruned generator and discriminator models
torch.save(generator.state_dict(), 'pruned_generator.pth')
torch.save(discriminator.state_dict(), 'pruned_discriminator.pth')

print("Pruned generator and discriminator models have been saved.")
