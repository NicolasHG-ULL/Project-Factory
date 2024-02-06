import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Definir la arquitectura de la red neuronal
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Definir capas convolucionales
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        # Definir capas completamente conectadas
        self.fc1 = nn.Linear(128 * 11 * 11, 512)  # 11x11 debido a las capas de agrupación (pooling)
        self.fc2 = nn.Linear(512, 2)  # 2 clases

    def forward(self, x):
        # Aplicar funciones de activación ReLU y MaxPooling
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        # Aplanar la salida para la capa completamente conectada
        x = x.view(-1, 128 * 11 * 11)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        # Aplicar softmax para obtener las probabilidades de clase
        return nn.functional.softmax(x, dim=1)

# Preprocesamiento de imágenes y carga de datos
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Crear un objeto DataLoader para cargar los datos
train_dataset = datasets.ImageFolder(root='dataset_directory', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)

# Crear el modelo, la función de pérdida y el optimizador
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento del modelo
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:  # Imprimir estadísticas cada 100 lotes
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0

print('Entrenamiento completado')

# Guardar el modelo entrenado
torch.save(model.state_dict(), 'model.pth')

# Cargar el modelo entrenado
model = CNN()
model.load_state_dict(torch.load('model.pth'))