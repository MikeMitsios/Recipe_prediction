# # Sample text data
# corpus = {
#     "Chicken with potatoes":["chicken", "potato", "oil"],
#     "Beef with rice":["beef", "rice", "carrot", "oil"],
#     "Chiicken with tomato and potato":["chicken", "potato", "oil", "potato"],
    # Add more sentences as needed
# }

import torch
import torch.nn as nn
import torch.optim as optim

# Sample movie data
recipe_data = [
    {"title": "Chicken with potatoes", "description": ["chicken", "potato", "oil"]},
    {"title": "Beef with rice", "description": ["beef", "rice", "carrot", "oil"]},
    {"title": "Chiicken with tomato and potato", "description": ["chicken", "potato", "oil", "potato"]},
    # Add more movie data as needed
]

# Tokenize movie titles and descriptions
tokenized_titles = [movie["title"].split() for movie in recipe_data]
tokenized_descriptions = [movie["description"] for movie in recipe_data]

# Build a combined vocabulary
vocabulary = set(word for title in tokenized_titles + tokenized_descriptions for word in title)
word_to_index = {word: idx for idx, word in enumerate(vocabulary)}
index_to_word = {idx: word for idx, word in enumerate(vocabulary)}

# Convert text data to numerical indices
numerical_titles = [[word_to_index[word] for word in title] for title in tokenized_titles]
numerical_descriptions = [[word_to_index[word] for word in desc] for desc in tokenized_descriptions]

# Define the MovieEmbedding model
class MovieEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(MovieEmbeddingModel, self).__init__()
        self.title_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.description_embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, title, description):
        title_embedding = self.title_embedding(title)
        description_embedding = self.description_embedding(description)
        return title_embedding, description_embedding

# Hyperparameters
embedding_dim = 50
learning_rate = 0.01
epochs = 100

# Initialize the model, loss function, and optimizer
model = MovieEmbeddingModel(vocab_size=len(vocabulary), embedding_dim=embedding_dim)
criterion = nn.CosineEmbeddingLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    total_loss = 0
    for idx, movie in enumerate(recipe_data):
        title_tensor = torch.LongTensor(numerical_titles[idx])
        description_tensor = torch.LongTensor(numerical_descriptions[idx])

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        title_embedding, description_embedding = model(title_tensor, description_tensor)

        # Calculate the loss (using cosine similarity as a measure)
        target = torch.tensor([1.0])  # Positive pair, assuming titles and descriptions are related
        loss = criterion(title_embedding, description_embedding, target)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Print average loss for the epoch
    print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {total_loss / len(recipe_data)}')

# Retrieve the learned embeddings for movie titles and descriptions
learned_title_embeddings = model.title_embedding.weight.data.numpy()
learned_description_embeddings = model.description_embedding.weight.data.numpy()

# Now, you can use these embeddings for various tasks, such as similarity matching or recommendation systems.