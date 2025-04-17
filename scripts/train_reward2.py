import torch
import torch.nn as nn
import torch.optim as optim
import json
import random

# --------------------------
# Reward Model
# --------------------------
class RewardModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim1=32, hidden_dim2=16, dropout_prob=0.2):
        super().__init__()
        # First hidden layer with increased size
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        # Second hidden layer
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        # Output layer producing a single scalar reward
        self.fc3 = nn.Linear(hidden_dim2, 1)

    def forward(self, x):
        # x: [batch_size, input_dim]
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x  # Output shape: [batch_size, 1]
# --------------------------
#
# --------------------------
def train_reward_model(reward_model, optimizer, train_data, val_data, num_epochs=20, device='cpu', patience=5, model_save_path='best_reward_model.pt'):
    """
    train_data, val_data: list of tuples, each tuple is (traj1, traj2, human_feedback)
       - traj1 and traj2 are lists of state (each state is a 5-D vector)
       - human_feedback: 1 means traj1 is preferred, 0 means traj2 is preferred.
       
    We use a pairwise logistic (BCE) loss based on the difference in mean rewards:
       d = (mean_reward(traj1) - mean_reward(traj2))
       p = sigmoid(d)
       loss = -[y * log(p + eps) + (1-y)*log(1-p + eps)]
       
    Early stopping: if validation loss does not improve for 'patience' consecutive epochs, stop training.
    """
    reward_model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    eps = 1e-8  # small epsilon to avoid log(0)
    
    for epoch in range(num_epochs):
        reward_model.train()
        total_train_loss = 0.0
        
        # Training phase
        for traj1, traj2, feedback in train_data:
            # Calculate mean reward for traj1
            rewards_traj1 = []
            for state in traj1:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                rewards_traj1.append(reward_model(state_tensor))
            R_traj1 = sum(rewards_traj1) / len(traj1)
            
            # Calculate mean reward for traj2
            rewards_traj2 = []
            for state in traj2:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                rewards_traj2.append(reward_model(state_tensor))
            R_traj2 = sum(rewards_traj2) / len(traj2)
            
            d_tensor = R_traj1 - R_traj2  # shape [1,1]
            d_tensor = d_tensor.squeeze()  # convert to scalar tensor
            target = torch.tensor(float(feedback), dtype=torch.float32, device=device)
            
            p = torch.sigmoid(d_tensor)
            loss = - ( target * torch.log(p + eps) + (1.0 - target) * torch.log(1.0 - p + eps) )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_data)
        
        # Validation phase
        reward_model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for traj1, traj2, feedback in val_data:
                rewards_traj1 = []
                for state in traj1:
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    rewards_traj1.append(reward_model(state_tensor))
                R_traj1 = sum(rewards_traj1) / len(traj1)
                
                rewards_traj2 = []
                for state in traj2:
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    rewards_traj2.append(reward_model(state_tensor))
                R_traj2 = sum(rewards_traj2) / len(traj2)
                
                d_tensor = R_traj1 - R_traj2
                d_tensor = d_tensor.squeeze()
                target = torch.tensor(float(feedback), dtype=torch.float32, device=device)
                p = torch.sigmoid(d_tensor)
                loss = - ( target * torch.log(p + eps) + (1.0 - target) * torch.log(1.0 - p + eps) )
                total_val_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {total_val_loss/len(val_data):.4f}")
        
        # Early stopping: if no improvement for 'patience' consecutive epochs, then stop.
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            torch.save(reward_model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch+1} with val loss {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}: No improvement for {patience} consecutive epochs.")
                break

# --------------------------
# 主流程：加载数据、划分训练集和验证集，训练 Reward Model
# --------------------------
if __name__ == "__main__":
    with open("training_data_lazy_2.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    # Format: List[(traj1, traj2, feedback)]
    full_data = []
    for item in json_data:
        traj1 = item["traj1"]
        traj2 = item["traj2"]
        feedback = item["feedback"]
        full_data.append((traj1, traj2, feedback))
    
    # split
    random.shuffle(full_data)
    split_index = int(0.8 * len(full_data))
    train_data = full_data[:split_index]
    val_data = full_data[split_index:]
    
    # initialize
    input_dim = 5
    reward_model = RewardModel(input_dim=input_dim)
    optimizer = optim.Adam(reward_model.parameters(), lr=0.01)
    
    print("Training reward model with human feedback...")
    train_reward_model(reward_model, optimizer, train_data, val_data, num_epochs=20, device='cpu', patience=5, model_save_path='best_reward_model.pt')
    print("Training finished.")
