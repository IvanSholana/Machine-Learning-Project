import numpy as np
import random

# Definisikan parameter Q-Learning
num_states = 9  # Total 3x3 grid
num_actions = 4  # Up, Down, Left, Right
q_table = np.zeros((num_states, num_actions))  # Q-Table

# Parameter Q-Learning
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
num_episodes = 1000  # Jumlah episode pelatihan

# Fungsi untuk mendapatkan state dari koordinat
def get_state(x, y):
    return x * 3 + y

# Fungsi untuk mendapatkan koordinat dari state
def get_coordinates(state):
    return divmod(state, 3)

# Fungsi untuk melakukan aksi dan mendapatkan reward
def step(state, action):
    x, y = get_coordinates(state)

    if action == 0:  # Up
        new_x, new_y = max(0, x - 1), y
    elif action == 1:  # Down
        new_x, new_y = min(2, x + 1), y
    elif action == 2:  # Left
        new_x, new_y = x, max(0, y - 1)
    elif action == 3:  # Right
        new_x, new_y = x, min(2, y + 1)

    new_state = get_state(new_x, new_y)
    
    # Reward jika mencapai tujuan
    if (new_x, new_y) == (2, 2):
        reward = 10
    else:
        reward = -1

    return new_state, reward

# Q-Learning
for episode in range(num_episodes):
    state = get_state(0, 0)  # Mulai dari (0, 0)
    
    while state != get_state(2, 2):  # Selama belum mencapai tujuan
        # Pilih aksi berdasarkan nilai Q tertinggi
        action = np.argmax(q_table[state])
        new_state, reward = step(state, action)

        # Pembaruan nilai Q
        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[new_state]) - q_table[state, action])
        
        print(new_state)
        state = new_state

# Menampilkan Q-Table
print("Q-Table setelah pelatihan:")
print(q_table)

# Menguji agen setelah pelatihan
state = get_state(0, 0)  # Mulai dari (0, 0)
path = [(0, 0)]  # Menyimpan path yang dilalui

while state != get_state(2, 2):
    action = np.argmax(q_table[state])  # Pilih aksi terbaik
    new_state, _ = step(state, action)
    path.append(get_coordinates(new_state))  # Simpan koordinat
    state = new_state

print("Path yang dilalui oleh agen:", path)
