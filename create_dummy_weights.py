import pickle
import os
from n_tuple_network import NTupleNetwork

# Create a network with optimistic initialization
network = NTupleNetwork(v_init=320000)

# Save the weights
os.makedirs("DRL/DRL_HW2/DRL-Assignment-2/weights", exist_ok=True)
network.save_weights("DRL/DRL_HW2/DRL-Assignment-2/weights/trained_weights.pkl")

print("Created dummy weights file with optimistic initialization.")
