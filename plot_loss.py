import pickle
import matplotlib.pyplot as plt

# To Load a file

with open('to_plot.pkl', 'rb') as handle:

    to_plot = pickle.load(handle)

plt.subplot(111)
plt.plot(to_plot['epoch_ticks'], to_plot['train_losses'], label = "Training Loss")
plt.plot(to_plot['epoch_ticks'], to_plot['val_losses'], label = "Validation Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(to_plot['epoch_  ticks'])

plt.tight_layout()

plt.savefig('Loss_Plot_SGPlus.png')
