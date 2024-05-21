# Script created by PeterC to learn Tensorboard 21-05-2024
# Documentation references: https://www.tensorflow.org/tensorboard/get_started, https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
# Tensorboard is a visualization toolkit for machine learning, mainly enabling tracking and visualization of metrics (loss and accuracy).

# %% Import modules
import torch 
from torch.utils.tensorboard import SummaryWriter # Key class to use tensorboard with PyTorch. VSCode will automatically ask if you want to load tensorboard in the current session.

# Define a writer
writer = SummaryWriter(log_dir='training_logs') # By default, this will write in a folder names "runs" in the directory of the main script. Else change providing path as first input.
# Note that Writer actually writes event files then loaded by Tensorboard.

# Example of usage of tensorboard from LDC main script:
#        from torch.utils.tensorboard import SummaryWriter # for torch 1.4 or greather
#        tb_writer = SummaryWriter(log_dir=training_dir)
#        # saving training settings
#        training_notes =['LDC, Xavier Normal Init, LR= ' + str(args.lr) + ' WD= '
#                          + str(args.wd) + ' image size = ' + str(args.img_width)
#                          + ' adjust LR=' + str(args.adjust_lr) +' LRs= '
#                          + str(args.lrs)+' Loss Function= CAST-loss2.py '
#                          + str(time.asctime())+args.version_notes]
#        info_txt = open(os.path.join(training_dir, 'training_settings.txt'), 'w')
#        info_txt.write(str(training_notes))
#        info_txt.close()

# To save a scalar value such as the loss at a given epoch:
loss = 10
epoch = 0
writer.add_scalar("Loss/train", loss, epoch) # In LDC this all that is done, simply adding the scalar loss metric at a given epoch to the writer and then flushing it.
                                             # The first input is simply a tag for the visualization.
# Then call flush to make sure all events are logged to disk:
writer.flush() 

# Once events are logged in a folder. Tensorboard must be enabled and pointed to the directory where the events are located:
# In command line: tensorboard --logdir=<log_dir> where events have been saved. By default this will open a localhost server at post 6006: http://localhost:6006/
# But VScode provides visualization integrated in the application.