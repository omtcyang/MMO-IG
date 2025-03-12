  from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
# resume_path = r'models/control_sd21_ini.ckpt'


batch_size = 24
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False
max_epochs = 40  # 设置最大训练轮数


if __name__ == '__main__':
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v21.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Misc
    dataset = MyDataset()
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq,save_dir="")
    
    # 设置训练轮数
    trainer = pl.Trainer(
        gpus=[0,1], 
        precision=32, 
        callbacks=[logger], 
        max_epochs=max_epochs  # 关键部分：设置最大训练轮数
    )

    # Train!
    trainer.fit(model, dataloader)