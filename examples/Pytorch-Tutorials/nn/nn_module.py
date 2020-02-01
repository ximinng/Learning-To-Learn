# -*- coding: utf-8 -*-
"""
   Description :
    CNN:
        base unit: conv + pooling + batch norm + relu
            1. 1x1 conv : c_in -> c_out
    nn.Module:
        1. current layer
        2. Container:
        ```
        nn.Sequential(
            nn.Conv2d
            ...
        )
        ```
        3. parameters
        ```
        net.parameters() # 当前net所有参数
        ```
        4. nn.Module nested in nn.Module
        5. to(device)
            ```
            device = torch.device('cuda)
            net = Net()
            net.to(device)
            ```
        6. save and load
            1) load:
                net.load_state_dict(torch.load('ckpt.mdl'))
            2) save:
                torch.save(net.state_dict(), 'ckpt.mdl')
        7. train and test
            ```
            # train
            net.train()

            # test
            net.eval()
            ```
        8. implement own layer
            ```
            class Flatten(nn.Module):
            ```
        9. own linear layer
   Author :        xxm
"""
