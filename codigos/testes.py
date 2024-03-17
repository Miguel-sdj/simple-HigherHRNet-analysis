import os

for i in range(1, 5): # iterando entre os videos
    for j in range(0, 3): # iterando entre os modelos
        if j == 0: # modelo muito fraco mas rapido
            print("vou testar o modelo muito fraco no video: ", i)
            os.system(f'python scripts/live-demo.py --hrnet_c 32 -r 512 --hrnet_weights weights/pose_higher_hrnet_w32_512.pth --filename videos/video{i}.mp4')
        elif j == 1: # modelo fraco mas rapido
            print("vou testar o modelo fraco no video: ", i)
            os.system(f'python scripts/live-demo.py --hrnet_c 32 -r 640 --hrnet_weights weights/pose_higher_hrnet_w32_640.pth.tar --filename videos/video{i}.mp4')
        else: #modelo forte mas devagar
            print("vou testar o modelo forte no video: ", i)
            os.system(f'python scripts/live-demo.py --hrnet_c 48 -r 640 --hrnet_weights weights/pose_higher_hrnet_w48_640.pth.tar --filename videos/video{i}.mp4')


