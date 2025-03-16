import os

def readname():
    # filePath = 'D://Dataset/BSD500'
    # filePath = './pristine_images'
    filePath = 'D://Dataset/LOL/our485/high'

    # filePath = 'D://Dataset/MIT5K/train/03-Experts-C'
    # filePath = "D://BaiduNetdiskDownload/SIDD-train/train_512/Noisy"
    # filePath = "D://Dataset/MIT5K/train/valid/06-Input-ExpertC1.5"
    name = os.listdir(filePath)
    return name, filePath

if __name__ == "__main__":
    name, filePath = readname()
    print(name)
    txt = open("lol_high.txt", 'w')
    # txt = open("BSD432_test.txt", 'w')
    for i in name:
        # print(filePath + "/" + i)
        image_dir = os.path.join(filePath + "/", str(i))
        # image_dir = os.path.join('./pristine_images/', str(i))
        # image_dir = os.path.join('./flower_2000', str(i))
        txt.write(image_dir + "\n")
