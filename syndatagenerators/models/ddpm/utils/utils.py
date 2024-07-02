import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
import os
from syndatagenerators.metrics.visualization import t_SNE
from syndatagenerators.models.ddpm.utils.Generator import Generator
import numpy as np
def safeImage(image, cmap=None, axis="on", path="path_not_provided.png"):
    plt.imshow(image[0], cmap=cmap)
    plt.axis(axis)
    plt.savefig(path)
    plt.close()


def loadModel(model, path):
    """Loads the state of a befor saved network

    Args:
        model (Neural Network): Neural net of the same type you want to load the state to
        path (String): Location of the file you want to load from
    """
    model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
    model.eval()


def showForward(ddpm, image, device, n_epochs,TAG, values=[.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]):
    fig = plt.figure(figsize=(20, 4))
    col, row = 11, 1

    fig.add_subplot(row, col, 1)
    plt.axis("off")
    plt.plot(image)

    image = torch.reshape(image, [1, 48]).to(device)
    noise = torch.randn_like(image).to(device)
    for i, percent in enumerate(values):
        t = torch.ones((1, 1), dtype=torch.int64).to(device)
        t = t * (int(ddpm.n_steps*percent)-1)
        noisedImage = ddpm.addNoise(image, t, noise)
        noisedImage = noisedImage.detach().cpu().numpy()
        fig.add_subplot(row, col, i+2)
        plt.axis("off")
        plt.plot(noisedImage[0])
    plt.savefig(
        f"../../../figures/forward/{TAG}.png")
    plt.close()

def showForward2(ddpm, image, device,TAG, values=[.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]):
    
    plt.plot(image, label=f't: {0}, mean: {image.mean():.3f}, std: {image.std():.3f}',)

    image = torch.reshape(image, [1, 48]).to(device)
    noise = torch.randn_like(image).to(device)
    for i, percent in enumerate(values):
        t = torch.ones((1, 1), dtype=torch.int64).to(device)
        t = t * (int(ddpm.n_steps*percent)-1)
        noisedImage = ddpm.addNoise(image, t, noise)
        noisedImage = noisedImage.detach().cpu().numpy()
        
        plt.plot(noisedImage[0], label=f't: {t[0].item()}, mean: {noisedImage[0].mean():.3f}, std: {noisedImage[0].std():.3f}',alpha=0.3)
    plt.legend()
    plt.savefig(
        f"../../../figures/forward/{TAG}.png")
    plt.close()


def showHist(ddpm, image, device,TAG):
    image = torch.reshape(image, [1, 48]).to(device)
    noise = torch.randn_like(image).to(device)
    t = torch.ones((1, 1), dtype=torch.int64).to(device)
    t = t * (ddpm.n_steps-1)
    noisedImage = ddpm.addNoise(image, t, noise)
    noisedImage = noisedImage.detach().cpu().numpy()
    plt.hist(noisedImage[0])
    plt.savefig(
        f"../../../figures/forward/hist/{TAG}.png")
    plt.close()


def writeData(size, key, generator: Generator,times,TAG):
    for i in range(times):
        print(f"Generating {i*size}")
        genSet = generator.generate_n_cond(size)
        genSet = genSet.squeeze().cpu().detach().numpy()
        with open(f'../../../outputs/{key}/{TAG}.txt', "ab") as f:
            np.savetxt(f, genSet)

def writeData_multivar(size,generator: Generator,store_path,train_size,gen_size,times,cats,conts):
    for i in range(times):
        print(f"Generating {i*size}")
        genSet = generator.generate_n_cond_multivar(size,train_size,gen_size,cats[i*size:(i*size+size)],conts[i*size:(i*size+size)])
        genSet = genSet.squeeze().cpu().detach().numpy()
        with open(f'{store_path}5ksamples.txt', "ab") as f:
            np.savetxt(f, genSet)


def checkTSNE(realSet, fakeSet, store_path):
    sampleReal, sampleFake = t_SNE(realSet, fakeSet)
    fig, ax = plt.subplots(figsize=(16, 16), dpi=200)

    ax.plot(sampleReal[:, 0], sampleReal[:, 1],'r+', alpha=0.2, markersize=1,label='real')
    ax.plot(sampleFake[:, 0], sampleFake[:, 1], 'bo', alpha=0.2, markersize=1,label='generated')
    plt.xlabel("embeding-dim 1")
    plt.ylabel("embeding-dim 2")
    plt.legend()
    plt.title(f'{realSet.shape[0]} t-SNE samples ')
    plt.savefig(f'{store_path}tsne.png')
    plt.close()
    plt.hist2d(sampleReal[:, 0], sampleReal[:, 1], bins=(
        160, 160), cmap=plt.cm.jet, vmin=0.0, vmax=10.0)
    plt.colorbar()
    plt.title(f"real t-SNE samples ")
    plt.xlabel("embeding-dim 1")
    plt.ylabel("embeding-dim 2")
    plt.savefig(f'{store_path}real_histo.png')
    plt.close()
    plt.hist2d(sampleFake[:, 0], sampleFake[:, 1], bins=(
        160, 160), cmap=plt.cm.jet, vmin=0.0, vmax=10.0)
    plt.colorbar()
    plt.title(f"generated t-SNE samples ")
    plt.xlabel("embeding-dim 1")
    plt.ylabel("embeding-dim 2")
    plt.savefig(f'{store_path}fake_histo.png')
    plt.close()
    return sampleReal, sampleFake


def checkCovMat(sampleSet, genSet, store_path,size):
    fig = plt.figure(figsize=(16, 9))

    convMatReal = torch.cov(sampleSet.view(-1, size).T)
    convMatGen = torch.cov(genSet.view(-1, size).T)

    grid = ImageGrid(
        fig, 111,  # similar to fig.add_subplot(142).
        nrows_ncols=(1, 2), axes_pad=0.2, label_mode="L",
        cbar_location="right", cbar_mode="single")

    im = grid[0].imshow(convMatReal, cmap=plt.cm.coolwarm,)
    im = grid[1].imshow(convMatGen, cmap=plt.cm.coolwarm,)
    grid.cbar_axes[0].colorbar(im)
    plt.title("covmat")
    plt.savefig(f'{store_path}covmat.png')
    plt.close()

def buildKeyList():
   
    keyList = []
    for value in range(2, 122, 1):
        if(value < 10):
            keyList.append(f"MAC00000{value}")
        elif (value < 100):
            keyList.append(f"MAC0000{value}")
        else:
            keyList.append(f"MAC000{value}")
    return keyList

def createDirs(key,TAG,path):
    path = os.path.join(path, key)
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)

    path2 = os.path.join(path, TAG)
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)
    try:
        os.mkdir(path2)
    except OSError as error:
        print(error)

def saveConditionsTxt(store_path,cats,conts):
    with open(f'{store_path}conditions.txt', 'w') as fp:
        for item in cats:
            # write each item on a new line
            fp.write("%s\n" % item)
        for item in conts:
            # write each item on a new line
            fp.write("%s\n" % item)