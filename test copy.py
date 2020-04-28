
import matplotlib.pyplot as plt

import numpy as np

from tensorboardX import SummaryWriter

tb_writer = SummaryWriter('./GoodResult/logs/')


def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    import PIL.Image as Image
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    return image


if __name__ == '__main__':
    # a = np.random.rand(53, 53, 3)
    # plt.imshow(a)
    # # plt.show()
    # plt.savefig('./image/abc.png')
    # x = np.random.randn(10, 20)
    # figure = plt.figure()
    # plt.hist(x.reshape(-1), 10, density=1,rwidth=0.8)
    # plt.savefig('./12333333333333333333.png')
    # tb_writer.add_histogram('N',x)
    # tb_writer.flush()
    x = [[1,3,4]]
    print(x*3)
