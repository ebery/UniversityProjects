
# coding: utf-8

# In[ ]:




# In[1]:

import numpy as np
import skimage.color as sm
from scipy.misc import imread as imread, imsave as imsave
from scipy.signal import convolve2d as convolve2d
import matplotlib.pyplot as plt
from sol2 import *


# In[1]:


im = read_image("monkey.jpg",1)
s1 = 1
s2 = 1
s3 = 1
if s1:
    dut_funcs = [DFT,IDFT,DFT2,IDFT2,DFT2,IDFT2]
    te_funcs = [np.fft.fft,np.fft.ifft,np.fft.fft2,np.fft.ifft2,np.fft.fft2,np.fft.ifft2]
    func_strs = ["DFT","IDFT","DFT2","IDFT2","DFT2_rand","IDFT2_rand"]
    rand_cplx = lambda c,r: (np.random.randint(255,size=(c,r)) + 1j*np.random.randint(255,size=(c,r)))/255
    sigs = [rand_cplx(1000,1), rand_cplx(1000,1), im, np.fft.fft2(im),rand_cplx(150,500),rand_cplx(150,500)]

    print("## starting DFT/IDFT/DFT2/IDFT2 test ##")
    for i in range(len(dut_funcs)):
        print("Checking ", func_strs[i], "...")
        sig = sigs[i]
        if i in [0,1]:
            error = np.max(np.abs(te_funcs[i](sig.reshape(1,sig.size)).reshape(sig.size,1)-dut_funcs[i](sig)))
        else:
            error = np.max(np.abs(te_funcs[i](sig)-dut_funcs[i](sig)))
        assert error<1e-7, "In function %s, max error allowed is 1e-7 and error is %e" % (func_strs[i],error)
    print("## test done ##\n")
if s2:
    print("## Starting deriv test ##")
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(conv_der(im),cmap="gray")
    plt.axis("off")
    plt.title("spatial deriv")
    plt.subplot(1,2,2)
    plt.imshow(np.log(1+conv_der(im)),cmap="gray")
    plt.axis("off")
    plt.title("log(1+spatial deriv)")
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(fourier_der(im),cmap="gray")
    plt.axis("off")
    plt.title("fourier deriv")
    plt.subplot(1,2,2)
    plt.imshow(np.log(1+fourier_der(im)),cmap="gray")
    plt.axis("off")
    plt.title("log(1+fourier deriv)")
    plt.show()
    print("## test done - check results visually ##")
if s3:
    print("## Starting gaussian test ##")
    print("Checking kernel generation ...")
    kernel3 = np.array([[ 0.0625,  0.125,   0.0625], [ 0.125,   0.25,    0.125 ], [ 0.0625,  0.125,   0.0625]])
    kernel11 = np.array([[  9.53674316e-07,   9.53674316e-06,   4.29153442e-05,           1.14440918e-04,   2.00271606e-04,   2.40325928e-04,           2.00271606e-04,   1.14440918e-04,   4.29153442e-05,           9.53674316e-06,   9.53674316e-07],        [  9.53674316e-06,   9.53674316e-05,   4.29153442e-04,           1.14440918e-03,   2.00271606e-03,   2.40325928e-03,           2.00271606e-03,   1.14440918e-03,   4.29153442e-04,           9.53674316e-05,   9.53674316e-06],        [  4.29153442e-05,   4.29153442e-04,   1.93119049e-03,           5.14984131e-03,   9.01222229e-03,   1.08146667e-02,           9.01222229e-03,   5.14984131e-03,   1.93119049e-03,           4.29153442e-04,   4.29153442e-05],        [  1.14440918e-04,   1.14440918e-03,   5.14984131e-03,           1.37329102e-02,   2.40325928e-02,   2.88391113e-02,           2.40325928e-02,   1.37329102e-02,   5.14984131e-03,           1.14440918e-03,   1.14440918e-04],        [  2.00271606e-04,   2.00271606e-03,   9.01222229e-03,           2.40325928e-02,   4.20570374e-02,   5.04684448e-02,           4.20570374e-02,   2.40325928e-02,   9.01222229e-03,           2.00271606e-03,   2.00271606e-04],        [  2.40325928e-04,   2.40325928e-03,   1.08146667e-02,           2.88391113e-02,   5.04684448e-02,   6.05621338e-02,           5.04684448e-02,   2.88391113e-02,   1.08146667e-02,           2.40325928e-03,   2.40325928e-04],        [  2.00271606e-04,   2.00271606e-03,   9.01222229e-03,           2.40325928e-02,   4.20570374e-02,   5.04684448e-02,           4.20570374e-02,   2.40325928e-02,   9.01222229e-03,           2.00271606e-03,   2.00271606e-04],        [  1.14440918e-04,   1.14440918e-03,   5.14984131e-03,           1.37329102e-02,   2.40325928e-02,   2.88391113e-02,           2.40325928e-02,   1.37329102e-02,   5.14984131e-03,           1.14440918e-03,   1.14440918e-04],        [  4.29153442e-05,   4.29153442e-04,   1.93119049e-03,           5.14984131e-03,   9.01222229e-03,   1.08146667e-02,           9.01222229e-03,   5.14984131e-03,   1.93119049e-03,           4.29153442e-04,   4.29153442e-05],        [  9.53674316e-06,   9.53674316e-05,   4.29153442e-04,           1.14440918e-03,   2.00271606e-03,   2.40325928e-03,           2.00271606e-03,   1.14440918e-03,   4.29153442e-04,           9.53674316e-05,   9.53674316e-06],        [  9.53674316e-07,   9.53674316e-06,   4.29153442e-05,           1.14440918e-04,   2.00271606e-04,   2.40325928e-04,           2.00271606e-04,   1.14440918e-04,   4.29153442e-05,           9.53674316e-06,   9.53674316e-07]])    
    
    
    assert np.all(get_gaussian_kernel(3)-kernel3<1e-10), "Kernel is " + str(get_gaussian_kernel(3)) + ", should be " + str(kernel3)
    assert np.all(get_gaussian_kernel(11)-kernel11<1e-10), "Kernel is " + str(get_gaussian_kernel(11)) + ", should be " + str(kernel11)
    for i in range(50):
        kernel_size = i*2 + 1
        kernel = get_gaussian_kernel(kernel_size)
        assert kernel.shape[0] == kernel_size and kernel.shape[1] == kernel_size, "Kernel size is %d, should be %d" % (kernel.shape[0],kernel_size)
    kernels = [1,7,15]
    funcs = [blur_spatial, blur_fourier]
#    funcs = [blur_fourier]
    funcs_names = ["blur_spatial","blur_fourier"]
    for j in range(len(funcs)):
        plt.figure()
        for i in range(len(kernels)):
            plt.subplot(1,len(kernels),i+1)
            plt.imshow(funcs[j](im,kernels[i]),cmap="gray")
            plt.axis("off")
            plt.title("%s, k=%d" % (funcs_names[j], kernels[i]))
        plt.show()
    
    print("## test done - check results visually ##")


# In[ ]:



